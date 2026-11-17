"""
Batch 2: Global Provider Scheduler & Lifecycle Tracker.
Manages local admission queues, explicit provider token/request quota reservations,
provider concurrency, and overlapping shared quota cooldowns.
Executes the three configured retry strategies (none, naive, aware).
"""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from src.groq_client import ProviderCallResult, GroqCompatibleClient
from src.logger import get_logger


class RequestContext:
    def __init__(self, request_id: str, start_perf: float, deadline_perf: float):
        self.request_id = request_id
        self.start_perf = start_perf
        self.deadline_perf = deadline_perf
        
        self.local_queue_wait_ms = 0.0
        self.primary_provider_queue_wait_ms = 0.0
        self.retry_queue_wait_ms = 0.0
        self.failed_attempt_work_ms = 0.0
        self.retry_backoff_ms = 0.0
        self.correction_work_ms = 0.0
        
        self.transport_retries = 0
        self.correction_attempts = 0
        self.client_abandoned = False
        self.call_traces: List[Dict[str, Any]] = []


@dataclass
class _QuotaReservation:
    id: int
    timestamp: float
    est_tokens: int
    actual_tokens: Optional[int] = None
    cancelled: bool = False


class QuotaManager:
    """Rolling window quota tracking with mutable reservations."""
    def __init__(self, rpm: int, tpm: int, rpd: int, tpd: int):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.tpd = tpd
        
        self.req_minute: List[_QuotaReservation] = []
        self.req_day: List[_QuotaReservation] = []
        self._next_id = 1

    def reserve(self, tokens: int) -> Tuple[Optional[int], str]:
        now = time.time()
        
        # Evict expired reservations
        self.req_minute = [r for r in self.req_minute if not r.cancelled and (now - r.timestamp <= 60.0)]
        self.req_day = [r for r in self.req_day if not r.cancelled and (now - r.timestamp <= 86400.0)]
            
        cur_rpm = len(self.req_minute)
        cur_tpm = sum(r.actual_tokens if r.actual_tokens is not None else r.est_tokens for r in self.req_minute)
        
        cur_rpd = len(self.req_day)
        cur_tpd = sum(r.actual_tokens if r.actual_tokens is not None else r.est_tokens for r in self.req_day)
        
        if self.tpd > 0 and cur_tpd + tokens > self.tpd: 
            return None, "tpd_limit"
        if self.rpd > 0 and cur_rpd + 1 > self.rpd: 
            return None, "rpd_limit"
        if self.tpm > 0 and cur_tpm + tokens > self.tpm: 
            return None, "tpm_limit"
        if self.rpm > 0 and cur_rpm + 1 > self.rpm: 
            return None, "rpm_limit"
        
        res_id = self._next_id
        self._next_id += 1
        
        res = _QuotaReservation(id=res_id, timestamp=now, est_tokens=tokens)
        self.req_minute.append(res)
        self.req_day.append(res)
        return res_id, ""

    def reconcile(self, res_id: int, actual_tokens: int):
        for r in self.req_minute:
            if r.id == res_id:
                r.actual_tokens = actual_tokens
        for r in self.req_day:
            if r.id == res_id:
                r.actual_tokens = actual_tokens

    def cancel(self, res_id: int):
        for r in self.req_minute:
            if r.id == res_id:
                r.cancelled = True
        for r in self.req_day:
            if r.id == res_id:
                r.cancelled = True


class ProviderScheduler:
    def __init__(self, config: Dict[str, Any], client: GroqCompatibleClient):
        self.config = config
        self.client = client
        self.logger = get_logger("scheduler")
        
        pipe_cfg = config.get("pipeline", {})
        prov_cfg = config.get("provider", {})
        self.retry_strategy = pipe_cfg.get("retry_strategy", "aware")
        
        self.max_retries = int(prov_cfg.get("max_transport_retries", 2))
        self.base_delay = float(prov_cfg.get("retry_base_delay_ms", 400.0))
        self.max_delay = float(prov_cfg.get("retry_max_delay_ms", 4000.0))
        
        self.max_local_queue = int(pipe_cfg.get("max_local_queue_depth", 500))
        self.local_concurrency = asyncio.Semaphore(int(pipe_cfg.get("max_local_concurrency", 100)))
        self.provider_concurrency = asyncio.Semaphore(int(prov_cfg.get("max_provider_concurrency", 10)))
        
        # We explicitly identify these as local limits; real provider headers update behavior via 429 processing
        self.quota = QuotaManager(
            rpm=int(prov_cfg.get("limit_rpm", 30)),
            tpm=int(prov_cfg.get("limit_tpm", 15000)),
            rpd=int(prov_cfg.get("limit_rpd", 14400)),
            tpd=int(prov_cfg.get("limit_tpd", 500000))
        )
        
        self.shared_cooldown = asyncio.Event()
        self.shared_cooldown.set()
        self.cooldown_expires_at = 0.0
        
        self.local_queue_depth = 0
        self.provider_queue_depth = 0
        self.in_flight = 0
        self.peak_local = 0
        self.peak_provider = 0
        self.peak_in_flight = 0

    async def acquire_admission(self, ctx: RequestContext) -> bool:
        if self.local_queue_depth >= self.max_local_queue:
            return False
            
        self.local_queue_depth += 1
        self.peak_local = max(self.peak_local, self.local_queue_depth)
        t0 = time.perf_counter()
        
        try:
            rem_s = ctx.deadline_perf - time.perf_counter()
            if rem_s <= 0:
                self.local_queue_depth -= 1
                ctx.local_queue_wait_ms += (time.perf_counter() - t0) * 1000.0
                ctx.client_abandoned = True
                return False
                
            await asyncio.wait_for(self.local_concurrency.acquire(), timeout=rem_s)
            self.local_queue_depth -= 1
            ctx.local_queue_wait_ms += (time.perf_counter() - t0) * 1000.0
            return True
        except (asyncio.TimeoutError, TimeoutError):
            self.local_queue_depth -= 1
            ctx.local_queue_wait_ms += (time.perf_counter() - t0) * 1000.0
            ctx.client_abandoned = True
            return False

    def release_admission(self, ctx: RequestContext):
        self.local_concurrency.release()

    async def _reset_cooldown(self, target_exp: float):
        now = time.perf_counter()
        if target_exp > now:
            await asyncio.sleep(target_exp - now)
        if time.perf_counter() >= self.cooldown_expires_at:
            self.shared_cooldown.set()

    async def execute_call(
        self,
        ctx: RequestContext,
        messages: List[Dict[str, str]],
        stage: str,
        model_id: str,
        max_tokens: int,
        reserve_after_ms: float,
        fault_inject: Optional[str] = None
    ) -> Tuple[ProviderCallResult, float]:
        attempt = 0
        primary_work = 0.0
        from src.chunking import count_tokens
        est_tokens = sum(count_tokens(m.get("content", "")) + 4 for m in messages) + max_tokens
        
        while True:
            # 1. Pre-Queue Deadline Verification
            now_perf = time.perf_counter()
            rem_ms = (ctx.deadline_perf - now_perf) * 1000.0
            allowed_call_ms = rem_ms - reserve_after_ms
            
            if allowed_call_ms <= 50.0:
                res = ProviderCallResult(
                    success=False, stage=stage, model_id=model_id, http_status=0,
                    content="", provider_first_token_ms=None, work_ms=0.0, tokens_in=0, tokens_out=0,
                    accounting_mode="none", error_code="deadline_exceeded",
                    error_message="Insufficient remaining deadline before dispatch.",
                    failure_reason="request_deadline_exceeded", is_retriable=False
                )
                ctx.call_traces.append(res.to_trace_dict(len(ctx.call_traces)+1))
                ctx.client_abandoned = True
                return res, 0.0

            # 2. Local Quota Pre-Reservation (Simulated Limits)
            res_id, q_reason = self.quota.reserve(est_tokens)
            if res_id is None:
                # Daily limit exhaustion is immediately terminal
                if "tpd" in q_reason or "rpd" in q_reason or self.retry_strategy in ("none", "naive"):
                    res = ProviderCallResult(
                        success=False, stage=stage, model_id=model_id, http_status=429,
                        content="", provider_first_token_ms=None, work_ms=0.0, tokens_in=0, tokens_out=0,
                        accounting_mode="none", error_code="local_quota",
                        error_message=f"Local simulated quota rejected request ({q_reason}).",
                        failure_reason=q_reason, is_retriable=False
                    )
                    ctx.call_traces.append(res.to_trace_dict(len(ctx.call_traces)+1))
                    return res, 0.0
                
                # 'aware' strategy explicitly yields/waits for local per-minute capacity
                await asyncio.sleep(1.0)
                ctx.retry_backoff_ms += 1000.0
                continue

            wait_t0 = time.perf_counter()
            if self.retry_strategy == "aware":
                await self.shared_cooldown.wait()
            
            # 3. Provider Semaphore Wait & Re-calculating Deadline
            self.provider_queue_depth += 1
            self.peak_provider = max(self.peak_provider, self.provider_queue_depth)
            
            try:
                rem_wait = (ctx.deadline_perf - time.perf_counter()) - (reserve_after_ms / 1000.0)
                if rem_wait <= 0:
                    raise TimeoutError("queue_expired_deadline")
                await asyncio.wait_for(self.provider_concurrency.acquire(), timeout=rem_wait)
            except (asyncio.TimeoutError, TimeoutError):
                self.provider_queue_depth -= 1
                self.quota.cancel(res_id)  # Cancel reservation before it dispatches
                
                q_wait = (time.perf_counter() - wait_t0) * 1000.0
                if attempt == 0 and stage in ("generate", "verify"):
                    ctx.primary_provider_queue_wait_ms += q_wait
                else:
                    ctx.retry_queue_wait_ms += q_wait
                ctx.client_abandoned = True
                
                res = ProviderCallResult(
                    success=False, stage=stage, model_id=model_id, http_status=0,
                    content="", provider_first_token_ms=None, work_ms=0.0, tokens_in=0, tokens_out=0,
                    accounting_mode="none", error_code="queue_timeout",
                    error_message="Expired in provider queue.", failure_reason="queue_expired_deadline",
                    is_retriable=False
                )
                ctx.call_traces.append(res.to_trace_dict(len(ctx.call_traces)+1))
                return res, 0.0

            self.provider_queue_depth -= 1
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            
            q_wait = (time.perf_counter() - wait_t0) * 1000.0
            if attempt == 0 and stage in ("generate", "verify"):
                ctx.primary_provider_queue_wait_ms += q_wait
            else:
                ctx.retry_queue_wait_ms += q_wait

            # 4. Immediate pre-dispatch deadline verification (protecting against queue-sleep staleness)
            rem_ms = (ctx.deadline_perf - time.perf_counter()) * 1000.0
            allowed_call_ms = rem_ms - reserve_after_ms
            
            # 5. Execute HTTP Request with strict cancellation safety
            try:
                res = await self.client.execute_single_call(
                    messages=messages, stage=stage, model_id=model_id, max_tokens=max_tokens, 
                    call_timeout_ms=allowed_call_ms, fault_inject=fault_inject
                )
            finally:
                self.in_flight -= 1
                self.provider_concurrency.release()
            
            # Reconcile exact consumed tokens over the initial reservation
            self.quota.reconcile(res_id, res.tokens_in + res.tokens_out)
            ctx.call_traces.append(res.to_trace_dict(len(ctx.call_traces)+1, wait_ms=q_wait))
            
            if res.success:
                if stage in ("regenerate", "reverify"):
                    ctx.correction_work_ms += res.work_ms
                else:
                    primary_work = res.work_ms
                return res, primary_work
            
            ctx.failed_attempt_work_ms += res.work_ms
            
            # 6. Retry Logic Analysis (Restricted to Transient Transport/Network failures)
            if self.retry_strategy == "none" or not res.is_retriable or attempt >= self.max_retries:
                if "tpd" in res.failure_reason or "rpd" in res.failure_reason:
                    pass
                if "timeout" in res.failure_reason or "deadline" in res.failure_reason:
                    ctx.client_abandoned = True
                return res, 0.0
                
            if res.retry_after_ms and res.retry_after_ms > 0:
                delay_ms = res.retry_after_ms + random.uniform(25.0, 100.0)
            else:
                exp_ms = min(self.max_delay, self.base_delay * (2 ** attempt))
                delay_ms = exp_ms + random.uniform(0.0, 0.25 * exp_ms)
                
            if self.retry_strategy == "aware":
                if res.http_status == 429:
                    exp_target = time.perf_counter() + (delay_ms / 1000.0)
                    if exp_target > self.cooldown_expires_at:
                        self.cooldown_expires_at = exp_target
                        self.shared_cooldown.clear()
                        asyncio.create_task(self._reset_cooldown(exp_target))
                
                if ((ctx.deadline_perf - time.perf_counter()) * 1000.0) < (delay_ms + reserve_after_ms + 50.0):
                    ctx.client_abandoned = True
                    return res, 0.0
            
            attempt += 1
            ctx.transport_retries += 1
            
            sleep_t0 = time.perf_counter()
            await asyncio.sleep(delay_ms / 1000.0)
            ctx.retry_backoff_ms += (time.perf_counter() - sleep_t0) * 1000.0
