"""
Batch 2 / Batch 3: OpenAI-compatible Groq/Stand-In SSE client with X-Relay-Fault headers.
Executes one single HTTP attempt per call, enforces monotonic request-wide deadlines
propagated from RequestContext, classifies Groq 429 dimensions accurately, discards
incomplete partial streams on disconnect, and records stage-aware retry diagnostics.
"""

import asyncio
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.chunking import count_tokens
from src.logger import get_logger, redact_dict

RATELIMIT_HEADER_KEYS = (
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)

CURRENT_REQUEST_SCOPE: ContextVar[Optional[Tuple[str, float]]] = ContextVar(
    "relay_current_request_scope", default=None
)


def start_request_scope(request_id: str, deadline_mono: float) -> Token:
    """Binds a logical request ID and absolute monotonic deadline to the current async task."""
    return CURRENT_REQUEST_SCOPE.set((request_id, deadline_mono))


def end_request_scope(token: Token) -> None:
    """Restores the previous async request scope during final request cleanup."""
    CURRENT_REQUEST_SCOPE.reset(token)


def parse_reset_duration_ms(val: Optional[str]) -> Optional[float]:
    if not val:
        return None
    s = str(val).strip().lower()
    try:
        return float(s) * 1000.0
    except ValueError:
        pass
    total_ms = 0.0
    matched = False
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", s):
        matched = True
        num = float(amount)
        if unit == "ms":
            total_ms += num
        elif unit == "s":
            total_ms += num * 1000.0
        elif unit == "m":
            total_ms += num * 60000.0
        elif unit == "h":
            total_ms += num * 3600000.0
    return round(total_ms, 2) if matched else None


def extract_request_id_from_fault(fault_inject: Optional[str]) -> Optional[str]:
    if not fault_inject:
        return None
    for part in str(fault_inject).split(";"):
        part = part.strip()
        if part.startswith("req_id="):
            val = part.split("=", 1)[1].strip()
            return val if val else None
    return None


def minimum_attempt_budget_ms(stage: str, failure_reason: str, prov_cfg: Dict[str, Any]) -> float:
    """
    Computes the minimum viable execution budget (ms) required for the next provider attempt
    in the given stage, conditioned on the observed failure classification.
    """
    min_stream_work_ms = float(prov_cfg.get("min_stream_work_ms", 1200.0))
    if failure_reason == "ttft_timeout":
        ttft_ms = float(prov_cfg.get("ttft_timeout_s", 15.0)) * 1000.0
        return ttft_ms + min_stream_work_ms
    if failure_reason == "inter_chunk_timeout":
        inter_chunk_ms = float(prov_cfg.get("inter_chunk_timeout_s", 8.0)) * 1000.0
        return inter_chunk_ms + min_stream_work_ms
    if stage in ("verify", "reverify"):
        return float(prov_cfg.get("min_verify_attempt_ms", 1500.0))
    return float(prov_cfg.get("min_gen_attempt_ms", 2000.0))


def required_downstream_reserve_ms(stage: str, config: Dict[str, Any]) -> float:
    """
    Computes the time reserve (ms) required for all downstream pipeline work
    after the current stage completes (deterministic verify, semantic verify, delivery).
    """
    pipe_cfg = config.get("pipeline", {})
    det_verify_ms = float(pipe_cfg.get("det_verify_reserve_ms", 100.0))
    sem_verify_ms = float(pipe_cfg.get("sem_verify_reserve_ms", 2000.0))
    delivery_ms = float(pipe_cfg.get("delivery_reserve_ms", 500.0))
    if stage in ("generate", "regenerate"):
        return det_verify_ms + sem_verify_ms + delivery_ms
    return delivery_ms


@dataclass
class ProviderCallResult:
    success: bool
    stage: str
    model_id: str
    http_status: int
    content: str
    provider_first_token_ms: Optional[float]
    work_ms: float
    tokens_in: int
    tokens_out: int
    accounting_mode: str
    error_code: str = ""
    error_message: str = ""
    failure_reason: str = "none"
    is_retriable: bool = False
    retry_after_ms: Optional[float] = None
    ratelimit_headers: Dict[str, str] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    client_abandoned: bool = False
    retries: int = 0
    retry_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_trace_dict(self, call_idx: int, wait_ms: float = 0.0) -> Dict[str, Any]:
        trace: Dict[str, Any] = {
            "call_idx": call_idx,
            "stage": self.stage,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "wait_ms": round(wait_ms, 2),
            "work_ms": round(self.work_ms, 2),
            "provider_first_token_ms": round(self.provider_first_token_ms, 2)
            if self.provider_first_token_ms is not None
            else None,
            "http_status": self.http_status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "ratelimit_headers": redact_dict(self.ratelimit_headers),
            "retry_after_ms": self.retry_after_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "accounting_mode": self.accounting_mode,
        }
        if self.retry_diagnostics:
            for k in (
                "remaining_deadline_ms",
                "next_attempt_budget_ms",
                "downstream_reserve_ms",
                "retry_after_ms",
                "required_total_ms",
                "retry_decision",
                "reject_reason",
            ):
                if k in self.retry_diagnostics:
                    trace[k] = self.retry_diagnostics[k]
        return trace


class GroqCompatibleClient:
    def __init__(
        self,
        config: Dict[str, Any],
        source: str = "real",
        api_key: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.config = config
        self.prov_cfg = config.get("provider", {})
        self.source = source
        self.base_url = (
            self.prov_cfg.get("real_base_url", "https://api.groq.com/openai/v1")
            if source == "real"
            else self.prov_cfg.get("standin_base_url", "http://127.0.0.1:8001/openai/v1")
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "").strip()
        self.default_model = str(self.prov_cfg.get("default_model", "openai/gpt-oss-20b"))
        self.connect_timeout_s = float(self.prov_cfg.get("connect_timeout_s", 10.0))
        self.ttft_timeout_s = float(self.prov_cfg.get("ttft_timeout_s", 15.0))
        self.inter_chunk_timeout_s = float(self.prov_cfg.get("inter_chunk_timeout_s", 8.0))
        self.per_call_timeout_s = float(self.prov_cfg.get("per_call_timeout_s", 30.0))
        self._external_client = http_client
        self._request_deadlines_mono: Dict[str, float] = {}
        self.logger = get_logger("groq_client")

    def register_request_deadline(self, request_id: str, deadline_mono: float) -> None:
        """Registers the logical request's absolute monotonic deadline at request admission."""
        if request_id:
            self._request_deadlines_mono[request_id] = deadline_mono

    def clear_request_deadline(self, request_id: Optional[str]) -> None:
        """Cleans up request-scoped deadline state after the entire logical request terminates."""
        if request_id:
            self._request_deadlines_mono.pop(request_id, None)

    def _evict_stale_deadlines(self, now_mono: float) -> None:
        stale = [k for k, dl in self._request_deadlines_mono.items() if now_mono > dl + 60.0]
        for k in stale:
            self._request_deadlines_mono.pop(k, None)

    def _resolve_request_deadline_mono(
        self,
        req_id: Optional[str],
        call_timeout_ms: Optional[float],
        explicit_deadline_mono: Optional[float] = None,
    ) -> float:
        """
        Resolves the request-wide monotonic deadline from active RequestContext scope,
        registered request ID, explicit deadline_mono, or explicit call_timeout_ms.
        Raises RuntimeError if no deadline context is present rather than silently
        inventing a fresh per-call timer.
        """
        now_mono = time.monotonic()
        scope = CURRENT_REQUEST_SCOPE.get()
        if scope is not None:
            scope_req_id, scope_deadline_mono = scope
            if not req_id:
                req_id = scope_req_id
            if explicit_deadline_mono is None:
                explicit_deadline_mono = scope_deadline_mono

        if explicit_deadline_mono is not None:
            proposed: Optional[float] = explicit_deadline_mono
        elif call_timeout_ms is not None:
            proposed = now_mono + (call_timeout_ms / 1000.0)
        elif req_id and req_id in self._request_deadlines_mono:
            proposed = self._request_deadlines_mono[req_id]
        else:
            raise RuntimeError(
                "Missing request-wide monotonic deadline: execute_single_call requires an active "
                "RequestContext scope, registered request_id deadline, deadline_mono, or call_timeout_ms."
            )

        if not req_id:
            return proposed

        if req_id not in self._request_deadlines_mono:
            if len(self._request_deadlines_mono) >= 4096:
                self._evict_stale_deadlines(now_mono)
            self._request_deadlines_mono[req_id] = proposed
            return proposed

        # Never reset or extend an existing request-wide monotonic deadline across stages
        existing = self._request_deadlines_mono[req_id]
        effective = min(existing, proposed)
        self._request_deadlines_mono[req_id] = effective
        return effective

    def _extract_ratelimit_headers(
        self, headers: httpx.Headers, dimension_hint: Optional[str] = None
    ) -> Tuple[Dict[str, str], Optional[float]]:
        extracted: Dict[str, str] = {}
        for k in RATELIMIT_HEADER_KEYS:
            if k in headers:
                extracted[k] = str(headers[k])

        retry_ms = parse_reset_duration_ms(headers.get("retry-after"))
        if retry_ms is not None:
            return extracted, retry_ms

        r_req = parse_reset_duration_ms(headers.get("x-ratelimit-reset-requests"))
        r_tok = parse_reset_duration_ms(headers.get("x-ratelimit-reset-tokens"))

        if dimension_hint == "rpd_limit":
            return extracted, r_req
        if dimension_hint == "tpm_limit":
            return extracted, r_tok

        if r_tok is not None and r_req is not None:
            return extracted, min(r_tok, r_req)
        return extracted, r_tok if r_tok is not None else r_req

    def _classify_429_dimension(self, error_msg: str, rl_headers: Dict[str, str]) -> Tuple[str, bool]:
        """
        Classifies HTTP 429 rate limits using explicit error-message evidence first,
        then Groq's documented header mapping:
          - x-ratelimit-remaining-requests / x-ratelimit-reset-requests -> RPD (daily)
          - x-ratelimit-remaining-tokens / x-ratelimit-reset-tokens -> TPM (per-minute)
        Preserves 'rate_limit_unknown' when evidence is insufficient rather than guessing RPM.
        """
        msg_lower = (error_msg or "").lower()
        if "tokens per day" in msg_lower or "tpd" in msg_lower:
            return "tpd_limit", False
        if "requests per day" in msg_lower or "rpd" in msg_lower:
            return "rpd_limit", False
        if "tokens per minute" in msg_lower or "tpm" in msg_lower:
            return "tpm_limit", True
        if "requests per minute" in msg_lower or "rpm" in msg_lower:
            return "rpm_limit", True

        rem_req = rl_headers.get("x-ratelimit-remaining-requests")
        rem_tok = rl_headers.get("x-ratelimit-remaining-tokens")
        if rem_req is not None and str(rem_req).strip() == "0":
            return "rpd_limit", False
        if rem_tok is not None and str(rem_tok).strip() == "0":
            return "tpm_limit", True

        has_reset_req = "x-ratelimit-reset-requests" in rl_headers
        has_reset_tok = "x-ratelimit-reset-tokens" in rl_headers
        if has_reset_tok and not has_reset_req:
            return "tpm_limit", True
        if has_reset_req and not has_reset_tok:
            return "rpd_limit", False

        return "rate_limit_unknown", True

    def evaluate_aware_retry(
        self,
        stage: str,
        failure_reason: str,
        is_retriable: bool,
        retry_after_ms: Optional[float],
        remaining_deadline_ms: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluates stage/failure-specific deadline feasibility for the 'aware' retry strategy.
        Preserves the original failure classification when blocking an infeasible retry.
        """
        if (
            failure_reason in ("tpd_limit", "rpd_limit", "auth_error", "internal_error", "request_deadline_exceeded")
            or not is_retriable
        ):
            diag = {
                "remaining_deadline_ms": round(remaining_deadline_ms, 2),
                "next_attempt_budget_ms": 0.0,
                "downstream_reserve_ms": 0.0,
                "retry_after_ms": round(retry_after_ms, 2) if retry_after_ms is not None else 0.0,
                "required_total_ms": 0.0,
                "retry_decision": False,
                "reject_reason": f"non_retriable:{failure_reason}",
            }
            return False, diag

        next_attempt_budget_ms = minimum_attempt_budget_ms(
            stage=stage,
            failure_reason=failure_reason,
            prov_cfg=self.prov_cfg,
        )
        downstream_reserve_ms = required_downstream_reserve_ms(
            stage=stage,
            config=self.config,
        )

        if failure_reason in ("rpm_limit", "tpm_limit", "rate_limit_unknown"):
            wait_ms = (
                float(retry_after_ms)
                if retry_after_ms is not None
                else float(self.prov_cfg.get("default_backoff_ms", 1000.0))
            )
        else:
            wait_ms = float(self.prov_cfg.get("transient_backoff_ms", 250.0))

        required_total_ms = wait_ms + next_attempt_budget_ms + downstream_reserve_ms
        feasible = remaining_deadline_ms >= required_total_ms

        diag = {
            "remaining_deadline_ms": round(remaining_deadline_ms, 2),
            "next_attempt_budget_ms": round(next_attempt_budget_ms, 2),
            "downstream_reserve_ms": round(downstream_reserve_ms, 2),
            "retry_after_ms": round(wait_ms, 2),
            "required_total_ms": round(required_total_ms, 2),
            "retry_decision": feasible,
            "reject_reason": "" if feasible else "insufficient_deadline_budget",
        }
        return feasible, diag

    def _finalize_call_result(
        self,
        result: ProviderCallResult,
        deadline_mono: float,
        req_id: Optional[str],
    ) -> ProviderCallResult:
        """
        Attaches deadline-aware retry feasibility diagnostics for the retry owner
        without prematurely clearing request-wide deadlines on mid-pipeline calls.
        """
        retry_strategy = str(self.config.get("pipeline", {}).get("retry_strategy", "none")).lower()

        if result.success:
            return result

        remaining_deadline_ms = max(0.0, (deadline_mono - time.monotonic()) * 1000.0)

        if retry_strategy == "aware":
            feasible, diag = self.evaluate_aware_retry(
                stage=result.stage,
                failure_reason=result.failure_reason,
                is_retriable=result.is_retriable,
                retry_after_ms=result.retry_after_ms,
                remaining_deadline_ms=remaining_deadline_ms,
            )
            result.retry_diagnostics = diag
            self.logger.info(
                f"aware_retry_check stage={result.stage} req_id={req_id or 'unscoped'} "
                f"failure={result.failure_reason} "
                f"remaining_ms={diag['remaining_deadline_ms']} "
                f"required_ms={diag['required_total_ms']} "
                f"decision={feasible} reject_reason={diag['reject_reason']}"
            )
            if not feasible:
                # Preserve original failure_reason (e.g., ttft_timeout, rpm_limit)
                # while blocking retry dispatch via is_retriable=False.
                result.is_retriable = False
                result.client_abandoned = True

        return result

    async def execute_single_call(
        self,
        messages: List[Dict[str, str]],
        stage: str,
        model_id: Optional[str] = None,
        max_tokens: int = 600,
        temperature: float = 0.0,
        call_timeout_ms: Optional[float] = None,
        fault_inject: Optional[str] = None,
        request_id: Optional[str] = None,
        deadline_mono: Optional[float] = None,
    ) -> ProviderCallResult:
        target_model = model_id or self.default_model
        start_epoch = time.time()
        t0 = time.perf_counter()

        scope = CURRENT_REQUEST_SCOPE.get()
        effective_req_id = (
            request_id
            or (scope[0] if scope is not None else None)
            or extract_request_id_from_fault(fault_inject)
        )
        resolved_deadline_mono = self._resolve_request_deadline_mono(
            effective_req_id, call_timeout_ms, explicit_deadline_mono=deadline_mono
        )

        est_prompt_tokens = sum(count_tokens(m.get("content", "")) + 4 for m in messages)

        if self.source == "real" and (not self.api_key or self.api_key == "your_groq_api_key_here"):
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ProviderCallResult(
                success=False,
                stage=stage,
                model_id=target_model,
                http_status=0,
                content="",
                provider_first_token_ms=None,
                work_ms=elapsed,
                tokens_in=0,
                tokens_out=0,
                accounting_mode="none",
                error_code="missing_api_key",
                error_message="GROQ_API_KEY is not configured.",
                failure_reason="auth_error",
                is_retriable=False,
                start_time=start_epoch,
                end_time=start_epoch + elapsed / 1000.0,
            )

        remaining_deadline_s = resolved_deadline_mono - time.monotonic()
        effective_call_timeout_s = min(self.per_call_timeout_s, remaining_deadline_s)

        if effective_call_timeout_s <= 0.05:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ProviderCallResult(
                success=False,
                stage=stage,
                model_id=target_model,
                http_status=0,
                content="",
                provider_first_token_ms=None,
                work_ms=elapsed,
                tokens_in=0,
                tokens_out=0,
                accounting_mode="none",
                error_code="deadline_exceeded",
                error_message="Insufficient remaining deadline before provider dispatch.",
                failure_reason="request_deadline_exceeded",
                is_retriable=False,
                start_time=start_epoch,
                end_time=start_epoch + elapsed / 1000.0,
                client_abandoned=True,
                retry_diagnostics={
                    "remaining_deadline_ms": round(max(0.0, remaining_deadline_s * 1000.0), 2),
                    "next_attempt_budget_ms": 0.0,
                    "downstream_reserve_ms": 0.0,
                    "retry_after_ms": 0.0,
                    "required_total_ms": 0.0,
                    "retry_decision": False,
                    "reject_reason": "deadline_already_expired",
                },
            )

        headers = {
            "Content-Type": "application/json",
            "X-Relay-Stage": stage,
        }
        # The local stand-in does not require authentication. Never emit an
        # empty `Bearer ` value: HTTPX rejects it before the request is sent.
        if self.api_key and self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"
        if fault_inject:
            headers["X-Relay-Fault"] = fault_inject

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "response_format": {"type": "json_object"},
        }

        timeout_cfg = httpx.Timeout(
            connect=min(self.connect_timeout_s, effective_call_timeout_s),
            read=max(self.ttft_timeout_s + 5.0, effective_call_timeout_s),
            write=min(self.connect_timeout_s, effective_call_timeout_s),
            pool=min(self.connect_timeout_s, effective_call_timeout_s),
        )

        client = self._external_client or httpx.AsyncClient(timeout=timeout_cfg)
        should_close = self._external_client is None

        content_parts: List[str] = []
        first_token_ms: Optional[float] = None
        reported_in: Optional[int] = None
        reported_out: Optional[int] = None
        rl_headers: Dict[str, str] = {}
        retry_after_ms: Optional[float] = None
        http_status = 0

        try:
            async with asyncio.timeout(effective_call_timeout_s):
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", headers=headers, json=payload
                ) as response:
                    http_status = int(response.status_code)
                    raw_rl_headers, retry_after_ms = self._extract_ratelimit_headers(response.headers)
                    rl_headers = raw_rl_headers

                    if http_status != 200:
                        raw_err = (await response.aread()).decode("utf-8", errors="replace")
                        elapsed = (time.perf_counter() - t0) * 1000.0
                        err_code = f"http_{http_status}"
                        err_msg = raw_err[:400]
                        try:
                            err_json = json.loads(raw_err)
                            if isinstance(err_json, dict) and "error" in err_json:
                                e_obj = err_json["error"]
                                err_code = str(e_obj.get("code") or e_obj.get("type") or err_code)
                                err_msg = str(e_obj.get("message") or err_msg)
                        except Exception:
                            pass

                        if http_status == 429:
                            reason, retriable = self._classify_429_dimension(err_msg, rl_headers)
                            _, refined_retry_ms = self._extract_ratelimit_headers(
                                response.headers, dimension_hint=reason
                            )
                            res = ProviderCallResult(
                                success=False,
                                stage=stage,
                                model_id=target_model,
                                http_status=429,
                                content="",
                                provider_first_token_ms=None,
                                work_ms=elapsed,
                                tokens_in=0,
                                tokens_out=0,
                                accounting_mode="none",
                                error_code=err_code,
                                error_message=err_msg,
                                failure_reason=reason,
                                is_retriable=retriable,
                                retry_after_ms=refined_retry_ms,
                                ratelimit_headers=rl_headers,
                                start_time=start_epoch,
                                end_time=start_epoch + elapsed / 1000.0,
                            )
                            return self._finalize_call_result(res, resolved_deadline_mono, effective_req_id)

                        is_5xx = 500 <= http_status < 600
                        is_auth = http_status in (401, 403)
                        res = ProviderCallResult(
                            success=False,
                            stage=stage,
                            model_id=target_model,
                            http_status=http_status,
                            content="",
                            provider_first_token_ms=None,
                            work_ms=elapsed,
                            tokens_in=0,
                            tokens_out=0,
                            accounting_mode="none",
                            error_code=err_code,
                            error_message=err_msg,
                            failure_reason="auth_error" if is_auth else ("http_5xx" if is_5xx else f"http_{http_status}"),
                            is_retriable=is_5xx,
                            retry_after_ms=retry_after_ms,
                            ratelimit_headers=rl_headers,
                            start_time=start_epoch,
                            end_time=start_epoch + elapsed / 1000.0,
                        )
                        return self._finalize_call_result(res, resolved_deadline_mono, effective_req_id)

                    line_iter = response.aiter_lines()
                    while True:
                        wait_limit = self.ttft_timeout_s if first_token_ms is None else self.inter_chunk_timeout_s
                        try:
                            raw_line = await asyncio.wait_for(anext(line_iter), timeout=wait_limit)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as te:
                            raise TimeoutError("ttft_timeout" if first_token_ms is None else "inter_chunk_timeout") from te

                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break

                        chunk_obj = json.loads(data_str)
                        choices = chunk_obj.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            piece_txt = delta.get("content")
                            if piece_txt:
                                if first_token_ms is None:
                                    first_token_ms = (time.perf_counter() - t0) * 1000.0
                                content_parts.append(str(piece_txt))

                        usage = chunk_obj.get("usage")
                        if not usage and "x_groq" in chunk_obj and isinstance(chunk_obj["x_groq"], dict):
                            usage = chunk_obj["x_groq"].get("usage")
                        if isinstance(usage, dict):
                            if usage.get("prompt_tokens") is not None:
                                reported_in = int(usage["prompt_tokens"])
                            if usage.get("completion_tokens") is not None:
                                reported_out = int(usage["completion_tokens"])

            elapsed = (time.perf_counter() - t0) * 1000.0
            full_content = "".join(content_parts)
            if first_token_ms is None and full_content:
                first_token_ms = elapsed

            if reported_in is not None and reported_out is not None:
                tok_in, tok_out, acc_mode = reported_in, reported_out, "exact_reported"
            else:
                tok_in, tok_out, acc_mode = est_prompt_tokens, count_tokens(full_content), "partial_estimated"

            res = ProviderCallResult(
                success=True,
                stage=stage,
                model_id=target_model,
                http_status=200,
                content=full_content,
                provider_first_token_ms=first_token_ms,
                work_ms=elapsed,
                tokens_in=tok_in,
                tokens_out=tok_out,
                accounting_mode=acc_mode,
                failure_reason="none",
                is_retriable=False,
                retry_after_ms=retry_after_ms,
                ratelimit_headers=rl_headers,
                start_time=start_epoch,
                end_time=start_epoch + elapsed / 1000.0,
            )
            return self._finalize_call_result(res, resolved_deadline_mono, effective_req_id)

        except (httpx.RequestError, TimeoutError, asyncio.TimeoutError) as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            elapsed_s = elapsed / 1000.0
            now_mono = time.monotonic()
            partial_text = "".join(content_parts)
            partial_out = count_tokens(partial_text)
            had_prog = first_token_ms is not None or partial_out > 0 or http_status == 200
            explicit_tag = str(exc) if isinstance(exc, TimeoutError) else ""

            if isinstance(exc, httpx.ConnectTimeout):
                f_reason = "connect_timeout"
                retriable = True
            elif now_mono >= resolved_deadline_mono - 0.05:
                f_reason = "request_deadline_exceeded"
                retriable = False
            elif first_token_ms is None and isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.ReadTimeout)):
                if elapsed_s + 0.05 < self.ttft_timeout_s:
                    f_reason = "request_deadline_exceeded"
                    retriable = False
                else:
                    f_reason = "ttft_timeout"
                    retriable = True
            elif isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.ReadTimeout)):
                if elapsed_s + 0.05 < self.inter_chunk_timeout_s and explicit_tag != "inter_chunk_timeout":
                    f_reason = "request_deadline_exceeded"
                    retriable = False
                else:
                    f_reason = "inter_chunk_timeout"
                    retriable = True
            elif first_token_ms is not None and isinstance(exc, (httpx.ReadError, httpx.RemoteProtocolError)):
                # A transport read failure after SSE content has started is a
                # mid-stream disconnect, even when HTTPX surfaces it as ReadError.
                f_reason = "disconnect_mid_stream"
                retriable = True
            elif isinstance(exc, httpx.RemoteProtocolError):
                # Preserve protocol-disconnect classification even before a token
                # has been decoded from the stream.
                f_reason = "disconnect_mid_stream"
                retriable = True
            else:
                f_reason = "network_error"
                retriable = True

            if isinstance(exc, httpx.RemoteProtocolError):
                err_code = "RemoteProtocolError"
                err_msg = "peer closed connection without sending complete message body (incomplete chunked read)"
            else:
                err_code = type(exc).__name__
                err_msg = str(exc)[:300]

            # Discard incomplete partial_text as candidate output; report known partial token usage
            res = ProviderCallResult(
                success=False,
                stage=stage,
                model_id=target_model,
                http_status=http_status,
                content="",
                provider_first_token_ms=first_token_ms,
                work_ms=elapsed,
                tokens_in=est_prompt_tokens if had_prog else 0,
                tokens_out=partial_out if had_prog else 0,
                accounting_mode="partial_estimated" if had_prog else "none",
                error_code=err_code,
                error_message=err_msg,
                failure_reason=f_reason,
                is_retriable=retriable,
                retry_after_ms=retry_after_ms,
                ratelimit_headers=rl_headers,
                start_time=start_epoch,
                end_time=start_epoch + elapsed / 1000.0,
            )
            return self._finalize_call_result(res, resolved_deadline_mono, effective_req_id)
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ProviderCallResult(
                success=False,
                stage=stage,
                model_id=target_model,
                http_status=http_status,
                content="",
                provider_first_token_ms=first_token_ms,
                work_ms=elapsed,
                tokens_in=0,
                tokens_out=0,
                accounting_mode="none",
                error_code="internal_error",
                error_message=str(exc)[:300],
                failure_reason="internal_error",
                is_retriable=False,
                retry_after_ms=retry_after_ms,
                ratelimit_headers=rl_headers,
                start_time=start_epoch,
                end_time=start_epoch + elapsed / 1000.0,
            )
        finally:
            if should_close:
                await client.aclose()