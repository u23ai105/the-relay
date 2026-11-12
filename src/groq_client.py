"""
Async OpenAI-compatible HTTP/SSE client for Groq (and later the Stand-In service).
Measures provider TTFT, active call work duration, token usage (exact provider-reported vs.
partial estimated on mid-stream drops), and sanitized rate-limit headers.
"""

import asyncio
from dataclasses import asdict, dataclass, field
import json
import os
import random
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


def parse_reset_duration_ms(val: Optional[str]) -> Optional[float]:
    if not val:
        return None
    s = str(val).strip().lower()
    try:
        return float(s) * 1000.0
    except ValueError:
        pass
    # Parse Groq duration strings like "1m2.5s", "450ms", "6.2s"
    total_ms = 0.0
    matched = False
    import re

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
    accounting_mode: str  # "exact_reported", "partial_estimated", "unknown", "none"
    error_code: str = ""
    error_message: str = ""
    failure_reason: str = "none"
    is_retriable: bool = False
    retry_after_ms: Optional[float] = None
    ratelimit_headers: Dict[str, str] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0

    def to_trace_dict(self, call_idx: int, wait_ms: float = 0.0) -> Dict[str, Any]:
        return {
            "call_idx": call_idx,
            "stage": self.stage,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "wait_ms": round(wait_ms, 2),
            "work_ms": round(self.work_ms, 2),
            "provider_first_token_ms": (
                round(self.provider_first_token_ms, 2)
                if self.provider_first_token_ms is not None
                else None
            ),
            "http_status": self.http_status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "ratelimit_headers": redact_dict(self.ratelimit_headers),
            "retry_after_ms": self.retry_after_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "accounting_mode": self.accounting_mode,
        }


class GroqCompatibleClient:
    """
    Executes streaming chat completion calls against Groq or the Stand-In service,
    supporting four-tier timeouts and bounded transport retries with backoff & jitter.
    """

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
        self.logger = get_logger("groq_client")

    def _extract_ratelimit_headers(
        self, headers: httpx.Headers
    ) -> Tuple[Dict[str, str], Optional[float]]:
        extracted: Dict[str, str] = {}
        for k in RATELIMIT_HEADER_KEYS:
            if k in headers:
                extracted[k] = str(headers[k])

        retry_ms = parse_reset_duration_ms(headers.get("retry-after"))
        if retry_ms is None:
            r_req = parse_reset_duration_ms(headers.get("x-ratelimit-reset-requests"))
            r_tok = parse_reset_duration_ms(headers.get("x-ratelimit-reset-tokens"))
            if r_req is not None and r_tok is not None:
                retry_ms = max(r_req, r_tok)
            else:
                retry_ms = r_req if r_req is not None else r_tok
        return extracted, retry_ms

    def _classify_429_dimension(self, error_msg: str, rl_headers: Dict[str, str]) -> Tuple[str, bool]:
        msg_lower = error_msg.lower()
        if "tokens per day" in msg_lower or "tpd" in msg_lower:
            return "tpd_limit", False
        if "requests per day" in msg_lower or "rpd" in msg_lower:
            return "rpd_limit", False
        if "tokens per minute" in msg_lower or "tpm" in msg_lower:
            return "tpm_limit", True
        if "requests per minute" in msg_lower or "rpm" in msg_lower:
            return "rpm_limit", True

        rem_req = rl_headers.get("x-ratelimit-remaining-requests")
        if rem_req is not None and str(rem_req).strip() == "0":
            return "rpd_limit", False
        return "rpm_limit", True

    async def execute_single_call(
        self,
        messages: List[Dict[str, str]],
        stage: str,
        model_id: Optional[str] = None,
        max_tokens: int = 600,
        temperature: float = 0.0,
        call_timeout_ms: Optional[float] = None,
    ) -> ProviderCallResult:
        target_model = model_id or self.default_model
        start_epoch = time.time()
        t0 = time.perf_counter()

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

        effective_call_timeout_s = min(
            self.per_call_timeout_s,
            (call_timeout_ms / 1000.0) if call_timeout_ms is not None else self.per_call_timeout_s,
        )
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
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Relay-Stage": stage,
        }
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
            read=min(self.ttft_timeout_s, effective_call_timeout_s),
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
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    http_status = int(response.status_code)
                    rl_headers, retry_after_ms = self._extract_ratelimit_headers(response.headers)

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
                            return ProviderCallResult(
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
                                retry_after_ms=retry_after_ms,
                                ratelimit_headers=rl_headers,
                                start_time=start_epoch,
                                end_time=start_epoch + elapsed / 1000.0,
                            )

                        is_5xx = 500 <= http_status < 600
                        return ProviderCallResult(
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
                            failure_reason="http_5xx" if is_5xx else f"http_{http_status}",
                            is_retriable=is_5xx,
                            retry_after_ms=retry_after_ms,
                            ratelimit_headers=rl_headers,
                            start_time=start_epoch,
                            end_time=start_epoch + elapsed / 1000.0,
                        )

                    line_iter = response.aiter_lines()
                    while True:
                        wait_limit = (
                            self.ttft_timeout_s
                            if first_token_ms is None
                            else self.inter_chunk_timeout_s
                        )
                        try:
                            raw_line = await asyncio.wait_for(
                                anext(line_iter), timeout=wait_limit
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as te:
                            raise TimeoutError(
                                "ttft_timeout" if first_token_ms is None else "inter_chunk_timeout"
                            ) from te

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
                tok_in, tok_out = reported_in, reported_out
                acc_mode = "exact_reported"
            else:
                tok_in = est_prompt_tokens
                tok_out = count_tokens(full_content)
                acc_mode = "partial_estimated"

            return ProviderCallResult(
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

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            partial_text = "".join(content_parts)
            partial_out = count_tokens(partial_text)
            had_stream_progress = first_token_ms is not None or partial_out > 0 or http_status == 200

            if isinstance(exc, httpx.ConnectTimeout):
                f_reason = "connect_timeout"
            elif isinstance(exc, TimeoutError) and str(exc) in ("ttft_timeout", "inter_chunk_timeout"):
                f_reason = str(exc)
            elif isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.ReadTimeout)):
                f_reason = "ttft_timeout" if first_token_ms is None else "inter_chunk_timeout"
            else:
                f_reason = "inter_chunk_timeout" if had_stream_progress else "http_5xx"

            return ProviderCallResult(
                success=False,
                stage=stage,
                model_id=target_model,
                http_status=http_status,
                content=partial_text,
                provider_first_token_ms=first_token_ms,
                work_ms=elapsed,
                tokens_in=est_prompt_tokens if had_stream_progress else 0,
                tokens_out=partial_out if had_stream_progress else 0,
                accounting_mode="partial_estimated" if had_stream_progress else "none",
                error_code=type(exc).__name__,
                error_message=str(exc)[:300],
                failure_reason=f_reason,
                is_retriable=True,
                retry_after_ms=retry_after_ms,
                ratelimit_headers=rl_headers,
                start_time=start_epoch,
                end_time=start_epoch + elapsed / 1000.0,
            )
        finally:
            if should_close:
                await client.aclose()