"""
Live Groq API Availability and Rate-Limit Preflight.
Executes at most ONE real inference request against the configured Groq API endpoint
with retries strictly disabled (retry_strategy='none', max_transport_retries=0).
Saves telemetry separately to results/raw/groq_live_preflight.json without touching local
benchmark artifacts or secrets.
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.groq_client import (
    GroqCompatibleClient,
    RATELIMIT_HEADER_KEYS,
    end_request_scope,
    start_request_scope,
)


def classify_live_result(
    http_status: int,
    success: bool,
    response_valid: bool,
    failure_reason: str,
    error_code: str,
) -> str:
    if http_status == 200 and success and response_valid:
        return "ok"
    if http_status == 200 and not response_valid:
        return "invalid_response"
    if http_status == 429 or failure_reason in ("rpm_limit", "tpm_limit", "rpd_limit", "tpd_limit", "rate_limit_unknown"):
        return "rate_limited"
    if http_status in (401, 403) or failure_reason == "auth_error":
        return "authentication_or_permission_error"
    if http_status in (400, 404, 422):
        return "request_or_model_error"
    if "timeout" in failure_reason or "Timeout" in error_code:
        return "timeout"
    if 500 <= http_status < 600 or failure_reason == "http_5xx":
        return "provider_error"
    return "network_error"


async def run_live_preflight() -> int:
    config = load_config()
    config.setdefault("pipeline", {})["retry_strategy"] = "none"
    config.setdefault("provider", {})["max_transport_retries"] = 0

    client = GroqCompatibleClient(config, source="real")
    model_id = config["provider"].get("default_model", "openai/gpt-oss-20b")
    out_path = PROJECT_ROOT / "results" / "raw" / "groq_live_preflight.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    if not client.api_key or client.api_key == "your_groq_api_key_here":
        report: Dict[str, Any] = {
            "provider": "groq",
            "endpoint": client.base_url,
            "model": model_id,
            "timestamp": timestamp,
            "http_status": 0,
            "success": False,
            "response_valid": False,
            "latency_ms": 0.0,
            "first_token_ms": None,
            "status": "missing_api_key",
            "tokens_in": 0,
            "tokens_out": 0,
            "token_accounting_mode": "none",
            "rate_limit_headers": {k: "unavailable" for k in RATELIMIT_HEADER_KEYS},
            "retry_after": "unavailable",
            "error_classification": "missing_api_key",
            "error_message": "GROQ_API_KEY is missing or unset in environment.",
        }
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        tmp.replace(out_path)
        print(f"Preflight skipped: GROQ_API_KEY missing. Saved report to {out_path}")
        return 1

    print(f"Executing single live preflight request to {client.base_url} with model {model_id}...")

    req_id = "groq_live_preflight_1"
    deadline_ms = 15000.0
    deadline_mono = time.monotonic() + (deadline_ms / 1000.0)
    client.register_request_deadline(req_id, deadline_mono)
    scope_token = start_request_scope(req_id, deadline_mono)

    try:
        result = await client.execute_single_call(
            messages=[
                {
                    "role": "user",
                    "content": 'Return valid JSON only: {"preflight": "ok"}',
                }
            ],
            stage="generate",
            model_id=model_id,
            max_tokens=64,
            temperature=0.0,
            call_timeout_ms=deadline_ms,
            request_id=req_id,
        )
    finally:
        end_request_scope(scope_token)
        client.clear_request_deadline(req_id)

    response_valid = False
    if result.success and result.content:
        try:
            parsed = json.loads(result.content)
            response_valid = isinstance(parsed, dict) and parsed.get("preflight") == "ok"
        except Exception:
            response_valid = False

    classification = classify_live_result(
        http_status=result.http_status,
        success=result.success,
        response_valid=response_valid,
        failure_reason=result.failure_reason,
        error_code=result.error_code,
    )

    headers_report: Dict[str, str] = {}
    for k in RATELIMIT_HEADER_KEYS:
        val = result.ratelimit_headers.get(k)
        headers_report[k] = val if val not in (None, "") else "unavailable"

    retry_after_val: Any = result.retry_after_ms if result.retry_after_ms is not None else "unavailable"

    report = {
        "provider": "groq",
        "endpoint": client.base_url,
        "model": model_id,
        "timestamp": timestamp,
        "http_status": result.http_status,
        "success": bool(result.success and response_valid),
        "response_valid": response_valid,
        "latency_ms": round(result.work_ms, 2),
        "first_token_ms": round(result.provider_first_token_ms, 2) if result.provider_first_token_ms is not None else None,
        "status": classification,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "token_accounting_mode": result.accounting_mode,
        "rate_limit_headers": headers_report,
        "retry_after": retry_after_val,
        "error_classification": "" if classification == "ok" else classification,
        "error_message": "" if classification == "ok" else result.error_message,
    }

    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    tmp.replace(out_path)

    print(f"Live Groq Preflight Status: {classification} (HTTP {result.http_status}, latency={report['latency_ms']} ms)")
    print(f"Saved preflight report to {out_path}")
    return 0 if classification == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_live_preflight()))