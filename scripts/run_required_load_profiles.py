"""Run the five load profiles required by The Relay assignment.

All requests use the local stateful stand-in. No Groq API key is read and no
real provider calls are made by this script.

Scenarios:
  1. steady
  2. burst
  3. sustained_overload
  4. per_minute_limit_halved
  5. daily_budget_exhausted

The RPM/TPD scenarios deliberately apply local scheduler quota caps. They are
stand-in simulations, not measurements of Groq's actual limits. Groq limits
must be reported separately from scripts/check_groq_live.py and provider docs.

Output files are timestamped under results/raw/ and never overwrite existing
benchmarks. All percentile latency metrics in the JSON summary use successful
logical requests only; unsuccessful requests are counted separately.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_load_experiments import safe_run_request, start_benchmark_server  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluation_questions import EVALUATION_QUESTIONS  # noqa: E402
from src.groq_client import GroqCompatibleClient  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.retrieval import RetrievalEngineRegistry  # noqa: E402

PORT = 8014
RETRIEVAL_SETTING = "500tok_12to4_rerank_yes"
DEADLINE_MS = 90_000.0

# Request count, spacing between arrivals, provider concurrency, and quota override.
# Arrival spacing is in seconds. 0.0 means submit the whole wave immediately.
PROFILES: dict[str, dict[str, Any]] = {
    "steady": {
        "description": "12 requests arriving at a steady 2.0-second interval",
        "requests": 12,
        "interval_s": 2.0,
        "provider_concurrency": 5,
        "quota": "normal_local_caps_disabled_for_isolation",
    },
    "burst": {
        "description": "24 requests arriving in a short burst at 10-ms intervals",
        "requests": 24,
        "interval_s": 0.01,
        "provider_concurrency": 3,
        "quota": "normal_local_caps_disabled_for_isolation",
    },
    "sustained_overload": {
        # The first run offered only ~306 requests/min while observed service
        # throughput was roughly 360 requests/min, so it did not build a queue.
        # This corrected profile offers 10 requests/s for 8 seconds (80 total),
        # above the measured two-slot stand-in throughput, to create sustained
        # queueing while keeping the run local and below the 90-second deadline.
        "description": "80 requests arriving every 0.10 seconds for 8 seconds (10 req/s), with provider concurrency capped at 2",
        "requests": 80,
        "interval_s": 0.10,
        "provider_concurrency": 2,
        "quota": "normal_local_caps_disabled_for_isolation",
    },
    "per_minute_limit_halved": {
        "description": "24-request burst with local provider-call RPM admission cap set to half the 30-RPM baseline (15 RPM)",
        "requests": 24,
        "interval_s": 0.01,
        "provider_concurrency": 10,
        "quota": "local_limit_rpm_halved",
    },
    "daily_budget_exhausted": {
        "description": "8 requests with local daily token budget set below one estimated provider call",
        "requests": 8,
        "interval_s": 0.01,
        "provider_concurrency": 10,
        "quota": "local_limit_tpd_exhausted",
    },
}

MANDATORY_COLUMNS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms",
    "total_ms", "status", "retries", "tokens_in", "tokens_out",
    "retrieval_setting", "answer_grade",
]


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile; callers pass successful-request values only."""
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    index = max(0, math.ceil((p / 100.0) * len(ordered)) - 1)
    return round(ordered[index], 2)


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError("No request rows were produced; refusing to write an empty CSV.")
    extra = [k for k in rows[0].keys() if k not in MANDATORY_COLUMNS]
    fieldnames = MANDATORY_COLUMNS + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe = {key: row.get(key, "") for key in fieldnames}
            if safe["status"] not in {"ok", "rate_limited", "timeout", "error"}:
                safe["status"] = "error"
            if safe["status"] != "ok":
                safe["answer_grade"] = ""
            elif safe["answer_grade"] not in {"correct", "wrong", "refused_correctly", "refused_wrongly", ""}:
                safe["answer_grade"] = ""
            writer.writerow(safe)


async def reset_provider() -> None:
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"http://127.0.0.1:{PORT}/reset")
        response.raise_for_status()


def configure_profile(base: dict[str, Any], name: str, profile: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("pipeline", {})
    cfg.setdefault("provider", {})

    # This suite isolates arrival shape and quota behavior. The earlier 450-run
    # benchmark already compares none/naive/aware retry policies.
    cfg["pipeline"]["retry_strategy"] = "none"
    cfg["pipeline"]["max_local_concurrency"] = 100
    cfg["pipeline"]["max_local_queue_depth"] = 500
    cfg["provider"]["max_provider_concurrency"] = int(profile["provider_concurrency"])
    cfg["provider"]["max_transport_retries"] = 0
    cfg["provider"]["standin_base_url"] = f"http://127.0.0.1:{PORT}/openai/v1"
    cfg["provider"]["ttft_timeout_s"] = 15.0

    # Disable quotas for the three traffic-shape tests so they isolate queueing.
    cfg["provider"]["limit_rpm"] = 1_000_000
    cfg["provider"]["limit_tpm"] = 1_000_000_000
    cfg["provider"]["limit_rpd"] = 1_000_000
    cfg["provider"]["limit_tpd"] = 1_000_000_000

    if name == "per_minute_limit_halved":
        baseline_rpm = int(base.get("provider", {}).get("limit_rpm", 30) or 30)
        cfg["provider"]["limit_rpm"] = max(1, baseline_rpm // 2)
    elif name == "daily_budget_exhausted":
        # A positive TPD limit smaller than one estimated prompt+completion
        # reservation deterministically blocks calls at local admission.
        cfg["provider"]["limit_tpd"] = 1

    return cfg


async def run_profile(
    name: str,
    profile: dict[str, Any],
    base_config: dict[str, Any],
    registry: RetrievalEngineRegistry,
    batch_id: str,
    logger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    await reset_provider()
    config = configure_profile(base_config, name, profile)
    client = GroqCompatibleClient(
        config,
        source="stand-in",
        http_client=httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{PORT}/openai/v1",
            timeout=httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=10.0),
        ),
    )
    pipeline = RAGPipeline(config, registry, client, None)
    question = EVALUATION_QUESTIONS[0]
    run_id = f"{batch_id}_{name}"

    logger.info(
        "Starting %s: requests=%s interval_s=%s concurrency=%s rpm=%s tpd=%s",
        name,
        profile["requests"],
        profile["interval_s"],
        profile["provider_concurrency"],
        config["provider"]["limit_rpm"],
        config["provider"]["limit_tpd"],
    )

    scenario_start_epoch = time.time()
    schedule_start = time.perf_counter()
    tasks: list[asyncio.Task] = []

    try:
        for i in range(int(profile["requests"])):
            request_id = f"{run_id}_r{i + 1:03d}"
            fault_mode = "load_sensitive" if name in ("steady", "burst", "sustained_overload") else "none"
            tasks.append(
                asyncio.create_task(
                    safe_run_request(
                        pipeline=pipeline,
                        question=question.question,
                        request_id=request_id,
                        scenario=name,
                        retrieval_setting=RETRIEVAL_SETTING,
                        eval_question=question,
                        enable_cache=False,
                        fault_inject=f"{fault_mode};req_id={request_id}",
                        request_deadline_ms=DEADLINE_MS,
                    )
                )
            )
            if i + 1 < int(profile["requests"]) and float(profile["interval_s"]) > 0:
                await asyncio.sleep(float(profile["interval_s"]))

        load_end_epoch = time.time()
        load_duration_s = max(0.0, time.perf_counter() - schedule_start)
        results = await asyncio.gather(*tasks)
        completion_epoch = time.time()

        rows: list[dict[str, Any]] = []
        for result in results:
            row = dict(result.telemetry_row)
            row.update(
                {
                    "scenario": name,
                    "source": "stand-in",
                    "scenario_start_time": scenario_start_epoch,
                    "scenario_load_end_time": load_end_epoch,
                    "scenario_completion_time": completion_epoch,
                    "scenario_load_duration_s": round(load_duration_s, 6),
                    "run_batch_id": batch_id,
                    "run_id": run_id,
                    "profile_description": profile["description"],
                    "arrival_interval_s": profile["interval_s"],
                    "offered_rate_rpm": round(len(results) / max(load_duration_s, 0.001) * 60.0, 2),
                    "configured_provider_concurrency": profile["provider_concurrency"],
                    "configured_local_rpm_cap": config["provider"]["limit_rpm"],
                    "configured_local_tpd_cap": config["provider"]["limit_tpd"],
                    "quota_mode": profile["quota"],
                }
            )
            if row.get("status") != "ok":
                row["answer_grade"] = ""
            rows.append(row)

        successful = [r for r in rows if r.get("status") == "ok"]
        first_tokens = [float(r["first_token_ms"]) for r in successful if r.get("first_token_ms") not in ("", None)]
        total_times = [float(r["total_ms"]) for r in successful if r.get("total_ms") not in ("", None)]
        statuses = [str(r.get("status", "error")) for r in rows]
        count = len(rows)
        success_count = statuses.count("ok")
        rate_limited_count = statuses.count("rate_limited")
        timeout_count = statuses.count("timeout")
        error_count = statuses.count("error")
        avg_tokens = (
            sum(int(r.get("tokens_in", 0) or 0) + int(r.get("tokens_out", 0) or 0) for r in successful)
            / success_count
            if success_count
            else None
        )
        avg_retries = sum(int(r.get("retries", 0) or 0) for r in rows) / max(1, count)
        peak_local = max((int(r.get("local_queue_depth_peak", 0) or 0) for r in rows), default=0)
        peak_provider = max((int(r.get("provider_queue_depth_peak", 0) or 0) for r in rows), default=0)
        peak_in_flight = max((int(r.get("in_flight_peak", 0) or 0) for r in rows), default=0)

        summary = {
            "scenario": name,
            "description": profile["description"],
            "source": "stand-in",
            "requests_sent": count,
            "scenario_duration_s": round(completion_epoch - scenario_start_epoch, 3),
            "load_generation_duration_s": round(load_duration_s, 6),
            "offered_rate_per_min": round(count / max(load_duration_s, 0.001) * 60.0, 2),
            "success_count": success_count,
            "success_pct": round(success_count / max(1, count) * 100.0, 1),
            "rate_limited_count": rate_limited_count,
            "rate_limited_pct": round(rate_limited_count / max(1, count) * 100.0, 1),
            "timeout_count": timeout_count,
            "timeout_pct": round(timeout_count / max(1, count) * 100.0, 1),
            "error_count": error_count,
            "error_pct": round(error_count / max(1, count) * 100.0, 1),
            "retries_per_request": round(avg_retries, 3),
            "peak_local_queue": peak_local,
            "peak_provider_queue": peak_provider,
            "peak_queue_length": max(peak_local, peak_provider),
            "peak_in_flight": peak_in_flight,
            "tokens_per_successful_answer": round(avg_tokens, 2) if avg_tokens is not None else None,
            "first_token_ms_success_only": stats(first_tokens),
            "total_ms_success_only": stats(total_times),
            "retry_strategy": "none",
            "quota_mode": profile["quota"],
            "configured_rpm_cap": config["provider"]["limit_rpm"],
            "configured_tpd_cap": config["provider"]["limit_tpd"],
            "note": (
                "RPM/TPD caps in this local profile runner are scheduler-side stand-in controls, "
                "not observations of Groq's provider-enforced limits."
                if name in ("per_minute_limit_halved", "daily_budget_exhausted")
                else "Traffic shape is controlled by request arrival spacing and stand-in concurrency."
            ),
        }
        logger.info(
            "%s complete: %s/%s successful, offered %.2f requests/min, duration %.2fs",
            name,
            success_count,
            count,
            summary["offered_rate_per_min"],
            summary["scenario_duration_s"],
        )
        return rows, summary
    finally:
        await client._external_client.aclose()


async def async_main(selected_profiles: list[str]) -> int:
    logger = get_logger("required_load_profiles")
    batch_id = datetime.now(timezone.utc).strftime("official_profiles_%Y%m%dT%H%M%SZ")
    raw_dir = PROJECT_ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"{batch_id}.csv"
    json_path = raw_dir / f"{batch_id}.json"

    server, thread = start_benchmark_server(port=PORT)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    try:
        config = load_config()
        indices_dir = PROJECT_ROOT / "data" / "indices"
        registry = RetrievalEngineRegistry(indices_dir, config)
        for name in selected_profiles:
            rows, summary = await run_profile(
                name=name,
                profile=PROFILES[name],
                base_config=config,
                registry=registry,
                batch_id=batch_id,
                logger=logger,
            )
            all_rows.extend(rows)
            summaries.append(summary)

        write_csv(all_rows, csv_path)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "batch_id": batch_id,
            "environment": "local_stateful_benchmark_provider",
            "source": "stand-in",
            "retrieval_setting": RETRIEVAL_SETTING,
            "retry_strategy": "none",
            "scenarios": summaries,
            "csv_path": str(csv_path),
            "interpretation": (
                "All five required load profiles run against the local stand-in. "
                "No Groq requests are made. RPM and TPD scenarios adjust the local "
                "scheduler quota manager and must not be reported as real Groq limits."
            ),
        }
        tmp = json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        tmp.replace(json_path)

        print(f"\nOfficial profile CSV: {csv_path}")
        print(f"Official profile JSON: {json_path}")
        print("\nScenario summary:")
        for item in summaries:
            print(
                f"{item['scenario']}: {item['success_count']}/{item['requests_sent']} ok | "
                f"rate-limited={item['rate_limited_count']} | timeout={item['timeout_count']} | "
                f"offered={item['offered_rate_per_min']:.2f}/min | duration={item['scenario_duration_s']:.2f}s"
            )
        return 0
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=[*PROFILES.keys(), "all"],
        default="all",
        help="Run all five official profiles or just one profile (default: all).",
    )
    args = parser.parse_args()
    chosen = list(PROFILES) if args.profile == "all" else [args.profile]
    return asyncio.run(async_main(chosen))


if __name__ == "__main__":
    raise SystemExit(main())
