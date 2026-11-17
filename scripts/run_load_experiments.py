"""
Batch 3: Automated Load Experiment Runner (Final Audit Correction).
Executes 5 stress scenarios across 3 retry strategies against the Stateful Benchmark Provider.
Preserves original baseline artifacts and all earlier corrected runs via unique timestamped
batch run IDs, establishes request-scoped monotonic deadlines at admission, and records
uniform scheduled scenario-level timing telemetry across all grouped rows.
"""

import asyncio
import copy
import csv
import json
import math
import shutil
import socket
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict

import httpx
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluation_questions import EVALUATION_QUESTIONS
from src.groq_client import GroqCompatibleClient, end_request_scope, start_request_scope
from src.logger import get_logger
from src.pipeline import RAGPipeline, PipelineExecutionResult
from src.retrieval import RetrievalEngineRegistry
from scripts.benchmark_provider import app

MANDATORY_COLS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms",
    "total_ms", "status", "retries", "tokens_in", "tokens_out",
    "retrieval_setting", "answer_grade"
]

PERMITTED_STATUSES = {"ok", "rate_limited", "timeout", "error"}
PERMITTED_ANSWER_GRADES = {"", "correct", "wrong", "refused_correctly", "refused_wrongly"}


def backup_existing_artifacts(out_dir: Path, logger) -> Dict[str, str]:
    """
    Preserves original baseline artifacts (load_experiment_report.json and
    load_experiments_results.csv) without overwriting any existing backup files.
    """
    preserved: Dict[str, str] = {}
    pairs = [
        ("load_experiment_report.json", "load_experiment_report_original.json"),
        ("load_experiments_results.csv", "load_experiments_results_original.csv"),
    ]
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for src_name, default_bak_name in pairs:
        src_path = out_dir / src_name
        if not src_path.exists():
            continue
        bak_path = out_dir / default_bak_name
        if bak_path.exists():
            stem = Path(default_bak_name).stem
            suffix = Path(default_bak_name).suffix
            bak_path = out_dir / f"{stem}_{ts_tag}{suffix}"
        shutil.copy2(src_path, bak_path)
        preserved[src_name] = str(bak_path)
        logger.info(f"Preserved original artifact: {src_path} -> {bak_path}")

    return preserved


def resolve_unique_corrected_paths(out_dir: Path) -> tuple[str, Path, Path]:
    """
    Generates a unique batch run ID and non-overwriting JSON/CSV paths so neither
    original baseline artifacts nor earlier corrected runs are ever overwritten.
    """
    base_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    counter = 0
    while True:
        suffix = f"_{counter}" if counter > 0 else ""
        batch_run_id = f"batch3_corrected_{base_ts}{suffix}"
        csv_path = out_dir / f"load_experiments_results_{batch_run_id}.csv"
        report_path = out_dir / f"load_experiment_report_{batch_run_id}.json"
        if not csv_path.exists() and not report_path.exists():
            return batch_run_id, report_path, csv_path
        counter += 1


def write_telemetry_csv_batch(rows: list[dict], filepath: Path, is_first_batch: bool) -> None:
    if not rows:
        return
    first_row = rows[0]
    aux_cols = [k for k in first_row.keys() if k not in MANDATORY_COLS]
    all_cols = MANDATORY_COLS + aux_cols

    mode = "w" if is_first_batch else "a"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode=mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        if is_first_batch:
            writer.writeheader()
        for r in rows:
            status_val = str(r.get("status", "error"))
            if status_val not in PERMITTED_STATUSES:
                status_val = "error"
            grade_val = str(r.get("answer_grade") or "")
            if status_val != "ok":
                grade_val = ""
            elif grade_val not in PERMITTED_ANSWER_GRADES:
                grade_val = ""
            safe_row = {col: r.get(col, "") for col in all_cols}
            safe_row["status"] = status_val
            safe_row["answer_grade"] = grade_val
            writer.writerow(safe_row)


def start_benchmark_server(port: int = 8013) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="critical")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    start_wait = time.time()
    while time.time() - start_wait < 5.0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return server, thread
        time.sleep(0.1)

    raise RuntimeError(f"Benchmark server failed to bind to port {port} in time.")


def calc_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    s = sorted(values)
    n = len(s)
    p95_idx = max(0, math.ceil(0.95 * n) - 1)
    return {
        "mean": round(sum(s) / n, 2),
        "median": round(statistics.median(s), 2),
        "p95": round(s[p95_idx], 2),
    }


async def safe_run_request(pipeline: RAGPipeline, **kwargs) -> PipelineExecutionResult:
    req_id = kwargs.get("request_id", "unknown")
    scenario = kwargs.get("scenario", "unknown")
    deadline_ms = float(kwargs.get("request_deadline_ms", 45000.0))
    start = time.time()
    deadline_mono = time.monotonic() + (deadline_ms / 1000.0)

    client = getattr(pipeline, "provider_client", None) or getattr(pipeline, "client", None)
    if client is not None and hasattr(client, "register_request_deadline"):
        client.register_request_deadline(req_id, deadline_mono)

    scope_token = start_request_scope(req_id, deadline_mono)
    try:
        return await pipeline.run_request(**kwargs)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        end = time.time()
        err_name = type(e).__name__
        status_cls = "error"
        if "Timeout" in err_name or "timeout" in str(e).lower():
            status_cls = "timeout"
        elif "429" in str(e) or "RateLimit" in err_name:
            status_cls = "rate_limited"

        telemetry_row = {
            "request_id": req_id, "scenario": scenario, "source": "stand-in",
            "start_time": start, "first_token_ms": "", "total_ms": (end - start) * 1000.0,
            "status": status_cls, "retries": 0, "tokens_in": 0, "tokens_out": 0,
            "retrieval_setting": pipeline.config["retrieval"].get("default_setting", ""),
            "answer_grade": "", "end_time": end, "scenario_start_time": start,
            "scenario_load_end_time": end, "scenario_completion_time": end,
            "scenario_load_duration_s": 0.0,
            "question_id": kwargs.get("eval_question").question_id if kwargs.get("eval_question") else "unknown",
            "model_id": pipeline.config["provider"].get("default_model", ""),
            "cache_hit": False, "retrieval_method": "hybrid_rrf", "chunk_size_tokens": 500,
            "candidates_fetched": 0, "pieces_passed": 0, "rerank_enabled": False,
            "provider_first_token_ms": "", "waiting_ms": 0.0, "working_ms": 0.0, "retrying_ms": 0.0,
            "local_queue_wait_ms": 0.0, "primary_provider_queue_wait_ms": 0.0, "query_proc_ms": 0.0,
            "retrieval_ms": 0.0, "rerank_ms": 0.0, "evidence_eval_ms": 0.0, "primary_gen_ms": 0.0,
            "det_verify_ms": 0.0, "primary_sem_verify_ms": 0.0, "stream_delivery_ms": 0.0,
            "failed_attempt_work_ms": 0.0, "retry_backoff_ms": 0.0, "retry_queue_wait_ms": 0.0,
            "correction_work_ms": 0.0, "local_queue_depth_peak": 0, "provider_queue_depth_peak": 0,
            "in_flight_peak": 0, "correction_attempts": 0, "total_provider_calls": 0,
            "http_status_code": 0, "failure_reason": f"unhandled_exception: {err_name}",
            "client_abandoned": False, "token_accounting_mode": "none", "estimated_tokens_total": 0,
            "actual_cost_usd": 0.0, "reference_cost_usd": 0.0, "call_trace_json": "[]",
            "limit_dimension": "", "limit_scope": "", "limit_documented_value": "",
            "limit_observed_value": "", "limit_evidence_source": "", "limit_confidence": "",
            "retry_after_ms": "", "observed_recovery_ms": "",
        }
        return PipelineExecutionResult(
            request_id=req_id, question=kwargs.get("question", ""), status=status_cls,
            abstained=False, verified=False, user_response_text="", streamed_chunks=[],
            generation_output=None, verification_output=None, retrieved_pieces=[],
            telemetry_row=telemetry_row,
        )
    finally:
        end_request_scope(scope_token)
        if client is not None and hasattr(client, "clear_request_deadline"):
            client.clear_request_deadline(req_id)


async def run_scenario_matrix(
    scenario_id: str, scenario_name: str, request_count: int, fault_inject: str | None,
    base_config: dict, registry: RetrievalEngineRegistry, csv_path: Path,
    is_first_scenario: bool, logger, batch_run_id: str
) -> list[dict]:
    strategies = ["none", "naive", "aware"]
    q1 = EVALUATION_QUESTIONS[0]
    results = []
    first_write = is_first_scenario

    for strategy in strategies:
        run_id = f"exp_{batch_run_id}_{scenario_id}_{strategy}"
        logger.info(f"Starting {scenario_id} [{strategy}] ({run_id}) - {request_count} reqs")

        async with httpx.AsyncClient() as hc:
            await hc.post("http://127.0.0.1:8013/reset")

        run_config = copy.deepcopy(base_config)
        run_config["pipeline"]["retry_strategy"] = strategy
        run_config["pipeline"]["max_local_concurrency"] = 100
        run_config["provider"]["max_provider_concurrency"] = 15

        run_config["provider"]["limit_rpm"] = 1_000_000
        run_config["provider"]["limit_tpm"] = 1_000_000
        run_config["provider"]["limit_rpd"] = 1_000_000
        run_config["provider"]["limit_tpd"] = 1_000_000
        run_config["provider"]["ttft_timeout_s"] = 15.0

        timeout_cfg = httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=10.0)
        client = GroqCompatibleClient(
            run_config,
            source="stand-in",
            http_client=httpx.AsyncClient(base_url="http://127.0.0.1:8013/openai/v1", timeout=timeout_cfg),
        )
        pipeline = RAGPipeline(run_config, registry, client, None)
        deadline_ms = 25000.0 if scenario_id == "s5_timeout" else 45000.0

        scenario_start_time = time.time()
        scenario_start_mono = time.perf_counter()
        tasks = []
        for i in range(request_count):
            req_id = f"{run_id}_r{i+1:03d}"
            fault_header = f"{fault_inject or 'none'};req_id={req_id}"
            task = asyncio.create_task(
                safe_run_request(
                    pipeline=pipeline,
                    question=q1.question,
                    request_id=req_id,
                    scenario=scenario_id,
                    eval_question=q1,
                    enable_cache=False,
                    fault_inject=fault_header,
                    request_deadline_ms=deadline_ms,
                )
            )
            tasks.append(task)

        scenario_load_end_time = time.time()
        scenario_load_end_mono = time.perf_counter()
        scenario_load_duration_s = round(scenario_load_end_mono - scenario_start_mono, 6)

        batch_results = await asyncio.gather(*tasks)
        scenario_completion_time = time.time()

        transport_success_count = 0
        transport_fail_count = 0
        correct_answer_count = 0
        wrong_answer_count = 0
        correct_refusal_count = 0
        wrong_refusal_count = 0
        ungraded_count = 0

        abandoned_count = 0
        total_transport_retries = 0
        tot_tokens_in = 0
        tot_tokens_out = 0

        succ_latencies = []
        succ_ttft = []
        failure_classes: Dict[str, int] = {}
        batch_rows = []

        for br in batch_results:
            row = br.telemetry_row

            row["scenario_start_time"] = scenario_start_time
            row["scenario_load_end_time"] = scenario_load_end_time
            row["scenario_completion_time"] = scenario_completion_time
            row["scenario_load_duration_s"] = scenario_load_duration_s
            row["run_batch_id"] = batch_run_id

            if row["status"] != "ok":
                row["answer_grade"] = ""

            batch_rows.append(row)

            if row["status"] == "ok":
                transport_success_count += 1
                succ_latencies.append(float(row["total_ms"]))
                if row.get("first_token_ms") not in ("", None):
                    succ_ttft.append(float(row["first_token_ms"]))
            else:
                transport_fail_count += 1
                reason = row.get("failure_reason", "unknown")
                failure_classes[reason] = failure_classes.get(reason, 0) + 1

            grade = row.get("answer_grade", "")
            if grade == "correct":
                correct_answer_count += 1
            elif grade == "wrong":
                wrong_answer_count += 1
            elif grade == "refused_correctly":
                correct_refusal_count += 1
            elif grade == "refused_wrongly":
                wrong_refusal_count += 1
            else:
                ungraded_count += 1

            if row.get("client_abandoned"):
                abandoned_count += 1
            total_transport_retries += int(row.get("retries", 0) or 0)
            tot_tokens_in += int(row.get("tokens_in", 0) or 0)
            tot_tokens_out += int(row.get("tokens_out", 0) or 0)

        write_telemetry_csv_batch(batch_rows, csv_path, first_write)
        first_write = False

        sched = pipeline.scheduler
        final_state = {
            "local_queue_depth": sched.local_queue_depth,
            "provider_queue_depth": sched.provider_queue_depth,
            "in_flight": sched.in_flight,
        }

        transport_success_rate = (transport_success_count / request_count) * 100.0 if request_count else 0.0
        avg_retries = round(total_transport_retries / request_count, 2) if request_count else 0.0

        lat_stats = calc_stats(succ_latencies)
        ttft_stats = calc_stats(succ_ttft)

        run_summary = {
            "run_id": run_id,
            "run_batch_id": batch_run_id,
            "scenario": scenario_id,
            "description": scenario_name,
            "retry_strategy": strategy,
            "fault_injected": fault_inject,
            "retrieval_setting": run_config["retrieval"].get("default_setting", ""),
            "provider_concurrency_configured": run_config["provider"]["max_provider_concurrency"],
            "local_concurrency_configured": run_config["pipeline"]["max_local_concurrency"],
            "request_count": request_count,
            "scenario_timing": {
                "scenario_start_time": scenario_start_time,
                "scenario_load_end_time": scenario_load_end_time,
                "scenario_completion_time": scenario_completion_time,
                "scenario_load_duration_s": scenario_load_duration_s,
            },
            "outcomes_transport": {
                "success_count": transport_success_count,
                "fail_count": transport_fail_count,
                "success_rate_pct": round(transport_success_rate, 2),
                "client_abandoned": abandoned_count,
                "failure_classifications": failure_classes,
            },
            "outcomes_answer_quality": {
                "correct_answer": correct_answer_count,
                "wrong_answer": wrong_answer_count,
                "correct_refusal": correct_refusal_count,
                "wrong_refusal": wrong_refusal_count,
                "ungraded_or_failed": ungraded_count,
            },
            "latency": {
                "avg_successful_latency_ms": lat_stats["mean"],
                "median_successful_latency_ms": lat_stats["median"],
                "p95_successful_latency_ms": lat_stats["p95"],
                "avg_successful_ttft_ms": ttft_stats["mean"],
            },
            "retries_and_tokens": {
                "total_transport_retries": total_transport_retries,
                "avg_retries_per_request": avg_retries,
                "total_tokens_in": tot_tokens_in,
                "total_tokens_out": tot_tokens_out,
            },
            "scheduler_metrics": {
                "peak_local_queue": sched.peak_local,
                "peak_provider_queue": sched.peak_provider,
                "peak_in_flight": sched.peak_in_flight,
                "final_state": final_state,
            },
        }

        results.append(run_summary)
        await client._external_client.aclose()

    return results


async def run_all_experiments():
    logger = get_logger("load_experiments")
    logger.info("Starting Corrected Load Experiments (Stateful Benchmark Provider Only)")

    out_dir = PROJECT_ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    preserved_files = backup_existing_artifacts(out_dir, logger)
    batch_run_id, report_path, csv_path = resolve_unique_corrected_paths(out_dir)

    server, thread = start_benchmark_server()

    try:
        config = load_config()
        config["provider"]["standin_base_url"] = "http://127.0.0.1:8013/openai/v1"
        indices_dir = PROJECT_ROOT / "data" / "indices"
        registry = RetrievalEngineRegistry(indices_dir, config)

        scenarios = [
            ("s1_baseline", "Baseline concurrent throughput", 50, None),
            ("s2_rpm", "Heavy RPM contention", 50, "429_rpm"),
            ("s3_disconnect", "Mid-stream disconnects", 20, "disconnect_mid_stream"),
            ("s4_tpd", "Daily token quota exhaustion", 10, "429_tpd"),
            ("s5_timeout", "Slow provider / high TTFT", 20, "timeout_ttft"),
        ]

        all_results = []
        is_first_scenario = True

        for sid, sname, count, fault in scenarios:
            res = await run_scenario_matrix(
                sid, sname, count, fault, config, registry, csv_path, is_first_scenario, logger, batch_run_id=batch_run_id
            )
            all_results.extend(res)
            is_first_scenario = False

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_batch_id": batch_run_id,
            "environment": "stateful-benchmark-provider",
            "seed_applied": False,
            "retrieval_setting_used": config["retrieval"].get("default_setting", ""),
            "retrieval_setting_rationale": (
                "Inherited configured default (500tok_12to4_rerank_yes) consistently across all 15 runs "
                "to stress-test upper-bound context size and reranking latency and preserve comparability "
                "with the original 450-request benchmark."
            ),
            "preserved_original_artifacts": preserved_files,
            "output_csv_path": str(csv_path),
            "output_report_path": str(report_path),
            "total_runs": len(all_results),
            "runs": all_results,
        }

        temp_path = report_path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        temp_path.replace(report_path)

        logger.info(f"Corrected load experiments complete [{batch_run_id}]. Report saved to {report_path}")
        logger.info(f"Corrected raw telemetry saved to {csv_path}")

    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


if __name__ == "__main__":
    asyncio.run(run_all_experiments())