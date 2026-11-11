import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, has_groq_api_key
from src.logger import get_logger
from src.telemetry import ALL_64_COLUMNS, MANDATORY_12_COLUMNS, TelemetryCSVWriter


def build_smoke_test_row(default_model: str) -> Dict[str, Any]:
    return {
        "request_id": "req_verify_001",
        "scenario": "steady",
        "source": "stand-in",
        "start_time": 1762850000.000,
        "first_token_ms": 980.0,
        "total_ms": 1000.0,
        "status": "ok",
        "retries": 0,
        "tokens_in": 1200,
        "tokens_out": 150,
        "retrieval_setting": "500tok_12to4_rerank_yes",
        "answer_grade": "correct",
        "end_time": 1762850001.000,
        "scenario_start_time": 1762850000.000,
        "scenario_load_end_time": 1762850060.000,
        "scenario_completion_time": 1762850062.000,
        "scenario_load_duration_s": 60.0,
        "question_id": "q01",
        "model_id": default_model,
        "cache_hit": False,
        "retrieval_method": "hybrid_rrf",
        "chunk_size_tokens": 500,
        "candidates_fetched": 12,
        "pieces_passed": 4,
        "rerank_enabled": True,
        "provider_first_token_ms": 210.0,
        "waiting_ms": 100.0,
        "working_ms": 900.0,
        "retrying_ms": 0.0,
        "local_queue_wait_ms": 5.0,
        "primary_provider_queue_wait_ms": 95.0,
        "query_proc_ms": 2.0,
        "retrieval_ms": 18.0,
        "rerank_ms": 25.0,
        "evidence_eval_ms": 1.0,
        "primary_gen_ms": 520.0,
        "det_verify_ms": 2.0,
        "primary_sem_verify_ms": 312.0,
        "stream_delivery_ms": 20.0,
        "failed_attempt_work_ms": 0.0,
        "retry_backoff_ms": 20.0 * 0.0,
        "retry_queue_wait_ms": 0.0,
        "correction_work_ms": 0.0,
        "local_queue_depth_peak": 1,
        "provider_queue_depth_peak": 2,
        "in_flight_peak": 2,
        "correction_attempts": 0,
        "total_provider_calls": 2,
        "http_status_code": 200,
        "failure_reason": "none",
        "client_abandoned": False,
        "token_accounting_mode": "exact_reported",
        "estimated_tokens_total": 0,
        "actual_cost_usd": 0.0,
        "reference_cost_usd": 0.000072,
        "call_trace_json": "[]",
        "limit_dimension": "",
        "limit_scope": "",
        "limit_documented_value": "",
        "limit_observed_value": "",
        "limit_evidence_source": "",
        "limit_confidence": "",
        "retry_after_ms": "",
        "observed_recovery_ms": "",
    }


def main() -> None:
    logger = get_logger("verify_env")
    logger.info(f"Python version: {sys.version.split()[0]}")

    required_dirs = [
        "configs",
        "data/papers",
        "data/indices",
        "data/cache",
        "results/raw",
        "src",
        "tests",
    ]
    for d in required_dirs:
        p = PROJECT_ROOT / d
        assert p.is_dir(), f"Missing directory: {d}"
    logger.info("Directory structure verified.")

    cfg = load_config()
    default_model = cfg["provider"]["default_model"]
    assert default_model == "openai/gpt-oss-20b", f"Unexpected default_model: {default_model}"

    papers = cfg["corpus"]["papers"]
    expected_papers = [
        ("1706.03762", "v7"),
        ("2410.00037", "v1"),
        ("2507.07955", "v2"),
    ]
    actual_papers = [(p["id"], p["version"]) for p in papers]
    assert actual_papers == expected_papers, (
        f"Corpus paper versions mismatch: expected {expected_papers}, got {actual_papers}"
    )
    logger.info(
        f"Config verified: default_model={default_model}, papers={[f'{pid} {ver}' for pid, ver in actual_papers]}"
    )

    logger.info(
        "Testing automatic logger redaction with dummy token gsk_1234567890abcdefghij",
        extra={"structured_data": {"Authorization": "Bearer gsk_1234567890abcdefghij", "ok": True}},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = Path(tmpdir) / "verify_results.csv"
        writer = TelemetryCSVWriter(tmp_csv)
        writer.write_row(build_smoke_test_row(default_model))
        assert tmp_csv.exists() and tmp_csv.stat().st_size > 0

    logger.info(
        f"Telemetry CSV writer & schema verified: {len(ALL_64_COLUMNS)} total columns (12 mandatory first: {MANDATORY_12_COLUMNS})."
    )
    logger.info(f"GROQ_API_KEY configured in .env: {has_groq_api_key()}")


if __name__ == "__main__":
    main()