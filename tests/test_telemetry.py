import csv
import io
import json
import logging
import pytest

from src.logger import RedactingJSONFormatter
from src.telemetry import (
    ALL_64_COLUMNS,
    MANDATORY_12_COLUMNS,
    TelemetryCSVWriter,
    validate_telemetry_row,
)


def make_valid_sample_row():
    return {
        "request_id": "req_test_001",
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
        "model_id": "openai/gpt-oss-20b",
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
        "retry_backoff_ms": 0.0,
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


def test_column_count_and_mandatory_order():
    assert len(MANDATORY_12_COLUMNS) == 12
    assert len(ALL_64_COLUMNS) == 64
    assert ALL_64_COLUMNS[:12] == [
        "request_id",
        "scenario",
        "source",
        "start_time",
        "first_token_ms",
        "total_ms",
        "status",
        "retries",
        "tokens_in",
        "tokens_out",
        "retrieval_setting",
        "answer_grade",
    ]


def test_valid_row_passes():
    row = make_valid_sample_row()
    validate_telemetry_row(row)


def test_partial_delivery_on_failed_request_passes():
    row = make_valid_sample_row()
    row["status"] = "timeout"
    row["answer_grade"] = ""
    row["stream_delivery_ms"] = 10.0
    row["working_ms"] = 890.0
    row["first_token_ms"] = 980.0
    row["total_ms"] = 990.0
    row["end_time"] = 1762850000.990
    validate_telemetry_row(row)


def test_end_time_mismatch_raises():
    row = make_valid_sample_row()
    row["end_time"] = 1762850001.050  # 1050 ms vs total_ms = 1000 ms
    with pytest.raises(ValueError, match="does not match"):
        validate_telemetry_row(row)


def test_failed_attempt_followed_by_successful_retry_partition():
    """
    Verifies non-overlapping accounting when Generation Call 1 fails after 150 ms,
    sleeps 250 ms in backoff, waits 40 ms in Stage 2 queue for retry, and then
    succeeds on the retry generation call (520 ms) + semantic verification (312 ms).
    """
    row = make_valid_sample_row()
    row["retries"] = 1
    row["total_provider_calls"] = 3
    row["token_accounting_mode"] = "mixed"
    row["estimated_tokens_total"] = 600

    # Failed attempt work, backoff, and retry queue wait go exclusively into retrying_ms
    row["failed_attempt_work_ms"] = 150.0
    row["retry_backoff_ms"] = 250.0
    row["retry_queue_wait_ms"] = 40.0
    row["correction_work_ms"] = 0.0
    row["retrying_ms"] = 440.0

    # Provider TTFT shifted by failed attempt (150) + backoff (250) + retry queue wait (40):
    # 100 + 150 + 250 + 40 + 210 = 750.0 ms
    row["provider_first_token_ms"] = 750.0

    # Primary waiting_ms (100.0) + working_ms (900.0) + retrying_ms (440.0) = 1440.0 ms
    row["total_ms"] = 1440.0
    row["first_token_ms"] = 1420.0  # total_ms (1440.0) - stream_delivery_ms (20.0)
    row["end_time"] = 1762850001.440

    validate_telemetry_row(row)


def test_failed_semantic_verification_followed_by_successful_retry_partition():
    """
    Verifies non-overlapping accounting when Generation Call 1 succeeds on the first try
    (primary_gen_ms = 520.0, provider_first_token_ms = 210.0), initial Semantic Verification
    Call 2 fails after 90 ms (counted only in failed_attempt_work_ms), backs off for 200 ms,
    waits 30 ms in the Stage 2 retry queue, and then succeeds on the retry verification
    call in 312 ms (counted only in primary_sem_verify_ms).
    """
    row = make_valid_sample_row()
    row["retries"] = 1
    row["total_provider_calls"] = 3
    row["tokens_in"] = 1800
    row["token_accounting_mode"] = "mixed"
    row["estimated_tokens_total"] = 300

    # Failed verification attempt (90 ms), backoff (200 ms), and retry queue wait (30 ms)
    row["failed_attempt_work_ms"] = 90.0
    row["retry_backoff_ms"] = 200.0
    row["retry_queue_wait_ms"] = 30.0
    row["correction_work_ms"] = 0.0
    row["retrying_ms"] = 320.0

    # Winning verification attempt remains in primary_sem_verify_ms (312.0) inside working_ms (900.0)
    assert row["primary_sem_verify_ms"] == 312.0
    assert row["working_ms"] == 900.0

    # waiting_ms (100.0) + working_ms (900.0) + retrying_ms (320.0) = 1320.0 ms
    row["total_ms"] = 1320.0
    row["first_token_ms"] = 1300.0  # total_ms (1320.0) - stream_delivery_ms (20.0)
    row["end_time"] = 1762850001.320

    validate_telemetry_row(row)


def test_csv_writer_header_once_and_append(tmp_path):
    csv_file = tmp_path / "results.csv"
    writer = TelemetryCSVWriter(csv_file)

    row1 = make_valid_sample_row()
    row2 = make_valid_sample_row()
    row2["request_id"] = "req_test_002"

    writer.write_row(row1)
    writer.write_row(row2)

    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
        lines = list(csv.reader(f))

    assert len(lines) == 3  # 1 header + 2 data rows
    assert lines[0] == ALL_64_COLUMNS
    assert lines[0][:12] == MANDATORY_12_COLUMNS
    assert lines[1][0] == "req_test_001"
    assert lines[2][0] == "req_test_002"
    assert lines[1][18] == "openai/gpt-oss-20b"


def test_logger_utc_and_automatic_redaction():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingJSONFormatter())

    test_logger = logging.getLogger("test_redacting_logger")
    test_logger.handlers = [handler]
    test_logger.setLevel(logging.INFO)
    test_logger.propagate = False

    record = test_logger.makeRecord(
        name="test_redacting_logger",
        level=logging.INFO,
        fn="test.py",
        lno=10,
        msg="Header Leak Check: Bearer gsk_secret1234567890abcdef",
        args=(),
        exc_info=None,
    )
    record.created = 1762850000.0  # 2025-11-11T08:33:20.000Z in exact UTC
    record.structured_data = {
        "Authorization": "Bearer gsk_secret1234567890abcdef",
        "nested": {"groq_api_key": "gsk_secret1234567890abcdef"},
        "status": 200,
    }
    test_logger.handle(record)

    output = stream.getvalue().strip()
    assert "gsk_secret" not in output
    parsed = json.loads(output)
    assert parsed["timestamp"] == "2025-11-11T08:33:20.000Z"
    assert parsed["message"] == "Header Leak Check: [REDACTED_CREDENTIAL]"
    assert parsed["data"]["Authorization"] == "[REDACTED]"
    assert parsed["data"]["nested"]["groq_api_key"] == "[REDACTED]"
    assert parsed["data"]["status"] == 200