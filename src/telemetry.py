"""
Telemetry schema, non-overlapping latency partition validator, and CSV writer.

Explicit Non-Overlapping Stage & Attempt Accounting Rules:
1. waiting_ms = local_queue_wait_ms + primary_provider_queue_wait_ms
   - Captures queue wait on the initial/primary dispatch path only.
   - Any Stage 2 FIFO queue wait incurred by transport retries or semantic
     correction loops is recorded exclusively in retry_queue_wait_ms (inside retrying_ms).
2. working_ms = query_proc_ms + retrieval_ms + rerank_ms + evidence_eval_ms
                + primary_gen_ms + det_verify_ms + primary_sem_verify_ms + stream_delivery_ms
   - Captures active execution time of the winning primary stages only.
   - primary_gen_ms counts ONLY the successful generation attempt of the original
     draft (including when that winning attempt is a transport retry after an earlier
     failed attempt).
   - primary_sem_verify_ms counts ONLY the successful semantic verification attempt
     of the original draft (including when that winning attempt is a transport retry
     after an earlier failed verification call).
3. retrying_ms = failed_attempt_work_ms + retry_backoff_ms + retry_queue_wait_ms + correction_work_ms
   - failed_attempt_work_ms: active network/provider time of any generation, verification,
     or regeneration call attempt that failed (HTTP 429, 5xx, timeout, or mid-stream drop)
     prior to a retry. Never counted in primary_gen_ms, primary_sem_verify_ms, or correction_work_ms.
   - retry_backoff_ms: sleep duration in exponential backoff or shared quota cooldown.
   - retry_queue_wait_ms: Stage 2 FIFO queue wait for transport retries and correction calls.
   - correction_work_ms: active execution time of semantic correction/regeneration (Call 3)
     and its mandatory semantic re-verification (Call 4) when correction_attempts > 0.
4. End-to-end invariants:
   - total_ms == waiting_ms + working_ms + retrying_ms
   - total_ms == (end_time - start_time) * 1000.0
   - If at least one verified token was delivered to the user (first_token_ms != ""):
     first_token_ms + stream_delivery_ms == total_ms
   - If zero verified tokens were delivered to the user (first_token_ms == ""):
     stream_delivery_ms == 0.0
"""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

MANDATORY_12_COLUMNS: List[str] = [
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

AUXILIARY_52_COLUMNS: List[str] = [
    "end_time",
    "scenario_start_time",
    "scenario_load_end_time",
    "scenario_completion_time",
    "scenario_load_duration_s",
    "question_id",
    "model_id",
    "cache_hit",
    "retrieval_method",
    "chunk_size_tokens",
    "candidates_fetched",
    "pieces_passed",
    "rerank_enabled",
    "provider_first_token_ms",
    "waiting_ms",
    "working_ms",
    "retrying_ms",
    "local_queue_wait_ms",
    "primary_provider_queue_wait_ms",
    "query_proc_ms",
    "retrieval_ms",
    "rerank_ms",
    "evidence_eval_ms",
    "primary_gen_ms",
    "det_verify_ms",
    "primary_sem_verify_ms",
    "stream_delivery_ms",
    "failed_attempt_work_ms",
    "retry_backoff_ms",
    "retry_queue_wait_ms",
    "correction_work_ms",
    "local_queue_depth_peak",
    "provider_queue_depth_peak",
    "in_flight_peak",
    "correction_attempts",
    "total_provider_calls",
    "http_status_code",
    "failure_reason",
    "client_abandoned",
    "token_accounting_mode",
    "estimated_tokens_total",
    "actual_cost_usd",
    "reference_cost_usd",
    "call_trace_json",
    "limit_dimension",
    "limit_scope",
    "limit_documented_value",
    "limit_observed_value",
    "limit_evidence_source",
    "limit_confidence",
    "retry_after_ms",
    "observed_recovery_ms",
]

ALL_64_COLUMNS: List[str] = MANDATORY_12_COLUMNS + AUXILIARY_52_COLUMNS

VALID_SOURCES = {"real", "stand-in"}
VALID_STATUSES = {"ok", "rate_limited", "timeout", "error"}
VALID_GRADES = {"correct", "wrong", "refused_correctly", "refused_wrongly", ""}
VALID_TOKEN_MODES = {"exact_reported", "partial_estimated", "mixed", "unknown", "none"}

WAITING_COMPONENTS = [
    "local_queue_wait_ms",
    "primary_provider_queue_wait_ms",
]

WORKING_COMPONENTS = [
    "query_proc_ms",
    "retrieval_ms",
    "rerank_ms",
    "evidence_eval_ms",
    "primary_gen_ms",
    "det_verify_ms",
    "primary_sem_verify_ms",
    "stream_delivery_ms",
]

RETRYING_COMPONENTS = [
    "failed_attempt_work_ms",
    "retry_backoff_ms",
    "retry_queue_wait_ms",
    "correction_work_ms",
]


def validate_telemetry_row(row: Dict[str, Any], tol_ms: float = 0.5) -> None:
    missing = [c for c in ALL_64_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Missing required schema columns: {missing}")

    extra = [c for c in row.keys() if c not in ALL_64_COLUMNS]
    if extra:
        raise ValueError(f"Unexpected columns not in 64-column schema: {extra}")

    if row["source"] not in VALID_SOURCES:
        raise ValueError(f"Invalid source: {row['source']}")
    if row["status"] not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {row['status']}")
    if row["answer_grade"] not in VALID_GRADES:
        raise ValueError(f"Invalid answer_grade: {row['answer_grade']}")
    if row["token_accounting_mode"] not in VALID_TOKEN_MODES:
        raise ValueError(f"Invalid token_accounting_mode: {row['token_accounting_mode']}")

    for comp in WAITING_COMPONENTS + WORKING_COMPONENTS + RETRYING_COMPONENTS:
        if float(row[comp]) < 0.0:
            raise ValueError(f"Negative duration not allowed for {comp}: {row[comp]}")

    waiting_calc = sum(float(row[c]) for c in WAITING_COMPONENTS)
    if abs(float(row["waiting_ms"]) - waiting_calc) > tol_ms:
        raise ValueError(f"waiting_ms mismatch: {row['waiting_ms']} vs {waiting_calc}")

    working_calc = sum(float(row[c]) for c in WORKING_COMPONENTS)
    if abs(float(row["working_ms"]) - working_calc) > tol_ms:
        raise ValueError(f"working_ms mismatch: {row['working_ms']} vs {working_calc}")

    retrying_calc = sum(float(row[c]) for c in RETRYING_COMPONENTS)
    if abs(float(row["retrying_ms"]) - retrying_calc) > tol_ms:
        raise ValueError(f"retrying_ms mismatch: {row['retrying_ms']} vs {retrying_calc}")

    total_calc = float(row["waiting_ms"]) + float(row["working_ms"]) + float(row["retrying_ms"])
    if abs(float(row["total_ms"]) - total_calc) > tol_ms:
        raise ValueError(f"total_ms partition mismatch: {row['total_ms']} vs {total_calc}")

    wall_clock_ms = (float(row["end_time"]) - float(row["start_time"])) * 1000.0
    if abs(float(row["total_ms"]) - wall_clock_ms) > tol_ms:
        raise ValueError(
            f"total_ms ({row['total_ms']}) does not match (end_time - start_time) * 1000 ({wall_clock_ms})"
        )

    ft = row["first_token_ms"]
    if ft != "" and ft is not None:
        expected_ft = float(row["total_ms"]) - float(row["stream_delivery_ms"])
        if abs(float(ft) - expected_ft) > tol_ms:
            raise ValueError(
                f"first_token_ms + stream_delivery_ms != total_ms: {ft} + {row['stream_delivery_ms']} != {row['total_ms']}"
            )
    else:
        if float(row["stream_delivery_ms"]) != 0.0:
            raise ValueError("stream_delivery_ms must be 0.0 when first_token_ms is empty")


class TelemetryCSVWriter:
    """Validated CSV writer that preserves the exact 64-column order and writes headers once."""

    def __init__(self, csv_path: Union[str, Path], tol_ms: float = 0.5) -> None:
        self.csv_path = Path(csv_path)
        self.tol_ms = tol_ms
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_header(self) -> None:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(ALL_64_COLUMNS)
        else:
            with open(self.csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
                if existing_header != ALL_64_COLUMNS:
                    raise ValueError(
                        f"Existing CSV header in {self.csv_path} does not match ALL_64_COLUMNS."
                    )

    def write_row(self, row: Dict[str, Any]) -> None:
        validate_telemetry_row(row, tol_ms=self.tol_ms)
        self._ensure_header()
        ordered_values = [row[col] for col in ALL_64_COLUMNS]
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ordered_values)

    def write_rows(self, rows: Iterable[Dict[str, Any]]) -> int:
        row_list = list(rows)
        for r in row_list:
            validate_telemetry_row(r, tol_ms=self.tol_ms)
        self._ensure_header()
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in row_list:
                writer.writerow([r[col] for col in ALL_64_COLUMNS])
        return len(row_list)