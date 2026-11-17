"""Run a quota-conscious end-to-end answer-quality evaluation on Groq.

Runs 15 DISTINCT authored questions, allocated five per retrieval configuration
across three configurations. Retry strategy and transport retries are disabled.
Requests are sequential. The script stops immediately after a provider rate-limit
or authentication/permission response and preserves partial results.

This default is a modest 15-logical-request evaluation, not 15 questions per
configuration. Each logical pipeline request may invoke more than one provider
call (generation and semantic verification). The three groups use different
question IDs, so the comparison is not paired; the JSON records the exact IDs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, has_groq_api_key  # noqa: E402
from src.evaluation_questions import EVALUATION_QUESTIONS  # noqa: E402
from src.groq_client import GroqCompatibleClient, end_request_scope, start_request_scope  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.retrieval import RetrievalEngineRegistry  # noqa: E402

MANDATORY_COLUMNS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms",
    "total_ms", "status", "retries", "tokens_in", "tokens_out",
    "retrieval_setting", "answer_grade",
]

# 15 unique questions total, five per setting, with four answerable and one
# unanswerable question in each group. Question IDs are deliberately recorded.
EVAL_GROUPS: list[tuple[str, list[str]]] = [
    ("250tok_4to4_rerank_no", ["q01", "q04", "q07", "q10", "q13"]),
    ("500tok_4to4_rerank_no", ["q02", "q05", "q08", "q11", "q14"]),
    ("500tok_12to4_rerank_yes", ["q03", "q06", "q09", "q12", "q15"]),
]


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    idx = max(0, math.ceil((p / 100.0) * len(ordered)) - 1)
    return round(ordered[idx], 2)


def cite_payload(gen_output: Any) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    if not gen_output:
        return payload
    for claim in getattr(gen_output, "claims", []) or []:
        for citation in getattr(claim, "citations", []) or []:
            payload.append(
                {
                    "claim_text": getattr(claim, "claim_text", ""),
                    "chunk_id": getattr(citation, "chunk_id", ""),
                    "paper_id": getattr(citation, "paper_id", ""),
                    "arxiv_version": getattr(citation, "arxiv_version", ""),
                    "page_number": getattr(citation, "page_number", None),
                    "section_title": getattr(citation, "section_title", ""),
                    "locator_detail": getattr(citation, "locator_detail", ""),
                    "paragraph_index": getattr(citation, "paragraph_index", None),
                }
            )
    return payload


def write_outputs(csv_path: Path, json_path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    extra_cols = [
        "question_id", "question", "expected_abstained", "verified", "abstained",
        "answer_text", "http_status_code", "failure_reason", "token_accounting_mode",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANDATORY_COLUMNS + extra_cols)
        writer.writeheader()
        for rec in records:
            row = {key: rec.get(key, "") for key in MANDATORY_COLUMNS + extra_cols}
            if row["status"] != "ok":
                row["answer_grade"] = ""
            writer.writerow(row)
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(json_path)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for setting, qids in EVAL_GROUPS:
        subset = [r for r in records if r["retrieval_setting"] == setting]
        n = len(subset)
        successful = [r for r in subset if r["status"] == "ok"]
        grades = [r.get("answer_grade", "") for r in successful]
        correct = grades.count("correct")
        wrong = grades.count("wrong") + grades.count("refused_wrongly")
        refused_correctly = grades.count("refused_correctly")
        ttft = [float(r["first_token_ms"]) for r in successful if r.get("first_token_ms") not in ("", None)]
        total = [float(r["total_ms"]) for r in successful if r.get("total_ms") not in ("", None)]
        token_totals = [
            int(r.get("tokens_in", 0) or 0) + int(r.get("tokens_out", 0) or 0)
            for r in successful
        ]
        output[setting] = {
            "question_ids_planned": qids,
            "questions_completed": n,
            "successful_pipeline_requests": len(successful),
            "correct_and_cited_pct": round(correct / n * 100.0, 1) if n else None,
            "wrong_pct": round(wrong / n * 100.0, 1) if n else None,
            "correctly_refused_pct": round(refused_correctly / n * 100.0, 1) if n else None,
            "infrastructure_failures": n - len(successful),
            "tokens_per_successful_answer_mean": round(sum(token_totals) / len(token_totals), 1) if token_totals else None,
            "first_token_ms_success_only": {"n": len(ttft), "p50": pct(ttft, 50), "p95": pct(ttft, 95)},
            "total_ms_success_only": {"n": len(total), "p50": pct(total, 50), "p95": pct(total, 95)},
        }
    return output


async def evaluate_one(
    pipeline: RAGPipeline,
    client: GroqCompatibleClient,
    question: Any,
    setting: str,
    request_id: str,
    deadline_ms: float,
) -> dict[str, Any]:
    deadline_mono = time.monotonic() + deadline_ms / 1000.0
    client.register_request_deadline(request_id, deadline_mono)
    scope_token = start_request_scope(request_id, deadline_mono)
    try:
        result = await pipeline.run_request(
            question=question.question,
            request_id=request_id,
            scenario="retrieval_quality",
            retrieval_setting=setting,
            eval_question=question,
            request_deadline_ms=deadline_ms,
            enable_cache=False,
            fault_inject=f"none;req_id={request_id}",
        )
    finally:
        end_request_scope(scope_token)
        client.clear_request_deadline(request_id)

    row = dict(result.telemetry_row)
    answer_text = result.user_response_text or "".join(result.streamed_chunks)
    return {
        **{k: row.get(k, "") for k in MANDATORY_COLUMNS},
        "question_id": question.question_id,
        "question": question.question,
        "expected_abstained": question.expected_abstained,
        "verified": bool(result.verified),
        "abstained": bool(result.abstained),
        "answer_text": answer_text,
        "reference_answer": question.reference_answer,
        "citations": cite_payload(result.generation_output),
        "http_status_code": row.get("http_status_code", 0),
        "failure_reason": row.get("failure_reason", ""),
        "token_accounting_mode": row.get("token_accounting_mode", "none"),
    }


async def async_main(pause_s: float) -> int:
    logger = get_logger("answer_quality_eval")
    if not has_groq_api_key():
        logger.error("GROQ_API_KEY is missing. Set it in the project-root .env before running this evaluation.")
        return 2

    config = load_config()
    config.setdefault("pipeline", {})["retry_strategy"] = "none"
    config["pipeline"]["max_correction_attempts"] = 0
    config["pipeline"]["max_local_concurrency"] = 1
    config.setdefault("provider", {})["max_provider_concurrency"] = 1
    config["provider"]["max_transport_retries"] = 0

    indices_dir = PROJECT_ROOT / "data" / "indices"
    registry = RetrievalEngineRegistry(indices_dir, config)
    client = GroqCompatibleClient(
        config,
        source="real",
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=50.0, write=10.0, pool=10.0)
        ),
    )
    pipeline = RAGPipeline(config, registry, client, None)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = PROJECT_ROOT / "results" / "raw"
    csv_path = out_dir / f"answer_quality_{ts}.csv"
    json_path = out_dir / f"answer_quality_{ts}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    q_by_id = {q.question_id: q for q in EVALUATION_QUESTIONS}
    records: list[dict[str, Any]] = []
    planned_total = sum(len(ids) for _, ids in EVAL_GROUPS)
    stop_reason = "completed_all_planned_questions"
    deadline_ms = 60_000.0
    request_index = 0

    try:
        for setting, question_ids in EVAL_GROUPS:
            for qid in question_ids:
                request_index += 1
                question = q_by_id[qid]
                request_id = f"answer_quality_{ts}_{setting}_{qid}"
                logger.info("Running %s/%s: setting=%s question=%s", request_index, planned_total, setting, qid)
                try:
                    record = await evaluate_one(
                        pipeline=pipeline,
                        client=client,
                        question=question,
                        setting=setting,
                        request_id=request_id,
                        deadline_ms=deadline_ms,
                    )
                except Exception as exc:
                    record = {
                        "request_id": request_id,
                        "scenario": "retrieval_quality",
                        "source": "real",
                        "start_time": time.time(),
                        "first_token_ms": "",
                        "total_ms": "",
                        "status": "error",
                        "retries": 0,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "retrieval_setting": setting,
                        "answer_grade": "",
                        "question_id": qid,
                        "question": question.question,
                        "expected_abstained": question.expected_abstained,
                        "verified": False,
                        "abstained": False,
                        "answer_text": "",
                        "reference_answer": question.reference_answer,
                        "citations": [],
                        "http_status_code": 0,
                        "failure_reason": f"exception:{type(exc).__name__}:{exc}",
                        "token_accounting_mode": "none",
                    }
                records.append(record)
                current_summary = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "provider": "groq",
                    "model": config["provider"].get("default_model", ""),
                    "planned_logical_requests": planned_total,
                    "completed_logical_requests": len(records),
                    "retry_strategy": "none",
                    "max_transport_retries": 0,
                    "question_allocation": {s: ids for s, ids in EVAL_GROUPS},
                    "per_setting": aggregate(records),
                    "records": records,
                    "stop_reason": "in_progress",
                    "csv_path": str(csv_path),
                }
                write_outputs(csv_path, json_path, records, current_summary)

                status = record.get("status")
                http_status = int(record.get("http_status_code") or 0)
                if status == "rate_limited" or http_status == 429:
                    stop_reason = "stopped_on_http_429_no_retry"
                    logger.error("Provider returned HTTP 429. Stopping immediately; no further questions will be sent.")
                    break
                if http_status in (401, 403) or status == "error" and str(record.get("failure_reason", "")).startswith("auth"):
                    stop_reason = "stopped_on_auth_or_permission_error"
                    logger.error("Provider authentication/permission failure. Stopping the evaluation.")
                    break
                if pause_s > 0 and request_index < planned_total:
                    await asyncio.sleep(pause_s)
            if stop_reason != "completed_all_planned_questions":
                break
    finally:
        await client._external_client.aclose()

    final_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "groq",
        "model": config["provider"].get("default_model", ""),
        "planned_logical_requests": planned_total,
        "completed_logical_requests": len(records),
        "retry_strategy": "none",
        "max_transport_retries": 0,
        "question_allocation": {s: ids for s, ids in EVAL_GROUPS},
        "per_setting": aggregate(records),
        "records": records,
        "stop_reason": stop_reason,
        "note": (
            "15 unique authored questions are allocated five per setting. The question sets are distinct, "
            "not paired across settings. Each logical RAG request may involve multiple underlying provider calls. "
            "Latency percentiles use successful logical requests only."
        ),
        "csv_path": str(csv_path),
    }
    write_outputs(csv_path, json_path, records, final_summary)
    print(f"Answer-quality CSV: {csv_path}")
    print(f"Answer-quality JSON: {json_path}")
    print(f"Completed {len(records)}/{planned_total} logical requests; stop_reason={stop_reason}")
    return 0 if stop_reason == "completed_all_planned_questions" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause-seconds", type=float, default=1.0, help="Pause between logical requests (default: 1s).")
    args = parser.parse_args()
    if args.pause_seconds < 0:
        parser.error("--pause-seconds must be non-negative")
    return asyncio.run(async_main(args.pause_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
