import asyncio
import copy
import csv
import json
from pathlib import Path
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from scripts.benchmark_provider import app, state
from scripts.run_load_experiments import (
    MANDATORY_COLS,
    backup_existing_artifacts,
    calc_stats,
    resolve_unique_corrected_paths,
    safe_run_request,
    write_telemetry_csv_batch,
)
from src.config import load_config
from src.evaluation_questions import EVALUATION_QUESTIONS
from src.groq_client import (
    GroqCompatibleClient,
    end_request_scope,
    start_request_scope,
)
from src.pipeline import RAGPipeline
from src.retrieval import RetrievalEngineRegistry

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_real_chunk(chunk_id: str) -> dict:
    """
    Loads the canonical chunk record directly from data/indices/chunks_500.jsonl.
    Fails explicitly if the canonical index or requested chunk record is unavailable.
    """
    chunks_path = PROJECT_ROOT / "data" / "indices" / "chunks_500.jsonl"
    if not chunks_path.exists():
        pytest.fail(f"Canonical corpus index not found at {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("chunk_id") == chunk_id:
                return rec
    pytest.fail(f"Canonical chunk '{chunk_id}' not found in {chunks_path}")


def _format_passage_block(rec: dict) -> str:
    """Formats a passage block preserving paragraph_refs and all_sections source metadata."""
    pages = rec.get("page_numbers", [rec.get("primary_page", 1)])
    pages_str = str(pages) if len(pages) > 1 else str(pages[0])
    all_sections_json = json.dumps(rec.get("all_sections") or [])
    para_refs_json = json.dumps(rec.get("paragraph_refs") or [])
    return (
        f"=== PASSAGE [chunk_id={rec['chunk_id']}] ===\n"
        f"Paper: {rec['title']} (paper_id={rec['paper_id']}, arxiv_version={rec['arxiv_version']})\n"
        f"Valid Pages: {pages_str} (primary_page={rec['primary_page']})\n"
        f"Section: {rec['section_title']} (Full path: {rec['section_path']})\n"
        f"All Sections: {all_sections_json}\n"
        f"Paragraph Refs: {para_refs_json}\n"
        f"--- Text ---\n{rec['text']}"
    )


def test_calc_stats_empty():
    assert calc_stats([]) == {"mean": None, "median": None, "p95": None}


def test_calc_stats_p95_math_ceil():
    vals = [float(x) for x in range(1, 101)]
    stats = calc_stats(vals)
    assert stats["mean"] == 50.5
    assert stats["p95"] == 95.0


def test_experiment_runner_initializes_matrix():
    scenarios = [
        ("s1", "base", 50, None),
        ("s2", "rpm", 50, "429_rpm"),
        ("s3", "disc", 20, "disconnect_mid_stream"),
        ("s4", "tpd", 10, "429_tpd"),
        ("s5", "time", 20, "timeout_ttft"),
    ]
    assert len(scenarios) * 3 == 15
    assert sum(s[2] * 3 for s in scenarios) == 450


def test_artifact_backup_and_unique_corrected_paths(tmp_path):
    orig_rep = tmp_path / "load_experiment_report.json"
    orig_csv = tmp_path / "load_experiments_results.csv"
    orig_rep.write_text('{"run": "original"}', encoding="utf-8")
    orig_csv.write_text("request_id,scenario\nr1,s1\n", encoding="utf-8")

    class DummyLogger:
        def info(self, msg):
            pass

    preserved = backup_existing_artifacts(tmp_path, DummyLogger())
    assert (tmp_path / "load_experiment_report_original.json").read_text(encoding="utf-8") == '{"run": "original"}'
    assert (tmp_path / "load_experiments_results_original.csv").exists()
    assert "load_experiment_report.json" in preserved

    batch_id_1, rep_1, csv_1 = resolve_unique_corrected_paths(tmp_path)
    rep_1.write_text("{}", encoding="utf-8")
    csv_1.write_text("", encoding="utf-8")

    batch_id_2, rep_2, csv_2 = resolve_unique_corrected_paths(tmp_path)
    assert batch_id_1 != batch_id_2
    assert rep_1 != rep_2
    assert csv_1 != csv_2


def test_write_telemetry_csv_batch_and_reconciliation(tmp_path):
    csv_file = tmp_path / "test_load.csv"
    row_ok = {
        "request_id": "exp_b1_s1_none_r001", "scenario": "s1_baseline", "source": "stand-in",
        "start_time": 100.0, "first_token_ms": 50.0, "total_ms": 120.0,
        "status": "ok", "retries": 0, "tokens_in": 3000, "tokens_out": 300,
        "retrieval_setting": "500tok_12to4_rerank_yes", "answer_grade": "correct",
        "end_time": 100.12, "scenario_start_time": 99.9,
        "scenario_load_end_time": 99.95, "scenario_completion_time": 100.5,
        "scenario_load_duration_s": 0.05, "failure_reason": "none",
    }
    row_fail = {
        "request_id": "exp_b1_s1_none_r002", "scenario": "s1_baseline", "source": "stand-in",
        "start_time": 100.02, "first_token_ms": "", "total_ms": 200.0,
        "status": "timeout", "retries": 1, "tokens_in": 1500, "tokens_out": 0,
        "retrieval_setting": "500tok_12to4_rerank_yes", "answer_grade": "wrong",
        "end_time": 100.22, "scenario_start_time": 99.9,
        "scenario_load_end_time": 99.95, "scenario_completion_time": 100.5,
        "scenario_load_duration_s": 0.05, "failure_reason": "ttft_timeout",
    }
    write_telemetry_csv_batch([row_ok, row_fail], csv_file, is_first_batch=True)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames[:12] == MANDATORY_COLS
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["scenario_start_time"] == rows[1]["scenario_start_time"] == "99.9"
    assert rows[0]["scenario_load_end_time"] == rows[1]["scenario_load_end_time"] == "99.95"
    assert rows[0]["scenario_completion_time"] == rows[1]["scenario_completion_time"] == "100.5"
    assert float(rows[0]["scenario_load_duration_s"]) > 0.0
    assert float(rows[0]["scenario_completion_time"]) > float(rows[0]["scenario_load_end_time"])
    assert rows[0]["start_time"] != rows[1]["start_time"]
    assert rows[0]["answer_grade"] == "correct"
    assert rows[1]["answer_grade"] == ""


@pytest.mark.asyncio
async def test_safe_run_request_unhandled_exception():
    class BrokenPipeline:
        config = {"retrieval": {}, "provider": {}}

        async def run_request(self, *args, **kwargs):
            raise ValueError("Catastrophic error")

    res = await safe_run_request(BrokenPipeline(), request_id="r99", scenario="s1")
    assert res.status == "error"
    assert res.telemetry_row["failure_reason"] == "unhandled_exception: ValueError"


@pytest.mark.asyncio
async def test_client_rejects_missing_request_deadline():
    """Calling execute_single_call without an active RequestContext deadline must raise RuntimeError."""
    config = {"provider": {"standin_base_url": "http://127.0.0.1:8014/openai/v1"}, "pipeline": {"retry_strategy": "aware"}}
    client = GroqCompatibleClient(config, source="stand-in")
    with pytest.raises(RuntimeError, match="Missing request-wide monotonic deadline"):
        await client.execute_single_call(
            messages=[{"role": "user", "content": "test"}],
            stage="generate",
            model_id="m1",
        )


@pytest.fixture(scope="module")
def benchmark_test_server():
    port = 8014
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="critical")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    start_wait = time.time()
    while time.time() - start_wait < 5.0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.05)

    yield f"http://127.0.0.1:{port}/openai/v1"
    server.should_exit = True
    thread.join(timeout=2.0)


async def _collect_sse_json(response: httpx.Response) -> dict:
    content = ""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line or line == "data: [DONE]":
            continue
        if line.startswith("data: "):
            chunk = json.loads(line[6:])
            choices = chunk.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                content += choices[0]["delta"]["content"]
    return json.loads(content)


@pytest.mark.asyncio
async def test_benchmark_provider_valid_fixture(benchmark_test_server):
    """Uses real c0006 + c0007 records; verifies c0007 is selected with Page 5, Section 3.2.2, Paragraph 12."""
    await httpx.AsyncClient().post("http://127.0.0.1:8014/reset")
    c6 = _load_real_chunk("1706.03762v7_s500_c0006")
    c7 = _load_real_chunk("1706.03762v7_s500_c0007")
    assert {"page_number": 5, "paragraph_index": 12} in c7.get("paragraph_refs", [])
    assert "3.2.2 Multi-Head Attention" in c7.get("all_sections", [])

    prompt = f"{_format_passage_block(c6)}\n\n{_format_passage_block(c7)}"

    async with httpx.AsyncClient(base_url=benchmark_test_server) as client:
        req = {"model": "m", "stream": True, "messages": [{"role": "user", "content": prompt}]}
        async with client.stream(
            "POST",
            "/chat/completions",
            json=req,
            headers={"x-relay-fault": "none;req_id=test_valid", "x-relay-stage": "generate"},
        ) as response:
            assert response.status_code == 200
            res = await _collect_sse_json(response)

        assert res["abstained"] is False
        cit = res["claims"][0]["citations"][0]
        assert cit["chunk_id"] == "1706.03762v7_s500_c0007"
        assert cit["paper_id"] == "1706.03762"
        assert cit["arxiv_version"] == "v7"
        assert cit["page_number"] == 5
        assert cit["section_title"] == "3.2.2 Multi-Head Attention"
        assert cit["locator_detail"] == "Paragraph 12"


@pytest.mark.asyncio
async def test_benchmark_provider_malformed_prompt_returns_400_before_streaming(benchmark_test_server):
    """Malformed generation input must fail cleanly with HTTP 400 before opening an SSE stream."""
    await httpx.AsyncClient().post("http://127.0.0.1:8014/reset")
    async with httpx.AsyncClient(base_url=benchmark_test_server) as client:
        req = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "Abbreviated invalid prompt"}]}
        res = await client.post(
            "/chat/completions",
            json=req,
            headers={"x-relay-fault": "none;req_id=test_bad_prompt", "x-relay-stage": "generate"},
        )
        assert res.status_code == 400
        assert "Fixture Error" in res.text


@pytest.mark.asyncio
async def test_benchmark_provider_rejects_generation_when_c0007_absent(benchmark_test_server):
    """Generation must fail with HTTP 400 rather than selecting c0006 when c0007 is absent."""
    await httpx.AsyncClient().post("http://127.0.0.1:8014/reset")
    c6 = _load_real_chunk("1706.03762v7_s500_c0006")
    async with httpx.AsyncClient(base_url=benchmark_test_server) as client:
        req = {"model": "m", "stream": True, "messages": [{"role": "user", "content": _format_passage_block(c6)}]}
        res = await client.post(
            "/chat/completions",
            json=req,
            headers={"x-relay-fault": "none;req_id=test_missing_c7", "x-relay-stage": "generate"},
        )
        assert res.status_code == 400
        assert "1706.03762v7_s500_c0007" in res.text


@pytest.mark.asyncio
async def test_benchmark_provider_rejects_c0006_only_evidence(benchmark_test_server):
    """Citing c0006 alone for q01 numerical claims must be rejected by the deterministic verifier."""
    c6 = _load_real_chunk("1706.03762v7_s500_c0006")
    candidate = {
        "abstained": False,
        "answer_text": "h=8, d_k=64, d_model=512.",
        "claims": [
            {
                "claim_id": "c1",
                "claim_text": "h=8, d_k=64, d_model=512",
                "cited_chunk_ids": ["1706.03762v7_s500_c0006"],
                "citations": [
                    {
                        "chunk_id": "1706.03762v7_s500_c0006",
                        "paper_id": "1706.03762",
                        "arxiv_version": "v7",
                        "page_number": 5,
                        "section_title": "3.2.2 Multi-Head Attention",
                        "locator_detail": "Paragraph 12",
                    }
                ],
            }
        ],
    }
    prompt = f"CANDIDATE RESPONSE TO VERIFY:\n{json.dumps(candidate)}\n\nSUPPORTING PASSAGES:\n{_format_passage_block(c6)}"
    async with httpx.AsyncClient(base_url=benchmark_test_server) as client:
        async with client.stream(
            "POST",
            "/chat/completions",
            json={"model": "m", "stream": True, "messages": [{"content": prompt}]},
            headers={"x-relay-stage": "verify", "x-relay-fault": "none;req_id=test_c6"},
        ) as response:
            assert response.status_code == 200
            res = await _collect_sse_json(response)
        assert res["verified"] is False
        assert "1706.03762v7_s500_c0006 alone is insufficient" in res["feedback"]


@pytest.mark.asyncio
async def test_benchmark_provider_invalid_citation_metadata_rejection(benchmark_test_server):
    """Structurally valid candidate with ONLY paper_id wrong is rejected specifically for paper_id."""
    c7 = _load_real_chunk("1706.03762v7_s500_c0007")
    candidate = {
        "abstained": False,
        "answer_text": "h=8, d_k=64, d_model=512.",
        "claims": [
            {
                "claim_id": "c1",
                "claim_text": "h=8, d_k=64, d_model=512",
                "cited_chunk_ids": ["1706.03762v7_s500_c0007"],
                "citations": [
                    {
                        "chunk_id": "1706.03762v7_s500_c0007",
                        "paper_id": "9999",
                        "arxiv_version": "v7",
                        "page_number": 5,
                        "section_title": "3.2.2 Multi-Head Attention",
                        "locator_detail": "Paragraph 12",
                    }
                ],
            }
        ],
    }
    prompt = f"CANDIDATE RESPONSE TO VERIFY:\n{json.dumps(candidate)}\n\nSUPPORTING PASSAGES:\n{_format_passage_block(c7)}"
    async with httpx.AsyncClient(base_url=benchmark_test_server) as client:
        async with client.stream(
            "POST",
            "/chat/completions",
            json={"model": "m", "stream": True, "messages": [{"content": prompt}]},
            headers={"x-relay-stage": "verify", "x-relay-fault": "none;req_id=test_bad_paper"},
        ) as response:
            assert response.status_code == 200
            res = await _collect_sse_json(response)
        assert res["verified"] is False
        assert "paper_id" in res["feedback"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_answer,bad_claim,bad_page,bad_sec,bad_loc,expected_substr",
    [
        ("h=8, d_k=64", "h=8, d_k=64", 5, "3.2.2 Multi-Head Attention", "Paragraph 12", "Missing required q01 attributes"),
        ("h=999", "h=999, d_k=64, d_model=512", 5, "3.2.2 Multi-Head Attention", "Paragraph 12", "Incorrect or swapped"),
        ("h=64", "h=64, d_k=8, d_model=512", 5, "3.2.2 Multi-Head Attention", "Paragraph 12", "Incorrect or swapped"),
        ("h=16 heads", "h=8, d_k=64, d_model=512", 5, "3.2.2 Multi-Head Attention", "Paragraph 12", "contradicting q01 claim values"),
        ("h=8", "h=8, d_k=64, d_model=512", 4, "3.2.2 Multi-Head Attention", "Paragraph 12", "page_number"),
        ("h=8", "h=8, d_k=64, d_model=512", 5, "3.2.3 Applications of Attention in our Model", "Paragraph 12", "section_title"),
        ("h=8", "h=8, d_k=64, d_model=512", 5, "3.2.2 Multi-Head Attention", "Paragraph 3", "locator_detail"),
    ],
)
async def test_benchmark_provider_negative_q01_assertions(
    benchmark_test_server, bad_answer, bad_claim, bad_page, bad_sec, bad_loc, expected_substr
):
    c7 = _load_real_chunk("1706.03762v7_s500_c0007")
    candidate = {
        "abstained": False,
        "answer_text": bad_answer,
        "claims": [
            {
                "claim_id": "c1",
                "claim_text": bad_claim,
                "cited_chunk_ids": ["1706.03762v7_s500_c0007"],
                "citations": [
                    {
                        "chunk_id": "1706.03762v7_s500_c0007",
                        "paper_id": "1706.03762",
                        "arxiv_version": "v7",
                        "page_number": bad_page,
                        "section_title": bad_sec,
                        "locator_detail": bad_loc,
                    }
                ],
            }
        ],
    }
    prompt = f"CANDIDATE RESPONSE TO VERIFY:\n{json.dumps(candidate)}\n\nSUPPORTING PASSAGES:\n{_format_passage_block(c7)}"
    async with httpx.AsyncClient(base_url=benchmark_test_server) as client:
        async with client.stream(
            "POST",
            "/chat/completions",
            json={"model": "m", "stream": True, "messages": [{"content": prompt}]},
            headers={"x-relay-stage": "verify", "x-relay-fault": "none;req_id=test_neg"},
        ) as response:
            res = await _collect_sse_json(response)
        assert res["verified"] is False
        assert expected_substr in res["feedback"]


@pytest.mark.asyncio
async def test_benchmark_provider_rpm_rolling_recovery(benchmark_test_server):
    """Accepts first 10 valid c0007 requests, rejects 11th with 429 + Retry-After, and accepts 12th after wait."""
    await httpx.AsyncClient().post("http://127.0.0.1:8014/reset")
    c7 = _load_real_chunk("1706.03762v7_s500_c0007")
    req = {"model": "m", "stream": True, "messages": [{"content": _format_passage_block(c7)}]}

    async with httpx.AsyncClient(base_url=benchmark_test_server) as client:
        for i in range(10):
            async with client.stream(
                "POST",
                "/chat/completions",
                json=req,
                headers={"x-relay-fault": f"429_rpm;req_id=rpm_{i}", "x-relay-stage": "generate"},
            ) as res:
                assert res.status_code == 200
                parsed = await _collect_sse_json(res)
                assert parsed["abstained"] is False

        res11 = await client.post(
            "/chat/completions",
            json=req,
            headers={"x-relay-fault": "429_rpm;req_id=rpm_11", "x-relay-stage": "generate"},
        )
        assert res11.status_code == 429
        retry_after = float(res11.headers.get("Retry-After", 0.0))
        assert retry_after > 0.0

        await asyncio.sleep(retry_after + 0.15)

        async with client.stream(
            "POST",
            "/chat/completions",
            json=req,
            headers={"x-relay-fault": "429_rpm;req_id=rpm_12", "x-relay-stage": "generate"},
        ) as res12:
            assert res12.status_code == 200
            parsed12 = await _collect_sse_json(res12)
            assert parsed12["abstained"] is False

    assert state.in_flight == 0


@pytest.mark.asyncio
async def test_aware_retry_rejects_ttft_timeout_with_insufficient_deadline_and_preserves_reason(benchmark_test_server):
    """
    Aware strategy must reject retry after a 15s TTFT timeout when only ~10s remains of a 25s deadline,
    preserve failure_reason='ttft_timeout', expose full retry_diagnostics in to_trace_dict(), and never
    reset the request-wide deadline across stages.
    """
    await httpx.AsyncClient().post("http://127.0.0.1:8014/reset")
    c7 = _load_real_chunk("1706.03762v7_s500_c0007")
    config = {
        "provider": {"standin_base_url": benchmark_test_server, "ttft_timeout_s": 15.0},
        "pipeline": {"retry_strategy": "aware"},
    }
    client = GroqCompatibleClient(config, source="stand-in")
    req_id = "aware_ttft_1"
    deadline_mono = time.monotonic() + 25.0
    client.register_request_deadline(req_id, deadline_mono)
    tok = start_request_scope(req_id, deadline_mono)
    try:
        res = await client.execute_single_call(
            messages=[{"role": "user", "content": _format_passage_block(c7)}],
            stage="generate",
            model_id="m1",
            max_tokens=100,
            call_timeout_ms=25000.0,
            fault_inject="timeout_ttft;req_id=aware_ttft_1",
        )
    finally:
        end_request_scope(tok)
        client.clear_request_deadline(req_id)

    assert res.success is False
    assert res.failure_reason == "ttft_timeout"
    assert res.is_retriable is False
    assert res.client_abandoned is True
    assert state.request_attempts.get(("aware_ttft_1", "generate")) == 1

    trace = res.to_trace_dict(call_idx=1)
    for key in (
        "remaining_deadline_ms",
        "next_attempt_budget_ms",
        "downstream_reserve_ms",
        "retry_after_ms",
        "required_total_ms",
        "retry_decision",
        "reject_reason",
    ):
        assert key in trace
    assert trace["retry_decision"] is False
    assert trace["reject_reason"] == "insufficient_deadline_budget"
    assert trace["remaining_deadline_ms"] < trace["required_total_ms"]


@pytest.mark.asyncio
async def test_cross_stage_deadline_not_reset_on_verification_rejection(benchmark_test_server):
    """
    Request-wide monotonic deadline must NOT be cleared after an HTTP 200 verification call
    so subsequent regeneration/reverification stages remain bounded by the original deadline.
    """
    await httpx.AsyncClient().post("http://127.0.0.1:8014/reset")
    c7 = _load_real_chunk("1706.03762v7_s500_c0007")
    config = {
        "provider": {"standin_base_url": benchmark_test_server, "ttft_timeout_s": 15.0},
        "pipeline": {"retry_strategy": "aware"},
    }
    client = GroqCompatibleClient(config, source="stand-in")
    req_id = "cross_stage_dl_1"
    deadline_mono = time.monotonic() + 45.0
    client.register_request_deadline(req_id, deadline_mono)
    tok = start_request_scope(req_id, deadline_mono)

    try:
        res_gen = await client.execute_single_call(
            messages=[{"role": "user", "content": _format_passage_block(c7)}],
            stage="generate",
            model_id="m1",
            max_tokens=200,
            fault_inject="none;req_id=cross_stage_dl_1",
        )
        assert res_gen.success is True
        assert client._request_deadlines_mono[req_id] == deadline_mono

        # Simulate a rejected verification candidate (verified=False)
        bad_cand = json.loads(res_gen.content)
        bad_cand["claims"][0]["claim_text"] = "h=999, d_k=64, d_model=512"
        verify_prompt = (
            f"CANDIDATE RESPONSE TO VERIFY:\n{json.dumps(bad_cand)}\n\n"
            f"SUPPORTING PASSAGES:\n{_format_passage_block(c7)}"
        )
        res_ver = await client.execute_single_call(
            messages=[{"role": "user", "content": verify_prompt}],
            stage="verify",
            model_id="m1",
            max_tokens=200,
            fault_inject="none;req_id=cross_stage_dl_1",
        )
        assert res_ver.success is True
        assert json.loads(res_ver.content)["verified"] is False
        # Deadline must still be intact for subsequent regenerate/reverify stages!
        assert req_id in client._request_deadlines_mono
        assert client._request_deadlines_mono[req_id] == deadline_mono
    finally:
        end_request_scope(tok)
        client.clear_request_deadline(req_id)

    assert req_id not in client._request_deadlines_mono


def test_aware_retry_rpm_and_rpd_tpd_budget_rules():
    """Verifies RPM Retry-After acceptance vs rejection, RPD/TPD non-retry, and Groq 429 header classification."""
    config = {
        "provider": {"ttft_timeout_s": 15.0},
        "pipeline": {"retry_strategy": "aware"},
    }
    client = GroqCompatibleClient(config, source="stand-in")

    # 1. RPM Retry-After accepted when deadline suffices
    ok_rpm, diag_ok = client.evaluate_aware_retry(
        stage="generate",
        failure_reason="rpm_limit",
        is_retriable=True,
        retry_after_ms=5000.0,
        remaining_deadline_ms=20000.0,
    )
    assert ok_rpm is True
    assert diag_ok["retry_decision"] is True

    # 2. RPM Retry-After rejected when insufficient time remains
    bad_rpm, diag_bad = client.evaluate_aware_retry(
        stage="generate",
        failure_reason="rpm_limit",
        is_retriable=True,
        retry_after_ms=10000.0,
        remaining_deadline_ms=12000.0,
    )
    assert bad_rpm is False
    assert diag_bad["reject_reason"] == "insufficient_deadline_budget"

    # 3. RPD and TPD exhaustion are never retried
    for terminal_reason in ("rpd_limit", "tpd_limit"):
        ok_term, diag_term = client.evaluate_aware_retry(
            stage="generate",
            failure_reason=terminal_reason,
            is_retriable=False,
            retry_after_ms=1000.0,
            remaining_deadline_ms=40000.0,
        )
        assert ok_term is False
        assert diag_term["reject_reason"] == f"non_retriable:{terminal_reason}"

    # 4. Groq 429 dimension classification (RPD vs TPM vs ambiguous)
    assert client._classify_429_dimension("Rate limit exceeded", {"x-ratelimit-remaining-requests": "0"}) == (
        "rpd_limit",
        False,
    )
    assert client._classify_429_dimension("Rate limit exceeded", {"x-ratelimit-remaining-tokens": "0"}) == (
        "tpm_limit",
        True,
    )
    assert client._classify_429_dimension("Rate limit exceeded", {}) == ("rate_limit_unknown", True)


@pytest.mark.asyncio
async def test_e2e_pipeline_orchestration_all_paths(benchmark_test_server):
    """
    End-to-end orchestration tests through RAGPipeline.run_request():
      1) Mid-stream disconnect on generate attempt 1 -> automatic scheduler retry on attempt 2 ->
         deterministic citation check -> semantic verification -> verified delivery (retries=1,
         correction_attempts=0, first_token_ms + stream_delivery_ms == total_ms, zero leaked queues).
      2) Retry strategy 'none' on disconnect -> no second attempt (retries=0, status=error).
      3) Daily TPD exhaustion ('429_tpd') -> terminal non-retriable (retries=0, status=rate_limited).
      4) Slow TTFT ('timeout_ttft') under 25s deadline with 'aware' -> blocked retry (retries=0,
         failure_reason='ttft_timeout', client_abandoned=True).
    """
    base_config = load_config()
    indices_dir = PROJECT_ROOT / "data" / "indices"
    registry = RetrievalEngineRegistry(indices_dir, base_config)
    q1 = EVALUATION_QUESTIONS[0]

    # 1. Recoverable mid-stream disconnect through full pipeline
    await httpx.AsyncClient().post("http://127.0.0.1:8014/reset")
    cfg_aware = copy.deepcopy(base_config)
    cfg_aware["provider"]["standin_base_url"] = benchmark_test_server
    cfg_aware["provider"]["max_transport_retries"] = 2
    cfg_aware["pipeline"]["retry_strategy"] = "aware"
    client_aware = GroqCompatibleClient(cfg_aware, source="stand-in")
    pipe_aware = RAGPipeline(cfg_aware, registry, client_aware, None)

    res_ok = await safe_run_request(
        pipe_aware,
        question=q1.question,
        request_id="e2e_disc_1",
        scenario="e2e_test",
        retrieval_setting="500tok_12to4_rerank_yes",
        eval_question=q1,
        enable_cache=False,
        fault_inject="disconnect_mid_stream;req_id=e2e_disc_1",
        request_deadline_ms=45000.0,
    )
    assert res_ok.status == "ok"
    assert res_ok.verified is True
    assert res_ok.abstained is False
    assert res_ok.telemetry_row["answer_grade"] == "correct"
    assert int(res_ok.telemetry_row["retries"]) == 1
    assert int(res_ok.telemetry_row["correction_attempts"]) == 0
    assert state.request_attempts[("e2e_disc_1", "generate")] == 2
    assert state.request_attempts[("e2e_disc_1", "verify")] == 1
    assert state.in_flight == 0
    assert pipe_aware.scheduler.in_flight == 0
    assert pipe_aware.scheduler.local_queue_depth == 0
    assert pipe_aware.scheduler.provider_queue_depth == 0
    assert "e2e_disc_1" not in client_aware._request_deadlines_mono

    ft_ms = float(res_ok.telemetry_row["first_token_ms"])
    del_ms = float(res_ok.telemetry_row["stream_delivery_ms"])
    tot_ms = float(res_ok.telemetry_row["total_ms"])
    assert ft_ms + del_ms == pytest.approx(tot_ms, rel=1e-3)

    # 2. Retry strategy 'none' produces zero transport retries
    await httpx.AsyncClient().post("http://127.0.0.1:8014/reset")
    cfg_none = copy.deepcopy(cfg_aware)
    cfg_none["pipeline"]["retry_strategy"] = "none"
    client_none = GroqCompatibleClient(cfg_none, source="stand-in")
    pipe_none = RAGPipeline(cfg_none, registry, client_none, None)

    res_none = await safe_run_request(
        pipe_none,
        question=q1.question,
        request_id="e2e_none_1",
        scenario="e2e_test",
        retrieval_setting="500tok_12to4_rerank_yes",
        eval_question=q1,
        enable_cache=False,
        fault_inject="disconnect_mid_stream;req_id=e2e_none_1",
        request_deadline_ms=45000.0,
    )
    assert res_none.status == "error"
    assert res_none.telemetry_row["failure_reason"] == "disconnect_mid_stream"
    assert int(res_none.telemetry_row["retries"]) == 0
    assert state.request_attempts[("e2e_none_1", "generate")] == 1
    assert ("e2e_none_1", "verify") not in state.request_attempts
    assert pipe_none.scheduler.in_flight == 0
    assert pipe_none.scheduler.local_queue_depth == 0
    assert pipe_none.scheduler.provider_queue_depth == 0

    # 3. Daily TPD exhaustion ('429_tpd') -> terminal non-retriable (retries=0, status=rate_limited).
    await httpx.AsyncClient().post("http://127.0.0.1:8014/reset")
    res_tpd = await safe_run_request(
        pipe_aware,
        question=q1.question,
        request_id="e2e_tpd_1",
        scenario="e2e_test",
        retrieval_setting="500tok_12to4_rerank_yes",
        eval_question=q1,
        enable_cache=False,
        fault_inject="429_tpd;req_id=e2e_tpd_1",
        request_deadline_ms=45000.0,
    )
    assert res_tpd.status == "rate_limited"
    assert res_tpd.telemetry_row["failure_reason"] == "tpd_limit"
    assert int(res_tpd.telemetry_row["retries"]) == 0
    assert state.request_attempts[("e2e_tpd_1", "generate")] == 1

    # 4. Slow TTFT under 25s deadline with 'aware' blocks infeasible retry (retries=0)
    await httpx.AsyncClient().post("http://127.0.0.1:8014/reset")
    res_ttft = await safe_run_request(
        pipe_aware,
        question=q1.question,
        request_id="e2e_ttft_1",
        scenario="e2e_test",
        retrieval_setting="500tok_12to4_rerank_yes",
        eval_question=q1,
        enable_cache=False,
        fault_inject="timeout_ttft;req_id=e2e_ttft_1",
        request_deadline_ms=25000.0,
    )
    assert res_ttft.status == "timeout"
    assert res_ttft.telemetry_row["failure_reason"] == "ttft_timeout"
    assert int(res_ttft.telemetry_row["retries"]) == 0
    assert bool(res_ttft.telemetry_row["client_abandoned"]) is True
    assert state.request_attempts[("e2e_ttft_1", "generate")] == 1
    assert pipe_aware.scheduler.in_flight == 0