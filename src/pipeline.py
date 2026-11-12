"""
Phase 6 / Batch 1: Connected 6-Stage RAG Orchestration Pipeline.
Flow:
  Request Arrival
    -> Step 1: Deterministic Query Processing & Verified Cache Check
    -> Step 2: Hybrid Retrieval Tool Call
    -> Step 3: Evidence Evaluation & Conditional Deterministic Query Expansion
    -> Step 4: Structured Answer Generation (Groq / Provider Call 1)
    -> Step 5: Two-Tier Verification Gate (Deterministic Python Check + Pruned/Full Groq
               Semantic Verification, Refusal Expansion Guard, and Bounded 1-Attempt
               Regeneration Loop with Mandatory Re-Verification)
    -> Step 6: Verified Chunked Response Delivery & 64-Column Telemetry Emission.
"""

import asyncio
from dataclasses import dataclass, field
import json
import random
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from src.config import load_config
from src.evaluation_questions import EvalQuestion, grade_pipeline_response
from src.groq_client import GroqCompatibleClient, ProviderCallResult
from src.logger import get_logger
from src.prompts_and_schemas import (
    ClaimItem,
    GenerationOutput,
    VerificationOutput,
    build_generation_messages,
    build_semantic_verification_messages,
    parse_json_payload,
)
from src.query_and_evidence import (
    VerifiedAnswerCache,
    evaluate_evidence_sufficiency,
    process_query_deterministically,
)
from src.retrieval import RetrievalEngineRegistry, RetrievalResult, RetrievedPiece
from src.telemetry import validate_telemetry_row
from src.verification import (
    DeterministicValidationResult,
    format_verified_user_response,
    select_passages_for_semantic_verification,
    validate_citations_deterministically,
)


@dataclass
class PipelineExecutionResult:
    request_id: str
    question: str
    status: str  # "ok", "rate_limited", "timeout", "error"
    abstained: bool
    verified: bool
    user_response_text: str
    streamed_chunks: List[str]
    generation_output: Optional[GenerationOutput]
    verification_output: Optional[VerificationOutput]
    retrieved_pieces: List[RetrievedPiece]
    telemetry_row: Dict[str, Any]


class RAGPipeline:
    """
    End-to-end RAG assistant implementing strict grounding, two-tier citation and
    semantic verification, bounded 1-attempt correction, and verified streaming.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        retrieval_registry: RetrievalEngineRegistry,
        provider_client: GroqCompatibleClient,
        answer_cache: Optional[VerifiedAnswerCache] = None,
    ) -> None:
        self.config = config
        self.prov_cfg = config.get("provider", {})
        self.ret_cfg = config.get("retrieval", {})
        self.pipe_cfg = config.get("pipeline", {})
        self.registry = retrieval_registry
        self.client = provider_client
        self.source = provider_client.source  # Maps source telemetry to client instantiation
        self.cache = answer_cache
        self.logger = get_logger("rag_pipeline")

        self.max_transport_retries = int(self.prov_cfg.get("max_transport_retries", 2))
        self.retry_base_delay_ms = float(self.prov_cfg.get("retry_base_delay_ms", 400.0))
        self.retry_max_delay_ms = float(self.prov_cfg.get("retry_max_delay_ms", 4000.0))
        self.max_correction_attempts = int(self.pipe_cfg.get("max_correction_attempts", 1))
        self.stream_chunk_chars = int(self.pipe_cfg.get("stream_chunk_chars", 48))

        self.price_in_per_tok = (
            float(self.prov_cfg.get("reference_price_per_1m_input_usd", 0.10)) / 1_000_000.0
        )
        self.price_out_per_tok = (
            float(self.prov_cfg.get("reference_price_per_1m_output_usd", 0.50)) / 1_000_000.0
        )

    async def _call_provider_with_retries(
        self,
        messages: List[Dict[str, str]],
        stage: str,
        model_id: str,
        max_tokens: int,
        deadline_perf: float,
        reserve_after_ms: float,
        call_traces: List[Dict[str, Any]],
    ) -> Tuple[ProviderCallResult, float, float, int]:
        """
        Executes a provider call with bounded transport retries and stage-sliced deadline
        protection. Returns:
          (final_result, failed_work_ms_accum, backoff_ms_accum, transport_retries_used)
        """
        failed_work_ms = 0.0
        backoff_ms = 0.0
        retries_used = 0
        attempt = 0

        while True:
            now_perf = time.perf_counter()
            remaining_total_ms = (deadline_perf - now_perf) * 1000.0
            allowed_call_ms = remaining_total_ms - reserve_after_ms

            res = await self.client.execute_single_call(
                messages=messages,
                stage=stage,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=float(self.prov_cfg.get("temperature", 0.0)),
                call_timeout_ms=allowed_call_ms,
            )
            call_traces.append(res.to_trace_dict(call_idx=len(call_traces) + 1, wait_ms=0.0))

            if res.success:
                return res, failed_work_ms, backoff_ms, retries_used

            if not res.is_retriable or attempt >= self.max_transport_retries:
                return res, failed_work_ms, backoff_ms, retries_used

            if res.retry_after_ms is not None and res.retry_after_ms > 0:
                delay_ms = res.retry_after_ms + random.uniform(25.0, 100.0)
            else:
                exp_ms = min(
                    self.retry_max_delay_ms,
                    self.retry_base_delay_ms * (2 ** attempt),
                )
                delay_ms = exp_ms + random.uniform(0.0, 0.25 * exp_ms)

            now_after_fail = time.perf_counter()
            rem_ms = (deadline_perf - now_after_fail) * 1000.0
            min_needed_ms = delay_ms + reserve_after_ms + 150.0
            if rem_ms < min_needed_ms:
                return res, failed_work_ms, backoff_ms, retries_used

            failed_work_ms += res.work_ms
            retries_used += 1
            attempt += 1

            t_sleep0 = time.perf_counter()
            await asyncio.sleep(delay_ms / 1000.0)
            backoff_ms += (time.perf_counter() - t_sleep0) * 1000.0

    def _parse_generation_output(self, raw_content: str) -> Tuple[Optional[GenerationOutput], Optional[str]]:
        try:
            data = parse_json_payload(raw_content)
            return GenerationOutput(**data), None
        except Exception as exc:
            return None, f"Failed to parse structured GenerationOutput JSON: {exc}"

    def _parse_verification_output(self, raw_content: str) -> Tuple[Optional[VerificationOutput], Optional[str]]:
        try:
            data = parse_json_payload(raw_content)
            return VerificationOutput(**data), None
        except Exception as exc:
            return None, f"Failed to parse structured VerificationOutput JSON: {exc}"

    async def _stream_verified_text(self, text: str) -> AsyncIterator[str]:
        if not text:
            return
        step = max(16, self.stream_chunk_chars)
        for i in range(0, len(text), step):
            yield text[i : i + step]
            await asyncio.sleep(0)

    async def run_request(
        self,
        question: str,
        request_id: str = "req_0001",
        scenario: str = "retrieval_eval",
        retrieval_setting: Optional[str] = None,
        eval_question: Optional[EvalQuestion] = None,
        request_deadline_ms: Optional[float] = None,
        enable_cache: Optional[bool] = None,
        scenario_metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineExecutionResult:
        start_epoch = time.time()
        t_start = time.perf_counter()
        deadline_ms = (
            float(request_deadline_ms)
            if request_deadline_ms is not None
            else float(self.prov_cfg.get("request_deadline_ms", 45000.0))
        )
        deadline_perf = t_start + (deadline_ms / 1000.0)

        setting_name = retrieval_setting or str(
            self.ret_cfg.get("default_setting", "500tok_12to4_rerank_yes")
        )
        setting_cfg = self.ret_cfg["settings"][setting_name]
        model_id = str(self.prov_cfg.get("default_model", "openai/gpt-oss-20b"))
        use_cache = (
            bool(enable_cache)
            if enable_cache is not None
            else bool(self.pipe_cfg.get("enable_cache", False))
        )

        local_queue_wait_ms = 0.0
        primary_provider_queue_wait_ms = 0.0
        query_proc_ms = 0.0
        retrieval_ms = 0.0
        rerank_ms = 0.0
        evidence_eval_ms = 0.0
        primary_gen_ms = 0.0
        det_verify_ms = 0.0
        primary_sem_verify_ms = 0.0
        stream_delivery_ms = 0.0
        failed_attempt_work_ms = 0.0
        retry_backoff_ms = 0.0
        retry_queue_wait_ms = 0.0
        correction_work_ms = 0.0

        provider_first_token_ms: Optional[float] = None
        first_token_ms: Optional[float] = None
        transport_retries = 0
        correction_attempts = 0
        call_traces: List[Dict[str, Any]] = []
        cache_hit = False
        status = "ok"
        failure_reason = "none"
        http_status_code = 0
        client_abandoned = False
        gen_output: Optional[GenerationOutput] = None
        ver_output: Optional[VerificationOutput] = None
        det_valid = False
        verified_flag = False
        final_pieces: List[RetrievedPiece] = []
        user_response_text = ""
        streamed_chunks: List[str] = []

        t_q0 = time.perf_counter()
        pq = process_query_deterministically(question)
        if use_cache and self.cache is not None:
            cached = self.cache.get(pq.normalized_query, setting_name, model_id)
        else:
            cached = None
        query_proc_ms += (time.perf_counter() - t_q0) * 1000.0

        if cached is not None:
            cache_hit = True
            gen_output = GenerationOutput(**cached["generation_output"])
            ver_output = (
                VerificationOutput(**cached["verification_output"])
                if cached.get("verification_output")
                else None
            )
            det_valid = True
            verified_flag = True
            user_response_text = str(cached["user_response_text"])
        else:
            ret_res: RetrievalResult = self.registry.retrieve_for_setting(
                query=pq.normalized_query,
                setting_name=setting_name,
                paper_boost_ids=pq.detected_paper_ids or None,
            )
            retrieval_ms += ret_res.retrieval_ms
            rerank_ms += ret_res.rerank_ms
            final_pieces = ret_res.final_pieces

            t_ev0 = time.perf_counter()
            ev_res = evaluate_evidence_sufficiency(
                processed_query=pq,
                retrieval_result=ret_res,
                already_expanded=False,
                min_lexical_overlap=float(self.ret_cfg.get("evidence_min_lexical_overlap", 0.12)),
                expansion_threshold=float(self.ret_cfg.get("evidence_expansion_threshold", 0.32)),
            )
            evidence_eval_ms += (time.perf_counter() - t_ev0) * 1000.0

            ran_query_expansion = False
            if ev_res.needs_query_expansion:
                ran_query_expansion = True
                ret_exp = self.registry.retrieve_for_setting(
                    query=pq.expanded_query,
                    setting_name=setting_name,
                    paper_boost_ids=pq.detected_paper_ids or None,
                )
                retrieval_ms += ret_exp.retrieval_ms
                rerank_ms += ret_exp.rerank_ms
                final_pieces = ret_exp.final_pieces

                t_ev1 = time.perf_counter()
                ev_res = evaluate_evidence_sufficiency(
                    processed_query=pq,
                    retrieval_result=ret_exp,
                    already_expanded=True,
                    min_lexical_overlap=float(self.ret_cfg.get("evidence_min_lexical_overlap", 0.12)),
                    expansion_threshold=float(self.ret_cfg.get("evidence_expansion_threshold", 0.32)),
                )
                evidence_eval_ms += (time.perf_counter() - t_ev1) * 1000.0

            if ev_res.deterministic_refusal:
                gen_output = GenerationOutput(
                    abstained=True,
                    abstention_reason=ev_res.refusal_reason,
                    answer_text=f"Insufficient evidence in the research library: {ev_res.refusal_reason}",
                    claims=[],
                )
                det_valid = True
                verified_flag = True
                user_response_text = gen_output.answer_text
            else:
                gen_messages = build_generation_messages(question, final_pieces)
                gen_call, f_work, b_wait, r_used = await self._call_provider_with_retries(
                    messages=gen_messages,
                    stage="generate",
                    model_id=model_id,
                    max_tokens=int(self.prov_cfg.get("max_generation_tokens", 700)),
                    deadline_perf=deadline_perf,
                    reserve_after_ms=300.0,
                    call_traces=call_traces,
                )
                failed_attempt_work_ms += f_work
                retry_backoff_ms += b_wait
                transport_retries += r_used
                http_status_code = gen_call.http_status

                if gen_call.provider_first_token_ms is not None:
                    provider_first_token_ms = (
                        (gen_call.start_time - start_epoch) * 1000.0
                        + gen_call.provider_first_token_ms
                    )

                if not gen_call.success:
                    failed_attempt_work_ms += gen_call.work_ms
                    failure_reason = gen_call.failure_reason
                    if "limit" in failure_reason:
                        status = "rate_limited"
                    elif "timeout" in failure_reason or "deadline" in failure_reason:
                        status = "timeout"
                        client_abandoned = True
                    else:
                        status = "error"
                else:
                    primary_gen_ms += gen_call.work_ms
                    gen_output, parse_err = self._parse_generation_output(gen_call.content)

                    # Refusal Expansion Guard (Decision 2):
                    if (
                        gen_output is not None
                        and gen_output.abstained
                        and not ran_query_expansion
                        and pq.expanded_query != pq.normalized_query
                    ):
                        old_ids = {p.chunk.chunk_id for p in final_pieces}
                        ret_exp = self.registry.retrieve_for_setting(
                            query=pq.expanded_query,
                            setting_name=setting_name,
                            paper_boost_ids=pq.detected_paper_ids or None,
                        )
                        retrieval_ms += ret_exp.retrieval_ms
                        rerank_ms += ret_exp.rerank_ms
                        new_ids = {p.chunk.chunk_id for p in ret_exp.final_pieces}
                        if new_ids != old_ids:
                            final_pieces = ret_exp.final_pieces
                            correction_attempts += 1
                            regen_msgs = build_generation_messages(question, final_pieces)
                            regen_call, f_w2, b_w2, r_u2 = await self._call_provider_with_retries(
                                messages=regen_msgs,
                                stage="regenerate",
                                model_id=model_id,
                                max_tokens=int(self.prov_cfg.get("max_generation_tokens", 700)),
                                deadline_perf=deadline_perf,
                                reserve_after_ms=250.0,
                                call_traces=call_traces,
                            )
                            failed_attempt_work_ms += f_w2
                            retry_backoff_ms += b_w2
                            transport_retries += r_u2
                            http_status_code = regen_call.http_status
                            
                            # CRITICAL FIX: Safe abort on failed refusal regeneration
                            if not regen_call.success:
                                failed_attempt_work_ms += regen_call.work_ms
                                failure_reason = regen_call.failure_reason
                                if "limit" in failure_reason:
                                    status = "rate_limited"
                                elif "timeout" in failure_reason or "deadline" in failure_reason:
                                    status = "timeout"
                                    client_abandoned = True
                                else:
                                    status = "error"
                            else:
                                correction_work_ms += regen_call.work_ms
                                gen_output, parse_err = self._parse_generation_output(
                                    regen_call.content
                                )

                    if status == "ok":
                        t_dv0 = time.perf_counter()
                        if gen_output is None:
                            det_res = DeterministicValidationResult(
                                valid=False, errors=[parse_err or "Invalid generation JSON"]
                            )
                        else:
                            det_res = validate_citations_deterministically(gen_output, final_pieces)
                        det_verify_ms += (time.perf_counter() - t_dv0) * 1000.0

                        needs_correction = not det_res.valid
                        correction_feedback = "; ".join(det_res.errors)

                        if det_res.valid and gen_output is not None:
                            ver_pieces = select_passages_for_semantic_verification(
                                gen_output, final_pieces, det_res
                            )
                            ver_msgs = build_semantic_verification_messages(
                                question=question,
                                gen_output=gen_output,
                                pieces_for_verification=ver_pieces,
                                include_boundary_context=True,
                            )
                            ver_call, f_wv, b_wv, r_uv = await self._call_provider_with_retries(
                                messages=ver_msgs,
                                stage="verify",
                                model_id=model_id,
                                max_tokens=int(self.prov_cfg.get("max_verification_tokens", 400)),
                                deadline_perf=deadline_perf,
                                reserve_after_ms=50.0,
                                call_traces=call_traces,
                            )
                            failed_attempt_work_ms += f_wv
                            retry_backoff_ms += b_wv
                            transport_retries += r_uv
                            http_status_code = ver_call.http_status

                            if not ver_call.success:
                                failed_attempt_work_ms += ver_call.work_ms
                                failure_reason = ver_call.failure_reason
                                if "limit" in failure_reason:
                                    status = "rate_limited"
                                elif "timeout" in failure_reason or "deadline" in failure_reason:
                                    status = "timeout"
                                    client_abandoned = True
                                else:
                                    status = "error"
                            else:
                                primary_sem_verify_ms += ver_call.work_ms
                                ver_output, v_err = self._parse_verification_output(ver_call.content)
                                if ver_output is not None and ver_output.verified:
                                    det_valid = True
                                    verified_flag = True
                                else:
                                    needs_correction = True
                                    correction_feedback = (
                                        ver_output.feedback
                                        if ver_output is not None and ver_output.feedback
                                        else (v_err or "Semantic verification rejected unsupported claims.")
                                    )

                        if (
                            status == "ok"
                            and needs_correction
                            and correction_attempts < self.max_correction_attempts
                        ):
                            correction_attempts += 1
                            regen_msgs = build_generation_messages(
                                question=question,
                                pieces=final_pieces,
                                correction_feedback=correction_feedback,
                            )
                            regen_call, f_wr, b_wr, r_ur = await self._call_provider_with_retries(
                                messages=regen_msgs,
                                stage="regenerate",
                                model_id=model_id,
                                max_tokens=int(self.prov_cfg.get("max_generation_tokens", 700)),
                                deadline_perf=deadline_perf,
                                reserve_after_ms=250.0,
                                call_traces=call_traces,
                            )
                            failed_attempt_work_ms += f_wr
                            retry_backoff_ms += b_wr
                            transport_retries += r_ur
                            http_status_code = regen_call.http_status

                            if not regen_call.success:
                                failed_attempt_work_ms += regen_call.work_ms
                                failure_reason = regen_call.failure_reason
                                if "limit" in failure_reason:
                                    status = "rate_limited"
                                elif "timeout" in failure_reason or "deadline" in failure_reason:
                                    status = "timeout"
                                    client_abandoned = True
                                else:
                                    status = "error"
                            else:
                                correction_work_ms += regen_call.work_ms
                                gen_output, parse_err2 = self._parse_generation_output(
                                    regen_call.content
                                )
                                t_dv1 = time.perf_counter()
                                if gen_output is None:
                                    det_res2 = DeterministicValidationResult(
                                        valid=False, errors=[parse_err2 or "Invalid JSON on regeneration"]
                                    )
                                else:
                                    det_res2 = validate_citations_deterministically(
                                        gen_output, final_pieces
                                    )
                                det_verify_ms += (time.perf_counter() - t_dv1) * 1000.0

                                if det_res2.valid and gen_output is not None:
                                    ver_pieces2 = select_passages_for_semantic_verification(
                                        gen_output, final_pieces, det_res2
                                    )
                                    ver_msgs2 = build_semantic_verification_messages(
                                        question=question,
                                        gen_output=gen_output,
                                        pieces_for_verification=ver_pieces2,
                                        include_boundary_context=True,
                                    )
                                    ver_call2, f_wv2, b_wv2, r_uv2 = await self._call_provider_with_retries(
                                        messages=ver_msgs2,
                                        stage="reverify",
                                        model_id=model_id,
                                        max_tokens=int(self.prov_cfg.get("max_verification_tokens", 400)),
                                        deadline_perf=deadline_perf,
                                        reserve_after_ms=30.0,
                                        call_traces=call_traces,
                                    )
                                    failed_attempt_work_ms += f_wv2
                                    retry_backoff_ms += b_wv2
                                    transport_retries += r_uv2
                                    http_status_code = ver_call2.http_status

                                    if not ver_call2.success:
                                        failed_attempt_work_ms += ver_call2.work_ms
                                        failure_reason = ver_call2.failure_reason
                                        if "limit" in failure_reason:
                                            status = "rate_limited"
                                        elif "timeout" in failure_reason or "deadline" in failure_reason:
                                            status = "timeout"
                                            client_abandoned = True
                                        else:
                                            status = "error"
                                    else:
                                        correction_work_ms += ver_call2.work_ms
                                        ver_output, _ = self._parse_verification_output(ver_call2.content)
                                        if ver_output is not None and ver_output.verified:
                                            det_valid = True
                                            verified_flag = True

                        if status == "ok" and not verified_flag:
                            failure_reason = "verification_failed_twice"
                            gen_output = GenerationOutput(
                                abstained=True,
                                abstention_reason=(
                                    "Unable to verify full factual and citation support in the retrieved "
                                    "passages after bounded verification and regeneration."
                                ),
                                answer_text=(
                                    "Insufficient verified evidence in the research library: the assistant "
                                    "refuses to guess when claims cannot be strictly verified against the papers."
                                ),
                                claims=[],
                            )
                            det_valid = True
                            verified_flag = True

                        if status == "ok" and gen_output is not None:
                            user_response_text = format_verified_user_response(
                                gen_output, final_pieces
                            )
                            if use_cache and self.cache is not None and failure_reason == "none":
                                self.cache.put(
                                    pq.normalized_query,
                                    setting_name,
                                    model_id,
                                    {
                                        "generation_output": gen_output.model_dump(),
                                        "verification_output": (
                                            ver_output.model_dump() if ver_output else None
                                        ),
                                        "user_response_text": user_response_text,
                                    },
                                )

        if status == "ok" and user_response_text:
            t_deliv_start = time.perf_counter()
            async for chunk_str in self._stream_verified_text(user_response_text):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t_start) * 1000.0
                streamed_chunks.append(chunk_str)
            t_deliv_end = time.perf_counter()
            raw_delivery_ms = (t_deliv_end - t_deliv_start) * 1000.0
        else:
            raw_delivery_ms = 0.0

        pre_delivery_waiting_ms = local_queue_wait_ms + primary_provider_queue_wait_ms
        pre_delivery_retrying_ms = (
            failed_attempt_work_ms
            + retry_backoff_ms
            + retry_queue_wait_ms
            + correction_work_ms
        )
        pre_delivery_working_ms = (
            query_proc_ms
            + retrieval_ms
            + rerank_ms
            + evidence_eval_ms
            + primary_gen_ms
            + det_verify_ms
            + primary_sem_verify_ms
        )

        end_epoch = time.time()
        wall_total_ms = max(
            (end_epoch - start_epoch) * 1000.0,
            pre_delivery_waiting_ms + pre_delivery_retrying_ms + pre_delivery_working_ms + raw_delivery_ms,
        )
        end_epoch = start_epoch + (wall_total_ms / 1000.0)

        if first_token_ms is not None:
            first_token_ms = pre_delivery_waiting_ms + pre_delivery_retrying_ms + pre_delivery_working_ms
            stream_delivery_ms = max(0.0, wall_total_ms - first_token_ms)
        else:
            stream_delivery_ms = 0.0
            overhead = wall_total_ms - (
                pre_delivery_waiting_ms + pre_delivery_retrying_ms + pre_delivery_working_ms
            )
            if overhead > 0:
                query_proc_ms += overhead

        waiting_ms = local_queue_wait_ms + primary_provider_queue_wait_ms
        working_ms = (
            query_proc_ms
            + retrieval_ms
            + rerank_ms
            + evidence_eval_ms
            + primary_gen_ms
            + det_verify_ms
            + primary_sem_verify_ms
            + stream_delivery_ms
        )
        retrying_ms = (
            failed_attempt_work_ms
            + retry_backoff_ms
            + retry_queue_wait_ms
            + correction_work_ms
        )
        total_ms = waiting_ms + working_ms + retrying_ms

        total_tokens_in = sum(int(c["tokens_in"]) for c in call_traces)
        total_tokens_out = sum(int(c["tokens_out"]) for c in call_traces)
        modes = {c["accounting_mode"] for c in call_traces if c["accounting_mode"] != "none"}
        if not call_traces or not modes:
            token_mode = "none"
        elif modes == {"exact_reported"}:
            token_mode = "exact_reported"
        elif modes == {"partial_estimated"}:
            token_mode = "partial_estimated"
        elif "unknown" in modes:
            token_mode = "unknown"
        else:
            token_mode = "mixed"

        est_tokens_total = sum(
            int(c["tokens_in"]) + int(c["tokens_out"])
            for c in call_traces
            if c["accounting_mode"] in ("partial_estimated", "unknown")
        )
        ref_cost_usd = (
            total_tokens_in * self.price_in_per_tok
            + total_tokens_out * self.price_out_per_tok
        )

        if eval_question is not None:
            answer_grade = grade_pipeline_response(
                eval_q=eval_question,
                status=status,
                gen_output=gen_output,
                citation_valid=det_valid,
            )
        else:
            answer_grade = "" if status != "ok" else "wrong"

        telemetry_row: Dict[str, Any] = {
            "request_id": request_id,
            "scenario": scenario,
            "source": self.source,
            "start_time": start_epoch,
            "first_token_ms": first_token_ms if first_token_ms is not None else "",
            "total_ms": total_ms,
            "status": status,
            "retries": transport_retries,
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "retrieval_setting": setting_name,
            "answer_grade": answer_grade,
            "end_time": end_epoch,
            "scenario_start_time": float(scenario_metadata.get("scenario_start_time", start_epoch)) if scenario_metadata else start_epoch,
            "scenario_load_end_time": float(scenario_metadata.get("scenario_load_end_time", end_epoch)) if scenario_metadata else end_epoch,
            "scenario_completion_time": float(scenario_metadata.get("scenario_completion_time", end_epoch)) if scenario_metadata else end_epoch,
            "scenario_load_duration_s": float(scenario_metadata.get("scenario_load_duration_s", 0.0)) if scenario_metadata else 0.0,
            "question_id": eval_question.question_id if eval_question else "unknown",
            "model_id": model_id,
            "cache_hit": cache_hit,
            "retrieval_method": "hybrid_rrf",
            "chunk_size_tokens": int(setting_cfg.get("chunk_size_tokens", 500)),
            "candidates_fetched": int(setting_cfg.get("candidates_fetched", 12)),
            "pieces_passed": len(final_pieces),
            "rerank_enabled": bool(setting_cfg.get("rerank_enabled", True)),
            "provider_first_token_ms": provider_first_token_ms if provider_first_token_ms is not None else "",
            "waiting_ms": waiting_ms,
            "working_ms": working_ms,
            "retrying_ms": retrying_ms,
            "local_queue_wait_ms": local_queue_wait_ms,
            "primary_provider_queue_wait_ms": primary_provider_queue_wait_ms,
            "query_proc_ms": query_proc_ms,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "evidence_eval_ms": evidence_eval_ms,
            "primary_gen_ms": primary_gen_ms,
            "det_verify_ms": det_verify_ms,
            "primary_sem_verify_ms": primary_sem_verify_ms,
            "stream_delivery_ms": stream_delivery_ms,
            "failed_attempt_work_ms": failed_attempt_work_ms,
            "retry_backoff_ms": retry_backoff_ms,
            "retry_queue_wait_ms": retry_queue_wait_ms,
            "correction_work_ms": correction_work_ms,
            "local_queue_depth_peak": int(scenario_metadata.get("local_queue_depth_peak", 0)) if scenario_metadata else 0,
            "provider_queue_depth_peak": int(scenario_metadata.get("provider_queue_depth_peak", 0)) if scenario_metadata else 0,
            "in_flight_peak": int(scenario_metadata.get("in_flight_peak", 0)) if scenario_metadata else 0,
            "correction_attempts": correction_attempts,
            "total_provider_calls": len(call_traces),
            "http_status_code": http_status_code,
            "failure_reason": failure_reason,
            "client_abandoned": client_abandoned,
            "token_accounting_mode": token_mode,
            "estimated_tokens_total": est_tokens_total,
            "actual_cost_usd": 0.0,
            "reference_cost_usd": ref_cost_usd,
            "call_trace_json": json.dumps(call_traces, ensure_ascii=False),
            "limit_dimension": "",
            "limit_scope": "",
            "limit_documented_value": "",
            "limit_observed_value": "",
            "limit_evidence_source": "",
            "limit_confidence": "",
            "retry_after_ms": "",
            "observed_recovery_ms": "",
        }

        validate_telemetry_row(telemetry_row)

        return PipelineExecutionResult(
            request_id=request_id,
            question=question,
            status=status,
            abstained=gen_output.abstained if gen_output else False,
            verified=verified_flag,
            user_response_text=user_response_text,
            streamed_chunks=streamed_chunks,
            generation_output=gen_output,
            verification_output=ver_output,
            retrieved_pieces=final_pieces,
            telemetry_row=telemetry_row,
        )
