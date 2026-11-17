"""
Phase 6 / Batch 2: Connected 6-Stage RAG Pipeline via Provider Scheduler.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from src.config import load_config
from src.evaluation_questions import EvalQuestion, grade_pipeline_response
from src.groq_client import GroqCompatibleClient
from src.logger import get_logger
from src.prompts_and_schemas import (
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
from src.scheduler import ProviderScheduler, RequestContext
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
    status: str
    abstained: bool
    verified: bool
    user_response_text: str
    streamed_chunks: List[str]
    generation_output: Optional[GenerationOutput]
    verification_output: Optional[VerificationOutput]
    retrieved_pieces: List[RetrievedPiece]
    telemetry_row: Dict[str, Any]


class RAGPipeline:
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
        self.source = provider_client.source
        self.cache = answer_cache
        self.logger = get_logger("rag_pipeline")

        self.scheduler = ProviderScheduler(config, self.client)
        
        self.max_correction_attempts = int(self.pipe_cfg.get("max_correction_attempts", 1))
        self.stream_chunk_chars = int(self.pipe_cfg.get("stream_chunk_chars", 48))
        self.price_in_per_tok = float(self.prov_cfg.get("reference_price_per_1m_input_usd", 0.10)) / 1_000_000.0
        self.price_out_per_tok = float(self.prov_cfg.get("reference_price_per_1m_output_usd", 0.50)) / 1_000_000.0

    def _parse_generation_output(self, raw_content: str) -> Tuple[Optional[GenerationOutput], Optional[str]]:
        try:
            data = parse_json_payload(raw_content)
            return GenerationOutput(**data), None
        except Exception as exc:
            return None, f"Failed to parse GenerationOutput JSON: {exc}"

    def _parse_verification_output(self, raw_content: str) -> Tuple[Optional[VerificationOutput], Optional[str]]:
        try:
            data = parse_json_payload(raw_content)
            return VerificationOutput(**data), None
        except Exception as exc:
            return None, f"Failed to parse VerificationOutput JSON: {exc}"

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
        fault_inject: Optional[str] = None,
    ) -> PipelineExecutionResult:
        start_epoch = time.time()
        t_start = time.perf_counter()
        deadline_ms = float(request_deadline_ms or self.prov_cfg.get("request_deadline_ms", 45000.0))
        deadline_perf = t_start + (deadline_ms / 1000.0)

        ctx = RequestContext(request_id, t_start, deadline_perf)
        
        setting_name = retrieval_setting or str(self.ret_cfg.get("default_setting", "500tok_12to4_rerank_yes"))
        setting_cfg = self.ret_cfg["settings"][setting_name]
        model_id = str(self.prov_cfg.get("default_model", "openai/gpt-oss-20b"))
        use_cache = bool(enable_cache) if enable_cache is not None else bool(self.pipe_cfg.get("enable_cache", False))

        query_proc_ms = 0.0
        retrieval_ms = 0.0
        rerank_ms = 0.0
        evidence_eval_ms = 0.0
        primary_gen_ms = 0.0
        det_verify_ms = 0.0
        primary_sem_verify_ms = 0.0
        stream_delivery_ms = 0.0

        provider_first_token_ms: Optional[float] = None
        first_token_ms: Optional[float] = None
        
        cache_hit = False
        status = "ok"
        failure_reason = "none"
        http_status_code = 0
        gen_output: Optional[GenerationOutput] = None
        ver_output: Optional[VerificationOutput] = None
        det_valid = False
        verified_flag = False
        final_pieces: List[RetrievedPiece] = []
        user_response_text = ""
        streamed_chunks: List[str] = []

        admitted = await self.scheduler.acquire_admission(ctx)
        if not admitted:
            status = "timeout" if ctx.client_abandoned else "error"
            failure_reason = "admission_shed_deadline" if ctx.client_abandoned else "queue_full"
            return self._finalize_telemetry(
                ctx, question, start_epoch, t_start, status, failure_reason, http_status_code, False, False, "", [],
                None, None, [], setting_name, setting_cfg, model_id, cache_hit, eval_question, scenario, scenario_metadata,
                query_proc_ms, retrieval_ms, rerank_ms, evidence_eval_ms, primary_gen_ms, det_verify_ms, primary_sem_verify_ms,
                0.0, None, None
            )

        try:
            t_q0 = time.perf_counter()
            pq = process_query_deterministically(question)
            cached = self.cache.get(pq.normalized_query, setting_name, model_id) if use_cache and self.cache else None
            query_proc_ms += (time.perf_counter() - t_q0) * 1000.0

            if cached is not None:
                cache_hit = True
                gen_output = GenerationOutput(**cached["generation_output"])
                ver_output = VerificationOutput(**cached["verification_output"]) if cached.get("verification_output") else None
                det_valid = True
                verified_flag = True
                user_response_text = str(cached["user_response_text"])
            else:
                ret_res = self.registry.retrieve_for_setting(pq.normalized_query, setting_name, pq.detected_paper_ids or None)
                retrieval_ms += ret_res.retrieval_ms
                rerank_ms += ret_res.rerank_ms
                final_pieces = ret_res.final_pieces

                t_ev0 = time.perf_counter()
                ev_res = evaluate_evidence_sufficiency(
                    pq, ret_res, False,
                    float(self.ret_cfg.get("evidence_min_lexical_overlap", 0.12)),
                    float(self.ret_cfg.get("evidence_expansion_threshold", 0.32)),
                )
                evidence_eval_ms += (time.perf_counter() - t_ev0) * 1000.0

                ran_query_expansion = False
                if ev_res.needs_query_expansion:
                    ran_query_expansion = True
                    ret_exp = self.registry.retrieve_for_setting(pq.expanded_query, setting_name, pq.detected_paper_ids or None)
                    retrieval_ms += ret_exp.retrieval_ms
                    rerank_ms += ret_exp.rerank_ms
                    final_pieces = ret_exp.final_pieces

                    t_ev1 = time.perf_counter()
                    ev_res = evaluate_evidence_sufficiency(
                        pq, ret_exp, True,
                        float(self.ret_cfg.get("evidence_min_lexical_overlap", 0.12)),
                        float(self.ret_cfg.get("evidence_expansion_threshold", 0.32)),
                    )
                    evidence_eval_ms += (time.perf_counter() - t_ev1) * 1000.0

                if ev_res.deterministic_refusal:
                    gen_output = GenerationOutput(
                        abstained=True, abstention_reason=ev_res.refusal_reason,
                        answer_text=f"Insufficient evidence in the research library: {ev_res.refusal_reason}", claims=[]
                    )
                    det_valid = True
                    verified_flag = True
                    user_response_text = gen_output.answer_text
                else:
                    gen_messages = build_generation_messages(question, final_pieces)
                    gen_call, gen_work = await self.scheduler.execute_call(
                        ctx, gen_messages, "generate", model_id, 
                        int(self.prov_cfg.get("max_generation_tokens", 1500)), 300.0, fault_inject
                    )
                    http_status_code = gen_call.http_status

                    if gen_call.provider_first_token_ms is not None:
                        provider_first_token_ms = ((gen_call.start_time - start_epoch) * 1000.0 + gen_call.provider_first_token_ms)

                    if not gen_call.success:
                        failure_reason = gen_call.failure_reason
                        status = "rate_limited" if "limit" in failure_reason else ("timeout" if "deadline" in failure_reason or "timeout" in failure_reason else "error")
                    else:
                        primary_gen_ms += gen_work
                        gen_output, parse_err = self._parse_generation_output(gen_call.content)

                        if gen_output and gen_output.abstained and not ran_query_expansion and pq.expanded_query != pq.normalized_query:
                            old_ids = {p.chunk.chunk_id for p in final_pieces}
                            ret_exp = self.registry.retrieve_for_setting(pq.expanded_query, setting_name, pq.detected_paper_ids or None)
                            retrieval_ms += ret_exp.retrieval_ms
                            rerank_ms += ret_exp.rerank_ms
                            new_ids = {p.chunk.chunk_id for p in ret_exp.final_pieces}
                            
                            if new_ids != old_ids:
                                final_pieces = ret_exp.final_pieces
                                ctx.correction_attempts += 1
                                regen_msgs = build_generation_messages(question, final_pieces)
                                regen_call, _ = await self.scheduler.execute_call(
                                    ctx, regen_msgs, "regenerate", model_id, 
                                    int(self.prov_cfg.get("max_generation_tokens", 1500)), 250.0, fault_inject
                                )
                                http_status_code = regen_call.http_status
                                
                                if not regen_call.success:
                                    failure_reason = regen_call.failure_reason
                                    status = "rate_limited" if "limit" in failure_reason else ("timeout" if "deadline" in failure_reason or "timeout" in failure_reason else "error")
                                else:
                                    gen_output, parse_err = self._parse_generation_output(regen_call.content)

                        if status == "ok":
                            t_dv0 = time.perf_counter()
                            det_res = validate_citations_deterministically(gen_output, final_pieces) if gen_output else DeterministicValidationResult(False, [parse_err or "Invalid generation JSON"])
                            det_verify_ms += (time.perf_counter() - t_dv0) * 1000.0

                            needs_correction = not det_res.valid
                            correction_feedback = "; ".join(det_res.errors)

                            if det_res.valid and gen_output is not None:
                                ver_pieces = select_passages_for_semantic_verification(gen_output, final_pieces, det_res)
                                ver_msgs = build_semantic_verification_messages(question, gen_output, ver_pieces, True)
                                ver_call, sem_work = await self.scheduler.execute_call(
                                    ctx, ver_msgs, "verify", model_id, 
                                    int(self.prov_cfg.get("max_verification_tokens", 400)), 50.0, fault_inject
                                )
                                http_status_code = ver_call.http_status

                                if not ver_call.success:
                                    failure_reason = ver_call.failure_reason
                                    status = "rate_limited" if "limit" in failure_reason else ("timeout" if "deadline" in failure_reason or "timeout" in failure_reason else "error")
                                else:
                                    primary_sem_verify_ms += sem_work
                                    ver_output, v_err = self._parse_verification_output(ver_call.content)
                                    if ver_output and ver_output.verified:
                                        det_valid = True
                                        verified_flag = True
                                    else:
                                        needs_correction = True
                                        correction_feedback = ver_output.feedback if ver_output and ver_output.feedback else (v_err or "Semantic verification rejected unsupported claims.")

                            if status == "ok" and needs_correction and ctx.correction_attempts < self.max_correction_attempts:
                                ctx.correction_attempts += 1
                                regen_msgs = build_generation_messages(question, final_pieces, correction_feedback)
                                regen_call, _ = await self.scheduler.execute_call(
                                    ctx, regen_msgs, "regenerate", model_id, 
                                    int(self.prov_cfg.get("max_generation_tokens", 1500)), 250.0, fault_inject
                                )
                                http_status_code = regen_call.http_status

                                if not regen_call.success:
                                    failure_reason = regen_call.failure_reason
                                    status = "rate_limited" if "limit" in failure_reason else ("timeout" if "deadline" in failure_reason or "timeout" in failure_reason else "error")
                                else:
                                    gen_output, parse_err2 = self._parse_generation_output(regen_call.content)
                                    t_dv1 = time.perf_counter()
                                    det_res2 = validate_citations_deterministically(gen_output, final_pieces) if gen_output else DeterministicValidationResult(False, [parse_err2 or "Invalid JSON on regeneration"])
                                    det_verify_ms += (time.perf_counter() - t_dv1) * 1000.0

                                    if det_res2.valid and gen_output is not None:
                                        ver_pieces2 = select_passages_for_semantic_verification(gen_output, final_pieces, det_res2)
                                        ver_msgs2 = build_semantic_verification_messages(question, gen_output, ver_pieces2, True)
                                        ver_call2, _ = await self.scheduler.execute_call(
                                            ctx, ver_msgs2, "reverify", model_id, 
                                            int(self.prov_cfg.get("max_verification_tokens", 400)), 30.0, fault_inject
                                        )
                                        http_status_code = ver_call2.http_status

                                        if not ver_call2.success:
                                            failure_reason = ver_call2.failure_reason
                                            status = "rate_limited" if "limit" in failure_reason else ("timeout" if "deadline" in failure_reason or "timeout" in failure_reason else "error")
                                        else:
                                            ver_output, _ = self._parse_verification_output(ver_call2.content)
                                            if ver_output and ver_output.verified:
                                                det_valid = True
                                                verified_flag = True

                            if status == "ok" and not verified_flag:
                                failure_reason = "verification_failed_twice"
                                gen_output = GenerationOutput(
                                    abstained=True,
                                    abstention_reason="Unable to verify full factual and citation support.",
                                    answer_text="Insufficient verified evidence in the research library.",
                                    claims=[],
                                )
                                det_valid = True
                                verified_flag = True

                            if status == "ok" and gen_output is not None:
                                user_response_text = format_verified_user_response(gen_output, final_pieces)
                                if use_cache and self.cache and failure_reason == "none":
                                    self.cache.put(
                                        pq.normalized_query, setting_name, model_id,
                                        {
                                            "generation_output": gen_output.model_dump(),
                                            "verification_output": ver_output.model_dump() if ver_output else None,
                                            "user_response_text": user_response_text,
                                        },
                                    )

            if status == "ok" and user_response_text:
                t_deliv_start = time.perf_counter()
                async for chunk_str in self._stream_verified_text(user_response_text):
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t_start) * 1000.0
                    streamed_chunks.append(chunk_str)
                stream_delivery_ms = (time.perf_counter() - t_deliv_start) * 1000.0

        finally:
            self.scheduler.release_admission(ctx)

        return self._finalize_telemetry(
            ctx, question, start_epoch, t_start, status, failure_reason, http_status_code, gen_output.abstained if gen_output else False,
            verified_flag, user_response_text, streamed_chunks, gen_output, ver_output, final_pieces, setting_name, setting_cfg,
            model_id, cache_hit, eval_question, scenario, scenario_metadata,
            query_proc_ms, retrieval_ms, rerank_ms, evidence_eval_ms, primary_gen_ms, det_verify_ms, primary_sem_verify_ms,
            stream_delivery_ms, provider_first_token_ms, first_token_ms
        )

    def _finalize_telemetry(
        self, ctx, question, start_epoch, t_start, status, failure_reason, http_status_code, abstained, verified_flag,
        user_response_text, streamed_chunks, gen_output, ver_output, final_pieces, setting_name, setting_cfg,
        model_id, cache_hit, eval_question, scenario, scenario_metadata,
        query_proc_ms, retrieval_ms, rerank_ms, evidence_eval_ms, primary_gen_ms, det_verify_ms, primary_sem_verify_ms,
        stream_delivery_ms, provider_first_token_ms, first_token_ms
    ) -> PipelineExecutionResult:
        
        pre_delivery_waiting_ms = ctx.local_queue_wait_ms + ctx.primary_provider_queue_wait_ms
        pre_delivery_retrying_ms = ctx.failed_attempt_work_ms + ctx.retry_backoff_ms + ctx.retry_queue_wait_ms + ctx.correction_work_ms
        pre_delivery_working_ms = query_proc_ms + retrieval_ms + rerank_ms + evidence_eval_ms + primary_gen_ms + det_verify_ms + primary_sem_verify_ms

        end_epoch = time.time()
        wall_total_ms = max(
            (end_epoch - start_epoch) * 1000.0,
            pre_delivery_waiting_ms + pre_delivery_retrying_ms + pre_delivery_working_ms + stream_delivery_ms
        )
        end_epoch = start_epoch + (wall_total_ms / 1000.0)

        if first_token_ms is not None:
            first_token_ms = pre_delivery_waiting_ms + pre_delivery_retrying_ms + pre_delivery_working_ms
            stream_delivery_ms = max(0.0, wall_total_ms - first_token_ms)
        else:
            stream_delivery_ms = 0.0
            overhead = wall_total_ms - (pre_delivery_waiting_ms + pre_delivery_retrying_ms + pre_delivery_working_ms)
            if overhead > 0:
                query_proc_ms += overhead

        waiting_ms = ctx.local_queue_wait_ms + ctx.primary_provider_queue_wait_ms
        working_ms = query_proc_ms + retrieval_ms + rerank_ms + evidence_eval_ms + primary_gen_ms + det_verify_ms + primary_sem_verify_ms + stream_delivery_ms
        retrying_ms = ctx.failed_attempt_work_ms + ctx.retry_backoff_ms + ctx.retry_queue_wait_ms + ctx.correction_work_ms
        total_ms = waiting_ms + working_ms + retrying_ms

        total_tokens_in = sum(int(c["tokens_in"]) for c in ctx.call_traces)
        total_tokens_out = sum(int(c["tokens_out"]) for c in ctx.call_traces)
        modes = {c["accounting_mode"] for c in ctx.call_traces if c["accounting_mode"] != "none"}
        if not ctx.call_traces or not modes:
            token_mode = "none"
        elif modes == {"exact_reported"}:
            token_mode = "exact_reported"
        elif modes == {"partial_estimated"}:
            token_mode = "partial_estimated"
        elif "unknown" in modes:
            token_mode = "unknown"
        else:
            token_mode = "mixed"

        est_tokens_total = sum(int(c["tokens_in"]) + int(c["tokens_out"]) for c in ctx.call_traces if c["accounting_mode"] in ("partial_estimated", "unknown"))
        ref_cost_usd = total_tokens_in * self.price_in_per_tok + total_tokens_out * self.price_out_per_tok

        answer_grade = grade_pipeline_response(eval_question, status, gen_output, det_verify_ms > 0) if eval_question else ("" if status != "ok" else "wrong")

        telemetry_row = {
            "request_id": ctx.request_id,
            "scenario": scenario,
            "source": self.source,
            "start_time": start_epoch,
            "first_token_ms": first_token_ms if first_token_ms is not None else "",
            "total_ms": total_ms,
            "status": status,
            "retries": ctx.transport_retries,
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
            "local_queue_wait_ms": ctx.local_queue_wait_ms,
            "primary_provider_queue_wait_ms": ctx.primary_provider_queue_wait_ms,
            "query_proc_ms": query_proc_ms,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "evidence_eval_ms": evidence_eval_ms,
            "primary_gen_ms": primary_gen_ms,
            "det_verify_ms": det_verify_ms,
            "primary_sem_verify_ms": primary_sem_verify_ms,
            "stream_delivery_ms": stream_delivery_ms,
            "failed_attempt_work_ms": ctx.failed_attempt_work_ms,
            "retry_backoff_ms": ctx.retry_backoff_ms,
            "retry_queue_wait_ms": ctx.retry_queue_wait_ms,
            "correction_work_ms": ctx.correction_work_ms,
            "local_queue_depth_peak": self.scheduler.peak_local,
            "provider_queue_depth_peak": self.scheduler.peak_provider,
            "in_flight_peak": self.scheduler.peak_in_flight,
            "correction_attempts": ctx.correction_attempts,
            "total_provider_calls": len(ctx.call_traces),
            "http_status_code": http_status_code,
            "failure_reason": failure_reason,
            "client_abandoned": ctx.client_abandoned,
            "token_accounting_mode": token_mode,
            "estimated_tokens_total": est_tokens_total,
            "actual_cost_usd": 0.0,
            "reference_cost_usd": ref_cost_usd,
            "call_trace_json": json.dumps(ctx.call_traces, ensure_ascii=False),
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
            request_id=ctx.request_id,
            question=question,
            status=status,
            abstained=abstained,
            verified=verified_flag,
            user_response_text=user_response_text,
            streamed_chunks=streamed_chunks,
            generation_output=gen_output,
            verification_output=ver_output,
            retrieved_pieces=final_pieces,
            telemetry_row=telemetry_row,
        )
