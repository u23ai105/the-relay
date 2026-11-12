"""
Phase 9: Table 3 Retrieval Setting Evaluation.
Executes the 18 authored questions across all 4 configurations WITHOUT LLM generation,
measuring strict retrieval recall, chunk distribution, and offline token overhead.
"""

import json
import statistics
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.logger import get_logger
from src.query_and_evidence import process_query_deterministically, evaluate_evidence_sufficiency
from src.retrieval import RetrievalEngineRegistry
from src.evaluation_questions import EVALUATION_QUESTIONS

def calc_stats(lst):
    if not lst:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    s = sorted(lst)
    n = len(s)
    p95_idx = int(0.95 * n)
    if p95_idx >= n: p95_idx = n - 1
    return {
        "mean": round(sum(s) / n, 2),
        "median": round(statistics.median(s), 2),
        "p95": round(s[p95_idx], 2)
    }

def main() -> None:
    logger = get_logger("retrieval_eval")
    config = load_config()
    indices_dir = PROJECT_ROOT / "data" / "indices"
    
    logger.info("Initializing and warming up retrieval engines...")
    registry = RetrievalEngineRegistry(indices_dir, config)
    _ = registry.get_engine(250)
    _ = registry.get_engine(500)
    
    settings_to_eval = [
        "250tok_4to4_rerank_no",
        "500tok_4to4_rerank_no",
        "500tok_12to4_rerank_yes",
        "500tok_12to2_rerank_yes"
    ]

    # Warmup query path for all 4 configurations to ensure models are loaded
    for s in settings_to_eval:
        _ = registry.retrieve_for_setting("warmup query", s)
    logger.info("Engines warmed up successfully.")
    
    results_dir = PROJECT_ROOT / "results" / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / "retrieval_offline_eval_report.json"
    
    overall_report = {}
    
    for setting in settings_to_eval:
        logger.info(f"Evaluating Setting: {setting}")
        
        total_q = 0
        total_tokens = 0
        top_1_hits = 0
        top_k_hits = 0
        unanswerable_refused = 0
        
        ret_ms_list = []
        rerank_ms_list = []
        tot_ms_list = []
        
        answerable_q = len([q for q in EVALUATION_QUESTIONS if not q.expected_abstained])
        unanswerable_q = len([q for q in EVALUATION_QUESTIONS if q.expected_abstained])
        actual_backend = "None"
        
        for q in EVALUATION_QUESTIONS:
            total_q += 1
            pq = process_query_deterministically(q.question)
            
            res = registry.retrieve_for_setting(
                query=pq.normalized_query, 
                setting_name=setting,
                paper_boost_ids=pq.detected_paper_ids or None
            )
            
            ret_ms = res.retrieval_ms
            rerank_ms = res.rerank_ms
            
            if res.reranker_backend_used != "None":
                actual_backend = res.reranker_backend_used
            
            ev_res = evaluate_evidence_sufficiency(
                processed_query=pq,
                retrieval_result=res,
                already_expanded=False,
                min_lexical_overlap=float(config.get("retrieval", {}).get("evidence_min_lexical_overlap", 0.12)),
                expansion_threshold=float(config.get("retrieval", {}).get("evidence_expansion_threshold", 0.32)),
            )
            
            if ev_res.needs_query_expansion:
                res = registry.retrieve_for_setting(
                    query=pq.expanded_query, 
                    setting_name=setting,
                    paper_boost_ids=pq.detected_paper_ids or None
                )
                ret_ms += res.retrieval_ms
                rerank_ms += res.rerank_ms
                
                ev_res = evaluate_evidence_sufficiency(
                    processed_query=pq,
                    retrieval_result=res,
                    already_expanded=True,
                    min_lexical_overlap=float(config.get("retrieval", {}).get("evidence_min_lexical_overlap", 0.12)),
                    expansion_threshold=float(config.get("retrieval", {}).get("evidence_expansion_threshold", 0.32)),
                )
            
            ret_ms_list.append(ret_ms)
            rerank_ms_list.append(rerank_ms)
            tot_ms_list.append(ret_ms + rerank_ms)
            
            prompt_tokens = sum(ch.chunk.token_count for ch in res.final_pieces)
            total_tokens += prompt_tokens
            
            if q.expected_abstained:
                if ev_res.deterministic_refusal:
                    unanswerable_refused += 1
                continue
            
            hit_at_1 = False
            hit_at_k = False
            
            if res.final_pieces:
                for idx, piece in enumerate(res.final_pieces):
                    is_match = (
                        piece.chunk.paper_id == q.expected_paper_id and 
                        piece.chunk.arxiv_version == q.expected_arxiv_version and
                        any(p in piece.chunk.page_numbers for p in q.expected_pages)
                    )
                    if is_match:
                        hit_at_k = True
                        if idx == 0:
                            hit_at_1 = True
                        break
            
            if hit_at_1:
                top_1_hits += 1
            if hit_at_k:
                top_k_hits += 1
                
        avg_tokens = total_tokens / max(1, total_q)
        
        if actual_backend == "lightweight":
            backend_label = "Lightweight Heuristic Reranker"
        elif actual_backend == "sentence_transformers":
            backend_label = "Cross-Encoder (SentenceTransformers)"
        else:
            backend_label = "None"
            
        report = {
            "setting": setting,
            "reranker_backend_used": backend_label,
            "questions_evaluated": total_q,
            "avg_retrieved_evidence_tokens": round(avg_tokens, 1),
            "top_1_passage_recall": f"{top_1_hits}/{answerable_q}",
            "top_k_passage_recall": f"{top_k_hits}/{answerable_q}",
            "diagnostic_deterministic_refusal_rate": f"{unanswerable_refused}/{unanswerable_q}",
            "latency_ms": {
                "retrieval": calc_stats(ret_ms_list),
                "reranking": calc_stats(rerank_ms_list),
                "combined": calc_stats(tot_ms_list)
            }
        }
        overall_report[setting] = report
        logger.info(
            f"Completed {setting} - Combined Median Latency: {report['latency_ms']['combined']['median']} ms, "
            f"Retrieved Tokens: {report['avg_retrieved_evidence_tokens']}"
        )
        
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(overall_report, f, indent=2)
        
    logger.info(f"Offline retrieval evaluation saved to {report_file}")


if __name__ == "__main__":
    main()
