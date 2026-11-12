"""
Phase 4 (Part 2): Retrieval & Re-ranking Engine.
Implements:
1. BM25 lexical retrieval (via rank_bm25) with technical identifier tokenization.
2. Dense vector retrieval (offline-cached L2-normalized embeddings; supports both
   lightweight SVD/n-gram dense embeddings for 8GB RAM safety and optional
   sentence-transformers bi-encoder).
3. Hybrid retrieval via Reciprocal Rank Fusion (RRF).
4. Cross-encoder reranking as a strictly separate operation recording its own latency.
5. Offline comparison of RRF top-4 vs Cross-Encoder top-4 on the identical 12-candidate pool.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.chunking import ChunkRecord, load_chunks_jsonl

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "what", "which", "who",
    "whom", "this", "that", "these", "those", "how", "why", "where", "when",
    "can", "could", "would", "should", "may", "might", "will", "shall", "into",
    "about", "than", "then", "so", "such", "no", "not", "only", "own", "same",
}

TECH_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*", re.IGNORECASE)

_ST_MODEL_CACHE: Dict[str, Any] = {}
_CE_MODEL_CACHE: Dict[str, Any] = {}


def tokenize_technical(text: str, remove_stopwords: bool = True) -> List[str]:
    if not text:
        return []
    raw_tokens = TECH_TOKEN_RE.findall(text.lower())
    expanded: List[str] = []
    for tok in raw_tokens:
        if remove_stopwords and tok in STOPWORDS:
            continue
        expanded.append(tok)
        if "-" in tok or "_" in tok:
            for part in re.split(r"[\-_]", tok):
                if len(part) >= 2 and (not remove_stopwords or part not in STOPWORDS):
                    expanded.append(part)
    return expanded


@dataclass
class RetrievedPiece:
    chunk: ChunkRecord
    rank: int
    score: float
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


@dataclass
class RetrievalResult:
    setting_name: str
    retrieval_method: str
    chunk_size_tokens: int
    candidates_fetched: int
    pieces_passed: int
    rerank_enabled: bool
    reranker_backend_used: str
    retrieval_ms: float
    rerank_ms: float
    candidates_before_rerank: List[RetrievedPiece]
    final_pieces: List[RetrievedPiece]
    max_lexical_overlap: float
    top_score: float


class BM25Retriever:
    """Lexical BM25 retriever preserving technical notation, section numbers, and IDs."""

    def __init__(self, chunks: List[ChunkRecord]) -> None:
        self.chunks = chunks
        self.corpus_tokens: List[List[str]] = []
        for ch in chunks:
            header_prefix = (
                f"{ch.paper_id} {ch.arxiv_version} {ch.title} "
                f"page {ch.primary_page} {ch.section_path} "
            )
            if ch.table_caption_ids:
                header_prefix += " ".join(f"table {tid}" for tid in ch.table_caption_ids) + " "
            if ch.equation_numbers:
                header_prefix += " ".join(f"equation {eid}" for eid in ch.equation_numbers) + " "
            self.corpus_tokens.append(tokenize_technical(header_prefix + ch.text))
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(
        self, query: str, top_k: int, paper_boost_ids: Optional[List[str]] = None
    ) -> List[RetrievedPiece]:
        q_tokens = tokenize_technical(query)
        if not q_tokens:
            return []
        raw_scores = np.array(self.bm25.get_scores(q_tokens), dtype=np.float64)

        if paper_boost_ids:
            boost_set = set(paper_boost_ids)
            for idx, ch in enumerate(self.chunks):
                if ch.paper_id in boost_set:
                    raw_scores[idx] *= 1.35

        top_indices = np.argsort(-raw_scores)[:top_k]
        results: List[RetrievedPiece] = []
        for rank_idx, doc_idx in enumerate(top_indices, start=1):
            sc = float(raw_scores[int(doc_idx)])
            results.append(
                RetrievedPiece(
                    chunk=self.chunks[int(doc_idx)],
                    rank=rank_idx,
                    score=sc,
                    bm25_rank=rank_idx,
                    bm25_score=sc,
                )
            )
        return results


class DenseRetriever:
    """
    Dense vector retriever with offline .npy caching.
    Supports:
    - "lightweight": Deterministic SVD / hashed word+char n-gram L2-normalized dense
      embeddings (128-dim, <15MB RAM, zero external weight downloads).
    - "sentence_transformers": HuggingFace SentenceTransformer bi-encoder loaded as a
      singleton when installed and requested.
    """

    def __init__(
        self,
        chunks: List[ChunkRecord],
        cache_dir: Optional[Path] = None,
        backend: str = "lightweight",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 128,
    ) -> None:
        self.chunks = chunks
        self.backend = backend
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.cache_dir = cache_dir

        self._vocab: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._proj_matrix: Optional[np.ndarray] = None
        self.embeddings: np.ndarray = self._build_or_load_embeddings()

    def _corpus_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(f"{self.backend}:{self.model_name}:{self.embedding_dim}:".encode("utf-8"))
        for ch in self.chunks:
            h.update(f"{ch.chunk_id}:{ch.sha256}:{ch.token_count}|".encode("utf-8"))
        return h.hexdigest()[:16]

    def _extract_features(self, text: str) -> List[str]:
        words = tokenize_technical(text, remove_stopwords=True)
        features = list(words)
        for i in range(len(words) - 1):
            features.append(f"{words[i]}_{words[i + 1]}")
        joined_alpha = re.sub(r"[^a-z0-9]", "", " ".join(words))
        for i in range(0, max(0, len(joined_alpha) - 3), 2):
            features.append(f"#{joined_alpha[i:i + 4]}")
        return features

    def _build_lightweight_embeddings(self) -> np.ndarray:
        doc_feats: List[List[str]] = []
        df_counts: Dict[str, int] = {}
        for ch in self.chunks:
            rich_text = f"{ch.title} {ch.section_path} {ch.text}"
            feats = self._extract_features(rich_text)
            doc_feats.append(feats)
            for f in set(feats):
                df_counts[f] = df_counts.get(f, 0) + 1

        n_docs = max(1, len(self.chunks))
        sorted_feats = sorted(
            df_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:4096]
        self._vocab = {f: idx for idx, (f, _) in enumerate(sorted_feats)}
        v_size = max(1, len(self._vocab))

        idf = np.zeros(v_size, dtype=np.float32)
        for f, idx in self._vocab.items():
            idf[idx] = float(np.log((1.0 + n_docs) / (1.0 + df_counts[f])) + 1.0)
        self._idf = idf

        tfidf = np.zeros((n_docs, v_size), dtype=np.float32)
        for d_idx, feats in enumerate(doc_feats):
            for f in feats:
                v_idx = self._vocab.get(f)
                if v_idx is not None:
                    tfidf[d_idx, v_idx] += 1.0
            row_sum = float(tfidf[d_idx].sum())
            if row_sum > 0:
                tfidf[d_idx] = (tfidf[d_idx] / row_sum) * idf

        k_dim = min(self.embedding_dim, n_docs - 1, v_size - 1)
        if k_dim >= 8:
            _, _, vt = np.linalg.svd(tfidf, full_matrices=False)
            self._proj_matrix = vt[:k_dim].T.astype(np.float32)
        else:
            rng = np.random.default_rng(42)
            self._proj_matrix = rng.standard_normal(
                (v_size, self.embedding_dim), dtype=np.float32
            ) / np.sqrt(self.embedding_dim)

        dense = tfidf @ self._proj_matrix
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense / norms).astype(np.float32)

    def _encode_query_lightweight(self, query: str) -> np.ndarray:
        if self._idf is None or self._proj_matrix is None or not self._vocab:
            return np.zeros(self.embeddings.shape[1], dtype=np.float32)
        feats = self._extract_features(query)
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for f in feats:
            idx = self._vocab.get(f)
            if idx is not None:
                vec[idx] += 1.0
        s = float(vec.sum())
        if s > 0:
            vec = (vec / s) * self._idf
        q_dense = vec @ self._proj_matrix
        norm = float(np.linalg.norm(q_dense))
        if norm > 0:
            q_dense = q_dense / norm
        return q_dense.astype(np.float32)

    def _build_or_load_embeddings(self) -> np.ndarray:
        if self.backend == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer

                fp = self._corpus_fingerprint()
                npy_path = (
                    self.cache_dir / f"dense_st_{fp}.npy"
                    if self.cache_dir is not None
                    else None
                )
                if npy_path is not None and npy_path.exists():
                    return np.load(npy_path)

                if self.model_name not in _ST_MODEL_CACHE:
                    _ST_MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
                model = _ST_MODEL_CACHE[self.model_name]
                texts = [f"{ch.title} | {ch.section_path}\n{ch.text}" for ch in self.chunks]
                emb = model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )
                emb_arr = np.asarray(emb, dtype=np.float32)
                if npy_path is not None:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    np.save(npy_path, emb_arr)
                return emb_arr
            except Exception:
                self.backend = "lightweight"

        return self._build_lightweight_embeddings()

    def search(
        self, query: str, top_k: int, paper_boost_ids: Optional[List[str]] = None
    ) -> List[RetrievedPiece]:
        if self.backend == "sentence_transformers" and self.model_name in _ST_MODEL_CACHE:
            model = _ST_MODEL_CACHE[self.model_name]
            q_vec = np.asarray(
                model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0],
                dtype=np.float32,
            )
        else:
            q_vec = self._encode_query_lightweight(query)

        scores = (self.embeddings @ q_vec).astype(np.float64)
        if paper_boost_ids:
            boost_set = set(paper_boost_ids)
            for idx, ch in enumerate(self.chunks):
                if ch.paper_id in boost_set:
                    scores[idx] += 0.12

        top_indices = np.argsort(-scores)[:top_k]
        results: List[RetrievedPiece] = []
        for rank_idx, doc_idx in enumerate(top_indices, start=1):
            sc = float(scores[int(doc_idx)])
            results.append(
                RetrievedPiece(
                    chunk=self.chunks[int(doc_idx)],
                    rank=rank_idx,
                    score=sc,
                    dense_rank=rank_idx,
                    dense_score=sc,
                )
            )
        return results


class CrossEncoderReranker:
    """
    Separate second-stage Cross-Encoder reranker that jointly scores (query, candidate_chunk)
    pairs and records its own execution time. 
    """

    def __init__(
        self,
        backend: str = "lightweight",
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.backend = backend
        self.model_name = model_name

    def _score_pair_interaction(self, query: str, piece: RetrievedPiece) -> float:
        q_tokens = tokenize_technical(query, remove_stopwords=True)
        if not q_tokens:
            return 0.0
        ch = piece.chunk
        doc_text_lower = f"{ch.title} {ch.section_path} {ch.text}".lower()
        doc_tokens = tokenize_technical(doc_text_lower, remove_stopwords=True)
        if not doc_tokens:
            return 0.0

        doc_token_set = set(doc_tokens)
        q_unique = list(dict.fromkeys(q_tokens))
        matched_q = [t for t in q_unique if t in doc_token_set]
        coverage = len(matched_q) / max(1, len(q_unique))

        q_bigrams = {
            f"{q_tokens[i]} {q_tokens[i + 1]}" for i in range(len(q_tokens) - 1)
        }
        bigram_hits = sum(1 for bg in q_bigrams if bg in doc_text_lower)
        bigram_score = bigram_hits / max(1, len(q_bigrams)) if q_bigrams else 0.0

        q_set = set(q_unique)
        win_size = 35
        max_win_coverage = 0.0
        if len(doc_tokens) <= win_size:
            max_win_coverage = coverage
        else:
            step = 10
            for start in range(0, len(doc_tokens) - win_size + 1, step):
                win_hits = len(q_set.intersection(doc_tokens[start : start + win_size]))
                c = win_hits / max(1, len(q_set))
                if c > max_win_coverage:
                    max_win_coverage = c

        locator_bonus = 0.0
        for m_tab in re.finditer(r"\btable\s+(\d+|[a-z]\.\d+)\b", query.lower()):
            tid = m_tab.group(1).upper()
            if tid in [x.upper() for x in ch.table_caption_ids]:
                locator_bonus += 0.45
            elif tid in [x.upper() for x in ch.table_references]:
                locator_bonus += 0.20

        for m_eq in re.finditer(
            r"\b(?:equation|eq\.?)\s*\(?(\d+|[a-z]\.\d+)\)?", query.lower()
        ):
            eid = m_eq.group(1).upper()
            if eid in [x.upper() for x in ch.equation_numbers]:
                locator_bonus += 0.45

        sec_tokens = set(tokenize_technical(ch.section_path, remove_stopwords=True))
        sec_overlap = len(q_set.intersection(sec_tokens)) / max(1, len(q_set))

        base_rrf = float(piece.rrf_score or piece.score or 0.0)
        return float(
            0.40 * coverage
            + 0.25 * max_win_coverage
            + 0.15 * bigram_score
            + 0.15 * sec_overlap
            + locator_bonus
            + 2.0 * base_rrf
        )

    def rerank(
        self, query: str, candidates: List[RetrievedPiece], top_k: int
    ) -> Tuple[List[RetrievedPiece], float, str]:
        """
        Returns: (reranked_pieces, elapsed_ms, backend_used)
        """
        t0 = time.perf_counter()
        if not candidates:
            return [], 0.0, "None"

        scored: List[RetrievedPiece] = []
        actual_backend_used = "lightweight"

        if self.backend == "sentence_transformers":
            try:
                from sentence_transformers import CrossEncoder

                if self.model_name not in _CE_MODEL_CACHE:
                    _CE_MODEL_CACHE[self.model_name] = CrossEncoder(self.model_name)
                ce = _CE_MODEL_CACHE[self.model_name]
                pairs = [
                    (query, f"{c.chunk.title} ({c.chunk.section_path}): {c.chunk.text}")
                    for c in candidates
                ]
                ce_scores = ce.predict(pairs, show_progress_bar=False)
                for cand, sc in zip(candidates, ce_scores):
                    cp = RetrievedPiece(**asdict(cand))
                    cp.chunk = cand.chunk
                    cp.rerank_score = float(sc)
                    cp.score = float(sc)
                    scored.append(cp)
                actual_backend_used = "sentence_transformers"
            except ImportError:
                actual_backend_used = "lightweight"
            except Exception:
                actual_backend_used = "lightweight"

        if actual_backend_used == "lightweight":
            for cand in candidates:
                sc = self._score_pair_interaction(query, cand)
                cp = RetrievedPiece(
                    chunk=cand.chunk,
                    rank=cand.rank,
                    score=sc,
                    bm25_rank=cand.bm25_rank,
                    bm25_score=cand.bm25_score,
                    dense_rank=cand.dense_rank,
                    dense_score=cand.dense_score,
                    rrf_score=cand.rrf_score,
                    rerank_score=sc,
                )
                scored.append(cp)

        scored.sort(key=lambda p: (-float(p.rerank_score or 0.0), p.chunk.chunk_id))
        selected = scored[:top_k]
        for idx, item in enumerate(selected, start=1):
            item.rank = idx

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return selected, elapsed_ms, actual_backend_used


class HybridRRFRetriever:
    """
    Combines BM25 lexical rankings and Dense vector rankings using Reciprocal Rank Fusion:
      RRF(d) = 1 / (k_rrf + rank_bm25(d)) + 1 / (k_rrf + rank_dense(d))
    """

    def __init__(
        self,
        chunks: List[ChunkRecord],
        cache_dir: Optional[Path] = None,
        rrf_k: int = 60,
        dense_backend: str = "lightweight",
        reranker_backend: str = "lightweight",
    ) -> None:
        self.chunks = chunks
        self.chunks_by_id: Dict[str, ChunkRecord] = {c.chunk_id: c for c in chunks}
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(
            chunks=chunks, cache_dir=cache_dir, backend=dense_backend
        )
        self.reranker = CrossEncoderReranker(backend=reranker_backend)

    def retrieve_candidates(
        self,
        query: str,
        candidates_k: int,
        method: str = "hybrid_rrf",
        paper_boost_ids: Optional[List[str]] = None,
    ) -> List[RetrievedPiece]:
        if method == "bm25":
            return self.bm25.search(
                query, top_k=candidates_k, paper_boost_ids=paper_boost_ids
            )
        if method == "dense":
            return self.dense.search(
                query, top_k=candidates_k, paper_boost_ids=paper_boost_ids
            )

        pool_depth = max(candidates_k * 3, 24)
        bm25_list = self.bm25.search(
            query, top_k=pool_depth, paper_boost_ids=paper_boost_ids
        )
        dense_list = self.dense.search(
            query, top_k=pool_depth, paper_boost_ids=paper_boost_ids
        )

        bm25_map: Dict[str, Tuple[int, float]] = {
            p.chunk.chunk_id: (p.rank, float(p.bm25_score or 0.0)) for p in bm25_list
        }
        dense_map: Dict[str, Tuple[int, float]] = {
            p.chunk.chunk_id: (p.rank, float(p.dense_score or 0.0)) for p in dense_list
        }

        all_ids = set(bm25_map.keys()) | set(dense_map.keys())
        fused: List[RetrievedPiece] = []

        for cid in all_ids:
            b_rank, b_score = bm25_map.get(cid, (pool_depth + 50, 0.0))
            d_rank, d_score = dense_map.get(cid, (pool_depth + 50, 0.0))
            rrf_val = (1.0 / (self.rrf_k + b_rank)) + (1.0 / (self.rrf_k + d_rank))
            fused.append(
                RetrievedPiece(
                    chunk=self.chunks_by_id[cid],
                    rank=0,
                    score=rrf_val,
                    bm25_rank=b_rank if cid in bm25_map else None,
                    bm25_score=b_score if cid in bm25_map else None,
                    dense_rank=d_rank if cid in dense_map else None,
                    dense_score=d_score if cid in dense_map else None,
                    rrf_score=rrf_val,
                )
            )

        fused.sort(
            key=lambda x: (
                -float(x.rrf_score or 0.0),
                -(x.bm25_score or 0.0),
                x.chunk.chunk_id,
            )
        )
        top_candidates = fused[:candidates_k]
        for idx, item in enumerate(top_candidates, start=1):
            item.rank = idx
        return top_candidates

    def execute_setting(
        self,
        query: str,
        setting_name: str,
        setting_cfg: Dict[str, Any],
        paper_boost_ids: Optional[List[str]] = None,
    ) -> RetrievalResult:
        method = str(setting_cfg.get("retrieval_method", "hybrid_rrf"))
        c_size = int(setting_cfg["chunk_size_tokens"])
        c_fetched = int(setting_cfg["candidates_fetched"])
        p_passed = int(setting_cfg["pieces_passed"])
        rerank_on = bool(setting_cfg["rerank_enabled"])

        t0 = time.perf_counter()
        candidates = self.retrieve_candidates(
            query=query,
            candidates_k=c_fetched,
            method=method,
            paper_boost_ids=paper_boost_ids,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        if rerank_on:
            final_pieces, rerank_ms, actual_backend = self.reranker.rerank(
                query=query, candidates=candidates, top_k=p_passed
            )
        else:
            final_pieces = candidates[:p_passed]
            rerank_ms = 0.0
            actual_backend = "None"

        q_toks = set(tokenize_technical(query, remove_stopwords=True))
        max_overlap = 0.0
        if q_toks and final_pieces:
            for fp in final_pieces:
                c_toks = set(
                    tokenize_technical(
                        f"{fp.chunk.title} {fp.chunk.section_path} {fp.chunk.text}",
                        remove_stopwords=True,
                    )
                )
                ov = len(q_toks.intersection(c_toks)) / len(q_toks)
                if ov > max_overlap:
                    max_overlap = ov

        top_sc = float(final_pieces[0].score) if final_pieces else 0.0

        return RetrievalResult(
            setting_name=setting_name,
            retrieval_method=method,
            chunk_size_tokens=c_size,
            candidates_fetched=c_fetched,
            pieces_passed=p_passed,
            rerank_enabled=rerank_on,
            reranker_backend_used=actual_backend,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            candidates_before_rerank=candidates,
            final_pieces=final_pieces,
            max_lexical_overlap=max_overlap,
            top_score=top_sc,
        )

    def compare_rrf_vs_reranker_on_same_pool(
        self,
        query: str,
        pool_size: int = 12,
        top_k: int = 4,
        paper_boost_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Isolates the Cross-Encoder reranking effect from candidate pool depth by comparing:
        - RRF top-k sliced directly from the 12-candidate pool
        - Cross-Encoder top-k reranked from the exact same 12-candidate pool.
        """
        candidates_12 = self.retrieve_candidates(
            query=query,
            candidates_k=pool_size,
            method="hybrid_rrf",
            paper_boost_ids=paper_boost_ids,
        )
        rrf_top_k = candidates_12[:top_k]
        reranked_top_k, rerank_ms, backend_used = self.reranker.rerank(
            query=query, candidates=candidates_12, top_k=top_k
        )
        rrf_ids = [p.chunk.chunk_id for p in rrf_top_k]
        reranked_ids = [p.chunk.chunk_id for p in reranked_top_k]
        return {
            "pool_size": len(candidates_12),
            "top_k": top_k,
            "rerank_ms": round(rerank_ms, 3),
            "backend_used": backend_used,
            "rrf_top_k_ids": rrf_ids,
            "reranked_top_k_ids": reranked_ids,
            "order_changed": rrf_ids != reranked_ids,
            "set_changed": set(rrf_ids) != set(reranked_ids),
            "rrf_top_pieces": rrf_top_k,
            "reranked_top_pieces": reranked_top_k,
        }


class RetrievalEngineRegistry:
    """Manages initialized HybridRRFRetriever instances keyed by chunk size (250 and 500)."""

    def __init__(
        self,
        indices_dir: Path,
        config: Dict[str, Any],
    ) -> None:
        self.indices_dir = indices_dir
        self.config = config
        self.ret_cfg = config.get("retrieval", {})
        self.engines_by_size: Dict[int, HybridRRFRetriever] = {}

    def get_engine(self, chunk_size_tokens: int) -> HybridRRFRetriever:
        if chunk_size_tokens not in self.engines_by_size:
            chunks_file = self.indices_dir / f"chunks_{chunk_size_tokens}.jsonl"
            chunks = load_chunks_jsonl(chunks_file)
            self.engines_by_size[chunk_size_tokens] = HybridRRFRetriever(
                chunks=chunks,
                cache_dir=self.indices_dir,
                rrf_k=int(self.ret_cfg.get("rrf_k", 60)),
                dense_backend=str(self.ret_cfg.get("dense_backend", "lightweight")),
                reranker_backend=str(self.ret_cfg.get("reranker_backend", "lightweight")),
            )
        return self.engines_by_size[chunk_size_tokens]

    def retrieve_for_setting(
        self,
        query: str,
        setting_name: str,
        paper_boost_ids: Optional[List[str]] = None,
    ) -> RetrievalResult:
        settings_map = self.ret_cfg.get("settings", {})
        if setting_name not in settings_map:
            raise ValueError(f"Unknown retrieval_setting: {setting_name}")
        scfg = settings_map[setting_name]
        c_size = int(scfg["chunk_size_tokens"])
        engine = self.get_engine(c_size)
        return engine.execute_setting(
            query=query,
            setting_name=setting_name,
            setting_cfg=scfg,
            paper_boost_ids=paper_boost_ids,
        )
