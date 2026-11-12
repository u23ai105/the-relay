"""
Step 1 & Step 3 of the RAG Pipeline:
- Deterministic technical query normalization (preserving math symbols, paper IDs, section/table locators).
- Paper mention detection & routing hints.
- Corpus Concept & Alias Map for conditional deterministic query expansion.
- Conservative deterministic evidence sufficiency evaluation (preventing false refusals).
- Metadata-scoped verified exact-match answer cache.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval import RetrievalResult, tokenize_technical

# Maps informal or paraphrased technical concepts to exact terminology from the 3 papers
CORPUS_CONCEPT_ALIAS_MAP: List[Tuple[List[str], str, Optional[str]]] = [
    (
        ["audio codec", "neural codec", "speech tokenizer", "acoustic tokens", "rvq", "residual vector quantization"],
        "Mimi neural audio codec split residual vector quantization RVQ 12.5Hz 1.1kbps Audio Tokenization",
        "2410.00037",
    ),
    (
        ["full duplex", "full-duplex", "spoken dialogue", "inner monologue", "overlapping speech", "voice model", "moshi", "helium"],
        "Moshi Helium 7B text language model RQ-Transformer Inner Monologue streaming real-time dialogue",
        "2410.00037",
    ),
    (
        ["watermarking", "voice consistency", "toxicity", "regurgitation", "safety"],
        "Moshi 6 Safety toxicity regurgitation system voice consistency watermarking",
        "2410.00037",
    ),
    (
        ["tokenizer free", "tokenizer-free", "byte level", "byte-level", "dynamic chunking", "h-net", "hnet", "boundary probability", "dechunking", "ratio loss"],
        "H-Net Dynamic Chunking routing module boundary probability cosine similarity smoothing module dechunking ratio loss",
        "2507.07955",
    ),
    (
        ["dna", "chinese", "code dataset", "isotropic", "bpe distillation", "byte distillation"],
        "H-Net Alternate Language Datasets Chinese Code DNA Ablation Studies",
        "2507.07955",
    ),
    (
        ["scaled dot product", "scaled dot-product", "multi-head attention", "multi head attention", "positional encoding", "label smoothing", "warmup_steps", "transformer architecture"],
        "Attention Is All You Need Scaled Dot-Product Attention Multi-Head Attention d_model 512 d_ff 2048 h=8 sinusoidal positional encoding label smoothing",
        "1706.03762",
    ),
    (
        ["wmt 2014", "english to german", "english-to-german", "english to french", "english-to-french", "bleu score", "constituency parsing"],
        "Attention Is All You Need Table 2 Table 3 WMT 2014 English-to-German 28.4 English-to-French 41.8 BLEU English Constituency Parsing",
        "1706.03762",
    ),
]

PAPER_EXPLICIT_PATTERNS: Dict[str, re.Pattern] = {
    "1706.03762": re.compile(
        r"\b(1706\.03762(?:v7)?|attention is all you need|vaswani)\b", re.IGNORECASE
    ),
    "2410.00037": re.compile(
        r"\b(2410\.00037(?:v1)?|moshi|helium|mimi\s+codec|inner\s+monologue|rq-transformer)\b",
        re.IGNORECASE,
    ),
    "2507.07955": re.compile(
        r"\b(2507\.07955(?:v2)?|dynamic\s+chunking|h-net|hnet|dechunking|ratio\s+loss)\b",
        re.IGNORECASE,
    ),
}

CORPUS_DOMAIN_VOCAB = {
    "attention", "transformer", "transformers", "encoder", "decoder", "multi", "head",
    "dot", "product", "softmax", "positional", "encoding", "embedding", "embeddings",
    "bleu", "wmt", "german", "french", "parsing", "layer", "layers", "d_model", "dff",
    "dropout", "optimizer", "adam", "warmup", "smoothing", " perplexity", "ppl",
    "moshi", "helium", "mimi", "audio", "speech", "voice", "dialogue", "streaming",
    "codec", "quantization", "rvq", "acoustic", "semantic", "monologue", "duplex",
    "latency", "asr", "tts", "toxicity", "watermarking", "regurgitation",
    "dynamic", "chunking", "dechunking", "h-net", "hnet", "hierarchical", "sequence",
    "byte", "bytes", "tokenizer", "tokenization", "boundary", "routing", "ratio",
    "smoothing", "isotropic", "mamba", "ssm", "autoregressive", "distillation",
    "dna", "chinese", "code", "flops", " equation", "table", "figure", "section",
    "paper", "arxiv", "1706.03762", "2410.00037", "2507.07955", "model", "training",
}


@dataclass
class ProcessedQuery:
    original_query: str
    normalized_query: str
    detected_paper_ids: List[str]
    query_tokens: List[str]
    has_domain_signal: bool
    expanded_query: str
    expansion_terms_added: List[str]


def normalize_technical_query(query: str) -> str:
    """
    Normalizes whitespace and casing while strictly preserving technical identifiers,
    numbers, underscores, hyphens, dots, and mathematical symbols.
    """
    cleaned = re.sub(r"\s+", " ", query.strip())
    return cleaned


def process_query_deterministically(query: str) -> ProcessedQuery:
    norm = normalize_technical_query(query)
    q_lower = norm.lower()

    detected_papers: List[str] = []
    for pid, pat in PAPER_EXPLICIT_PATTERNS.items():
        if pat.search(q_lower):
            detected_papers.append(pid)

    q_tokens = tokenize_technical(norm, remove_stopwords=True)
    expansion_terms: List[str] = []
    inferred_papers: List[str] = list(detected_papers)

    for triggers, alias_expansion, target_pid in CORPUS_CONCEPT_ALIAS_MAP:
        if any(trig in q_lower for trig in triggers):
            expansion_terms.append(alias_expansion)
            if target_pid and target_pid not in inferred_papers:
                inferred_papers.append(target_pid)

    has_domain = bool(
        inferred_papers
        or any(t in CORPUS_DOMAIN_VOCAB for t in q_tokens)
        or bool(expansion_terms)
    )

    if expansion_terms:
        expanded_q = f"{norm} {' '.join(expansion_terms)}"
    else:
        expanded_q = norm

    return ProcessedQuery(
        original_query=query,
        normalized_query=norm,
        detected_paper_ids=inferred_papers,
        query_tokens=q_tokens,
        has_domain_signal=has_domain,
        expanded_query=expanded_q,
        expansion_terms_added=expansion_terms,
    )


@dataclass
class EvidenceEvaluationResult:
    is_sufficient_for_generation: bool
    needs_query_expansion: bool
    deterministic_refusal: bool
    refusal_reason: str
    max_lexical_overlap: float
    top_retrieval_score: float


def evaluate_evidence_sufficiency(
    processed_query: ProcessedQuery,
    retrieval_result: RetrievalResult,
    already_expanded: bool,
    min_lexical_overlap: float = 0.12,
    expansion_threshold: float = 0.32,
) -> EvidenceEvaluationResult:
    """
    Conservative evidence evaluation:
    1. If no chunks were returned or the query has zero domain signal and near-zero
       lexical overlap even after expansion -> deterministic refusal (0 LLM calls).
    2. If initial lexical overlap is weak (< expansion_threshold) and query expansion
       has not been attempted yet -> trigger conditional deterministic query expansion.
    3. Otherwise -> proceed to LLM generation (since keyword overlap alone cannot prove
       a subtle unanswerable question is answerable, nor should borderline phrasing be
       falsely refused without LLM evaluation).
    """
    if not retrieval_result.final_pieces:
        if not already_expanded and processed_query.expansion_terms_added:
            return EvidenceEvaluationResult(
                is_sufficient_for_generation=False,
                needs_query_expansion=True,
                deterministic_refusal=False,
                refusal_reason="",
                max_lexical_overlap=0.0,
                top_retrieval_score=0.0,
            )
        return EvidenceEvaluationResult(
            is_sufficient_for_generation=False,
            needs_query_expansion=False,
            deterministic_refusal=True,
            refusal_reason="No relevant passages were retrieved from the three-paper research library.",
            max_lexical_overlap=0.0,
            top_retrieval_score=0.0,
        )

    overlap = retrieval_result.max_lexical_overlap
    top_score = retrieval_result.top_score

    # Trigger conditional query expansion if initial evidence is weak
    if not already_expanded and (
        overlap < expansion_threshold or len(processed_query.expansion_terms_added) > 0 and overlap < 0.45
    ):
        if processed_query.expanded_query != processed_query.normalized_query:
            return EvidenceEvaluationResult(
                is_sufficient_for_generation=False,
                needs_query_expansion=True,
                deterministic_refusal=False,
                refusal_reason="",
                max_lexical_overlap=overlap,
                top_retrieval_score=top_score,
            )

    # Conservative deterministic refusal ONLY for clearly out-of-domain queries
    if not processed_query.has_domain_signal and overlap < min_lexical_overlap:
        return EvidenceEvaluationResult(
            is_sufficient_for_generation=False,
            needs_query_expansion=False,
            deterministic_refusal=True,
            refusal_reason=(
                "The query is outside the scope of the three provided research papers "
                "(Attention Is All You Need 1706.03762v7, Moshi 2410.00037v1, and "
                "Dynamic Chunking 2507.07955v2)."
            ),
            max_lexical_overlap=overlap,
            top_retrieval_score=top_score,
        )

    return EvidenceEvaluationResult(
        is_sufficient_for_generation=True,
        needs_query_expansion=False,
        deterministic_refusal=False,
        refusal_reason="",
        max_lexical_overlap=overlap,
        top_retrieval_score=top_score,
    )


class VerifiedAnswerCache:
    """
    Exact-match cache for verified responses scoped by:
      (normalized_technical_query, corpus_version_hash, retrieval_setting, model_id)
    """

    def __init__(self, cache_file: Optional[Path] = None, corpus_version_hash: str = "v1") -> None:
        self.cache_file = cache_file
        self.corpus_version_hash = corpus_version_hash
        self._store: Dict[str, Dict[str, Any]] = {}
        if self.cache_file is not None and self.cache_file.exists():
            try:
                self._store = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                self._store = {}

    def _make_key(
        self, query: str, retrieval_setting: str, model_id: str
    ) -> str:
        norm = normalize_technical_query(query).lower()
        raw = f"{norm}|{self.corpus_version_hash}|{retrieval_setting}|{model_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self, query: str, retrieval_setting: str, model_id: str
    ) -> Optional[Dict[str, Any]]:
        k = self._make_key(query, retrieval_setting, model_id)
        return self._store.get(k)

    def put(
        self,
        query: str,
        retrieval_setting: str,
        model_id: str,
        verified_payload: Dict[str, Any],
    ) -> None:
        k = self._make_key(query, retrieval_setting, model_id)
        self._store[k] = verified_payload
        if self.cache_file is not None:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(
                json.dumps(self._store, indent=2, ensure_ascii=False), encoding="utf-8"
            )