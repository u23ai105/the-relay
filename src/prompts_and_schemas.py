"""
Pydantic schemas and prompt builders for:
1. Evidence-grounded structured answer generation (with granular citations or explicit abstention).
2. Context-pruned semantic claim verification and full-context refusal verification.
"""

import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.retrieval import RetrievedPiece


class CitationLocator(BaseModel):
    chunk_id: str = Field(..., description="Exact chunk_id from the provided evidence passages.")
    paper_id: str = Field(..., description="Exact arXiv paper_id (1706.03762, 2410.00037, or 2507.07955).")
    arxiv_version: str = Field(..., description="Exact arXiv version (v7, v1, or v2).")
    page_number: int = Field(..., description="Page number within the PDF.")
    section_title: str = Field(..., description="Section title from the chunk metadata.")
    locator_detail: str = Field(
        default="",
        description="Specific table, figure, or equation identifier if applicable (e.g., 'Table 1', 'Equation 1').",
    )


class ClaimItem(BaseModel):
    claim_id: str = Field(..., description="Identifier such as 'c1', 'c2'.")
    claim_text: str = Field(..., description="Single factual statement strictly supported by the cited chunks.")
    cited_chunk_ids: List[str] = Field(..., description="List of supporting chunk_ids.")
    citations: List[CitationLocator] = Field(..., description="Detailed source locators for this claim.")


class GenerationOutput(BaseModel):
    abstained: bool = Field(
        ...,
        description="True if the provided passages do NOT contain sufficient evidence to answer the question.",
    )
    abstention_reason: str = Field(
        default="",
        description="If abstained is True, explicitly explain what information is missing from the provided passages.",
    )
    answer_text: str = Field(
        ...,
        description="Complete user-facing response with inline citations [paper_id<version>, p.<page>, <section>].",
    )
    claims: List[ClaimItem] = Field(
        default_factory=list,
        description="List of atomic grounded claims with citations (empty when abstained is True).",
    )


class VerificationOutput(BaseModel):
    verified: bool = Field(
        ...,
        description="True ONLY if all claims are directly entailed by the cited passages (or if the refusal is valid).",
    )
    supported_claim_ids: List[str] = Field(default_factory=list)
    unsupported_claim_ids: List[str] = Field(default_factory=list)
    refusal_valid: Optional[bool] = Field(
        default=None,
        description="When verifying a refusal, True if the passages genuinely lack the answer; False if an answer was overlooked.",
    )
    overlooked_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk IDs that actually contain the answer when a refusal was invalid.",
    )
    feedback: str = Field(
        default="",
        description="Specific corrective instructions if verified is False.",
    )


def format_passages_for_generation(pieces: List[RetrievedPiece]) -> str:
    blocks: List[str] = []
    for p in pieces:
        ch = p.chunk
        loc_extras: List[str] = []
        if ch.table_caption_ids:
            loc_extras.append(f"Tables: {', '.join(ch.table_caption_ids)}")
        if ch.equation_numbers:
            loc_extras.append(f"Equations: {', '.join(ch.equation_numbers)}")
        if ch.figure_caption_ids:
            loc_extras.append(f"Figures: {', '.join(ch.figure_caption_ids)}")
        extra_str = f" | {'; '.join(loc_extras)}" if loc_extras else ""

        header = (
            f"=== PASSAGE [chunk_id={ch.chunk_id}] ===\n"
            f"Paper: {ch.title} (paper_id={ch.paper_id}, arxiv_version={ch.arxiv_version})\n"
            f"Valid Pages: {ch.page_numbers} (primary_page={ch.primary_page})\n"
            f"Section: {ch.section_title} (Full path: {ch.section_path}){extra_str}\n"
            f"--- Text ---\n{ch.text}\n"
        )
        blocks.append(header)
    return "\n".join(blocks)


def build_generation_messages(
    question: str,
    pieces: List[RetrievedPiece],
    correction_feedback: Optional[str] = None,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a rigorous research assistant answering questions strictly from three arXiv papers:\n"
        "1. Attention Is All You Need (1706.03762, v7)\n"
        "2. Moshi: a speech-text foundation model for real-time dialogue (2410.00037, v1)\n"
        "3. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (2507.07955, v2)\n\n"
        "NON-NEGOTIABLE RULES:\n"
        "- Use ONLY the retrieved passages provided below. Never use external or pre-trained knowledge.\n"
        "- If the provided passages do not explicitly contain the answer to the user's question, set "
        "\"abstained\": true, state clearly what is not covered in \"abstention_reason\" and \"answer_text\", "
        "and return an empty \"claims\" list.\n"
        "- When answering (\"abstained\": false), decompose your answer into atomic factual claims in \"claims\".\n"
        "- Every claim MUST cite the exact \"chunk_id\", \"paper_id\", \"arxiv_version\", \"page_number\" "
        "(must be one of the chunk's Valid Pages), and \"section_title\" from the passage header.\n"
        "- Never invent chunk IDs, paper versions, page numbers, sections, or numbers.\n"
        "- Respond ONLY with a single valid JSON object matching this exact schema:\n"
        "{\n"
        "  \"abstained\": bool,\n"
        "  \"abstention_reason\": str,\n"
        "  \"answer_text\": str,\n"
        "  \"claims\": [\n"
        "    {\n"
        "      \"claim_id\": \"c1\",\n"
        "      \"claim_text\": str,\n"
        "      \"cited_chunk_ids\": [str],\n"
        "      \"citations\": [\n"
        "        {\n"
        "          \"chunk_id\": str,\n"
        "          \"paper_id\": str,\n"
        "          \"arxiv_version\": str,\n"
        "          \"page_number\": int,\n"
        "          \"section_title\": str,\n"
        "          \"locator_detail\": str\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    user_content = (
        f"USER QUESTION:\n{question}\n\n"
        f"RETRIEVED EVIDENCE PASSAGES:\n{format_passages_for_generation(pieces)}\n"
    )
    if correction_feedback:
        user_content += (
            f"\nCRITICAL CORRECTION REQUIRED FROM VERIFIER:\n{correction_feedback}\n"
            "Fix the unsupported claims or citation errors strictly using the passages above. "
            "If the passages do not support the answer, set \"abstained\": true.\n"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_semantic_verification_messages(
    question: str,
    gen_output: GenerationOutput,
    pieces_for_verification: List[RetrievedPiece],
    include_boundary_context: bool = True,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a strict scientific fact-checking verifier.\n"
        "Evaluate the candidate response against the provided paper passages:\n"
        "1. If the candidate response is a positive answer (abstained=false):\n"
        "   - Check whether EVERY claim in 'claims' is directly supported by its cited passage(s) "
        "(including adjacent boundary context where provided).\n"
        "   - Check that the claims actually answer the specific user question without extrapolating.\n"
        "   - Set 'verified': true ONLY if all claims are supported (`unsupported_claim_ids` is empty).\n"
        "2. If the candidate response is a refusal (abstained=true):\n"
        "   - Inspect ALL provided passages carefully.\n"
        "   - If the passages genuinely do NOT contain the answer, set 'verified': true and 'refusal_valid': true.\n"
        "   - If any passage DOES contain the answer that the generator overlooked, set 'verified': false, "
        "'refusal_valid': false, list the 'overlooked_chunk_ids', and explain where the answer is in 'feedback'.\n"
        "Respond ONLY with a single valid JSON object matching:\n"
        "{\n"
        "  \"verified\": bool,\n"
        "  \"supported_claim_ids\": [str],\n"
        "  \"unsupported_claim_ids\": [str],\n"
        "  \"refusal_valid\": bool | null,\n"
        "  \"overlooked_chunk_ids\": [str],\n"
        "  \"feedback\": str\n"
        "}"
    )

    passage_strs: List[str] = []
    for p in pieces_for_verification:
        ch = p.chunk
        b_prev = (
            f"[Preceding boundary context: {ch.prev_boundary_context}]\n"
            if include_boundary_context and ch.prev_boundary_context
            else ""
        )
        b_next = (
            f"\n[Following boundary context: {ch.next_boundary_context}]"
            if include_boundary_context and ch.next_boundary_context
            else ""
        )
        passage_strs.append(
            f"=== PASSAGE [chunk_id={ch.chunk_id} | {ch.paper_id}{ch.arxiv_version} p.{ch.page_numbers} | {ch.section_title}] ===\n"
            f"{b_prev}{ch.text}{b_next}\n"
        )

    candidate_json = json.dumps(gen_output.model_dump(), indent=2, ensure_ascii=False)
    user_content = (
        f"USER QUESTION:\n{question}\n\n"
        f"CANDIDATE RESPONSE TO VERIFY:\n{candidate_json}\n\n"
        f"SUPPORTING PASSAGES:\n{''.join(passage_strs)}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def parse_json_payload(raw_text: str) -> Dict[str, Any]:
    """Extracts and parses a JSON object from raw model output (handling markdown code fences)."""
    cleaned = raw_text.strip()
    m_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if m_fence:
        cleaned = m_fence.group(1)
    else:
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace : last_brace + 1]
    return json.loads(cleaned)
