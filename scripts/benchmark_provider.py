"""
Batch 3: Stateful Deterministic Benchmark Provider.
Deterministic local integration-test fixture simulating OpenAI-compatible SSE endpoints.
Validates prompts before opening SSE streams (failing closed with HTTP 400 on malformed inputs),
verifies q01 evidence against the canonical corpus record (1706.03762v7_s500_c0007, Page 5,
Section 3.2.2 Multi-Head Attention, Paragraph 12), and enforces strict deterministic q01 validation.
"""

import asyncio
import json
from pathlib import Path
import re
import time
from collections import deque
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Relay Stateful Benchmark Provider")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_500_PATH = PROJECT_ROOT / "data" / "indices" / "chunks_500.jsonl"

Q01_EXPECTED_CHUNK_ID = "1706.03762v7_s500_c0007"
Q01_INSUFFICIENT_CHUNK_ID = "1706.03762v7_s500_c0006"
Q01_EXPECTED_PAPER_ID = "1706.03762"
Q01_EXPECTED_VERSION = "v7"
Q01_EXPECTED_PAGE = 5
Q01_EXPECTED_PARAGRAPH = 12
Q01_EXPECTED_SECTION = "3.2.2 Multi-Head Attention"
Q01_EXPECTED_LOCATOR = "Paragraph 12"

_CANONICAL_CHUNK_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def load_canonical_corpus_index() -> Dict[str, Dict[str, Any]]:
    """
    Loads canonical chunk records from data/indices/chunks_500.jsonl.
    Fails explicitly if the canonical index is missing or empty rather than pretending
    verification was performed against an empty mapping.
    """
    global _CANONICAL_CHUNK_CACHE
    if _CANONICAL_CHUNK_CACHE is not None:
        return _CANONICAL_CHUNK_CACHE
    if not CHUNKS_500_PATH.exists():
        raise RuntimeError(f"Canonical corpus index missing: {CHUNKS_500_PATH}")
    records: Dict[str, Dict[str, Any]] = {}
    with open(CHUNKS_500_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            if cid:
                records[cid] = rec
    if not records:
        raise RuntimeError(f"Canonical corpus index is empty: {CHUNKS_500_PATH}")
    _CANONICAL_CHUNK_CACHE = records
    return records


class ProviderState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.request_attempts: Dict[tuple[str, str], int] = {}
        self.request_evidence: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.rpm_window: deque[float] = deque()
        self.tpd_consumed: int = 0
        self.in_flight: int = 0


state = ProviderState()


@app.post("/reset")
async def reset_state() -> Dict[str, str]:
    state.reset()
    return {"status": "reset_ok"}


def extract_json_object(text: str) -> Dict[str, Any]:
    """String-aware brace-matching JSON extractor for nested candidate objects."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise ValueError("Unmatched braces in JSON extraction")


def _parse_optional_json_field(block_header: str, field_label: str) -> Any:
    m = re.search(rf"{re.escape(field_label)}:\s*(\[.*?\])(?=\n|$)", block_header)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _normalise_evidence_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _canonicalize_prompt_passage(
    chunk_id: str,
    canonical: Dict[str, Any],
    supplied: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate prompt metadata against the canonical index, then return canonical data.

    Prompt passage blocks are transport representations, not sources of provenance.
    Their metadata must agree with the local index and their body must contain the exact
    canonical chunk text. Extra boundary-context text is ignored for evidence validation.
    """
    if not canonical:
        raise ValueError(f"Fixture Error: Chunk '{chunk_id}' is absent from the canonical corpus index.")

    checks = (
        ("title", "title"),
        ("paper_id", "paper_id"),
        ("arxiv_version", "arxiv_version"),
        ("primary_page", "primary_page"),
        ("section_title", "section_title"),
        ("section_path", "section_path"),
        ("all_sections", "all_sections"),
        ("paragraph_refs", "paragraph_refs"),
    )
    for supplied_key, canonical_key in checks:
        value = supplied.get(supplied_key)
        if value is None or value == "":
            continue
        expected = canonical.get(canonical_key)
        if value != expected:
            raise ValueError(
                f"Fixture Error: Prompt metadata mismatch for chunk '{chunk_id}': "
                f"{supplied_key}={value!r}, canonical {canonical_key}={expected!r}."
            )

    supplied_pages = supplied.get("valid_pages")
    canonical_pages = list(canonical.get("page_numbers") or [])
    if supplied_pages is not None and list(supplied_pages) != canonical_pages:
        raise ValueError(
            f"Fixture Error: Prompt metadata mismatch for chunk '{chunk_id}': "
            f"valid_pages={list(supplied_pages)!r}, canonical page_numbers={canonical_pages!r}."
        )

    canonical_text = str(canonical.get("text") or "")
    supplied_text = str(supplied.get("text") or "")
    canonical_norm = _normalise_evidence_text(canonical_text)
    supplied_norm = _normalise_evidence_text(supplied_text)
    if not canonical_norm or canonical_norm not in supplied_norm:
        raise ValueError(
            f"Fixture Error: Supplied passage text for chunk '{chunk_id}' does not contain "
            "the exact canonical chunk text."
        )

    # Return canonical metadata and canonical text. Never propagate prompt-supplied
    # provenance or altered evidence into the fixture verifier.
    return {
        "chunk_id": chunk_id,
        "title": canonical.get("title"),
        "paper_id": canonical.get("paper_id"),
        "arxiv_version": canonical.get("arxiv_version"),
        "valid_pages": canonical_pages,
        "primary_page": canonical.get("primary_page"),
        "section_title": canonical.get("section_title"),
        "section_path": canonical.get("section_path"),
        "all_sections": list(canonical.get("all_sections") or []),
        "paragraph_refs": list(canonical.get("paragraph_refs") or []),
        "text": canonical_text,
    }


def parse_passage_blocks(prompt: str) -> Dict[str, Dict[str, Any]]:
    """Parse passage IDs, validate any supplied metadata, and return canonical evidence.

    The canonical chunks_500.jsonl index is authoritative. Prompt headers and bodies
    are checked for consistency but never overwrite canonical provenance or source text.
    """
    canonical_index = load_canonical_corpus_index()
    passages: Dict[str, Dict[str, Any]] = {}

    full_pattern = re.compile(
        r"=== PASSAGE \[chunk_id=(?P<chunk_id>[^\]]+)\] ===\s*"
        r"Paper:\s*(?P<title>.+?)\s*\(paper_id=(?P<paper_id>[^,\)]+),\s*arxiv_version=(?P<arxiv_version>[^\)]+)\)\s*"
        r"Valid Pages:\s*(?P<valid_pages>[^\n]+?)\s*\(primary_page=(?P<primary_page>\d+)\)\s*"
        r"Section:\s*(?P<section_title>.+?)\s*\(Full path:\s*(?P<section_path>[^\)]+)\)\s*"
        r"(?P<extra_headers>.*?)"
        r"--- Text ---\n(?P<text>.*?)(?=\n=== PASSAGE |\Z)",
        re.DOTALL,
    )

    full_matches = list(full_pattern.finditer(prompt))
    if full_matches:
        for match in full_matches:
            chunk_id = match.group("chunk_id").strip()
            canonical = canonical_index.get(chunk_id)
            if canonical is None:
                raise ValueError(
                    f"Fixture Error: Prompt references unknown chunk '{chunk_id}'."
                )
            extra_headers = match.group("extra_headers") or ""
            all_sections = _parse_optional_json_field(extra_headers, "All Sections")
            paragraph_refs = _parse_optional_json_field(extra_headers, "Paragraph Refs")
            supplied = {
                "title": match.group("title").strip(),
                "paper_id": match.group("paper_id").strip(),
                "arxiv_version": match.group("arxiv_version").strip(),
                "valid_pages": [
                    int(page) for page in re.findall(r"\d+", match.group("valid_pages"))
                ],
                "primary_page": int(match.group("primary_page")),
                "section_title": match.group("section_title").strip(),
                "section_path": match.group("section_path").strip(),
                "all_sections": all_sections,
                "paragraph_refs": paragraph_refs,
                "text": match.group("text").strip(),
            }
            passages[chunk_id] = _canonicalize_prompt_passage(
                chunk_id, canonical, supplied
            )
        return passages

    # Verification prompts use a compact header with a pipe-delimited locator and may
    # include preceding/following boundary context around the actual chunk text.
    header_matches = list(
        re.finditer(
            r"(?:=== PASSAGE )?\[chunk_id=(?P<chunk_id>[^\]|\s]+)(?:[^\]]*)\](?: ===)?",
            prompt,
        )
    )
    for index, header_match in enumerate(header_matches):
        chunk_id = header_match.group("chunk_id").strip()
        canonical = canonical_index.get(chunk_id)
        if canonical is None:
            raise ValueError(f"Fixture Error: Prompt references unknown chunk '{chunk_id}'.")

        block_start = header_match.end()
        block_end = (
            header_matches[index + 1].start()
            if index + 1 < len(header_matches)
            else len(prompt)
        )
        segment = prompt[header_match.start():block_end]
        body = prompt[block_start:block_end].strip()
        if "--- Text ---" in body:
            body = body.split("--- Text ---", 1)[1].strip()

        paper_match = re.search(r"paper_id=([^,\)\s]+)", segment)
        version_match = re.search(r"arxiv_version=([^\)\s]+)", segment)
        pages_match = re.search(r"Valid Pages:\s*([^\n\(]+)", segment)
        primary_match = re.search(r"primary_page=(\d+)", segment)
        section_match = re.search(r"Section:\s*([^\n\(]+)", segment)
        section_path_match = re.search(r"Full path:\s*([^\)]+)", segment)
        all_sections = _parse_optional_json_field(segment, "All Sections")
        paragraph_refs = _parse_optional_json_field(segment, "Paragraph Refs")

        supplied = {
            "paper_id": paper_match.group(1).strip() if paper_match else None,
            "arxiv_version": version_match.group(1).strip() if version_match else None,
            "valid_pages": [int(p) for p in re.findall(r"\d+", pages_match.group(1))]
            if pages_match else None,
            "primary_page": int(primary_match.group(1)) if primary_match else None,
            "section_title": section_match.group(1).strip() if section_match else None,
            "section_path": section_path_match.group(1).strip() if section_path_match else None,
            "all_sections": all_sections,
            "paragraph_refs": paragraph_refs,
            "text": body,
        }
        passages[chunk_id] = _canonicalize_prompt_passage(
            chunk_id, canonical, supplied
        )

    return passages

def _passage_supports_q01_math(passage_text: str) -> bool:
    """
    Verifies the passage text states BOTH h = 8 AND the relational equation
    dk = dv = dmodel/h = 64 (establishing d_k = 64 and d_model = 64 * 8 = 512).
    """
    norm = re.sub(r"\s+", " ", passage_text.lower())
    has_h8 = bool(re.search(r"\bh\s*=\s*8\b", norm))
    has_relational_dk64 = bool(
        re.search(r"d_?k\s*=\s*d_?v\s*=\s*d_?model\s*/\s*h\s*=\s*64\b", norm)
        or re.search(r"d_?model\s*/\s*h\s*=\s*64\b", norm)
    )
    return has_h8 and has_relational_dk64


def _verify_c0007_structural_locator(passage: Dict[str, Any]) -> tuple[int, str, str]:
    """
    Verifies against the canonical passage's paragraph_refs, all_sections, and section heading
    boundaries that:
      1) paragraph_refs contains {page_number: 5, paragraph_index: 12},
      2) all_sections contains '3.2.2 Multi-Head Attention', and
      3) the h=8 / d_model/h=64 paragraph precedes the '3.2.3 Applications of Attention' heading.
    """
    para_refs = passage.get("paragraph_refs") or []
    matching_ref = None
    for ref in para_refs:
        if (
            isinstance(ref, dict)
            and ref.get("page_number") == Q01_EXPECTED_PAGE
            and ref.get("paragraph_index") == Q01_EXPECTED_PARAGRAPH
        ):
            matching_ref = ref
            break

    if matching_ref is None:
        raise ValueError(
            f"Fixture Error: Chunk {passage.get('chunk_id')} lacks canonical paragraph reference "
            f"(page_number={Q01_EXPECTED_PAGE}, paragraph_index={Q01_EXPECTED_PARAGRAPH})."
        )

    all_sections = passage.get("all_sections") or []
    matching_section = None
    for sec in all_sections:
        if str(sec).strip() == Q01_EXPECTED_SECTION:
            matching_section = str(sec).strip()
            break

    if matching_section is None:
        raise ValueError(
            f"Fixture Error: Chunk {passage.get('chunk_id')} all_sections {all_sections} "
            f"does not include '{Q01_EXPECTED_SECTION}'."
        )

    norm = re.sub(r"\s+", " ", passage.get("text", "").lower())
    m_ev = re.search(r"d_?model\s*/\s*h\s*=\s*64\b", norm)
    m_next_sec = re.search(r"3\.2\.3\s+applications of attention", norm)
    if not m_ev or (m_next_sec and m_ev.start() > m_next_sec.start()):
        raise ValueError(
            f"Fixture Error: Chunk {passage.get('chunk_id')} structural heading boundary check failed "
            f"for section '{Q01_EXPECTED_SECTION}'."
        )

    return (
        int(matching_ref["page_number"]),
        matching_section,
        f"Paragraph {int(matching_ref['paragraph_index'])}",
    )


def _answer_consistent_with_q01_claims(answer_text: str) -> bool:
    """Ensures answer prose does not state contradictory values for h, d_k, or d_model."""
    lower_txt = (answer_text or "").lower()
    for attr, expected in (("h", "8"), ("d_k", "64"), ("dk", "64"), ("d_model", "512"), ("dmodel", "512")):
        for m in re.finditer(rf"(?<![a-z0-9_]){attr}\s*=\s*(\d+)(?!\d)", lower_txt):
            if m.group(1) != expected:
                return False
    return True


def build_q01_generation_fixture(passages: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Requires the exact canonical q01 supporting chunk (1706.03762v7_s500_c0007) to be present
    in the retrieved passages, verifies its text and canonical paragraph-level metadata, and
    builds the generation response. Never substitutes an unrelated passage when c0007 is absent.
    """
    if not passages:
        raise ValueError("Fixture Error: No structured passage blocks found in generation prompt.")

    if Q01_EXPECTED_CHUNK_ID not in passages:
        raise ValueError(
            f"Fixture Error: Required supporting chunk '{Q01_EXPECTED_CHUNK_ID}' is absent from retrieved passages "
            f"(received: {list(passages.keys())})."
        )

    canonical_index = load_canonical_corpus_index()
    canonical_c7 = canonical_index.get(Q01_EXPECTED_CHUNK_ID)
    if not canonical_c7:
        raise ValueError(f"Fixture Error: Canonical record '{Q01_EXPECTED_CHUNK_ID}' not found in corpus index.")

    supporting_passage = dict(canonical_c7)
    supporting_passage.update({k: v for k, v in passages[Q01_EXPECTED_CHUNK_ID].items() if v not in (None, [], "")})

    if not _passage_supports_q01_math(supporting_passage.get("text", "")):
        raise ValueError(
            f"Fixture Error: Passage '{Q01_EXPECTED_CHUNK_ID}' text does not contain required q01 mathematical "
            "support (h = 8 and dk = dv = dmodel/h = 64)."
        )

    paper_id = supporting_passage.get("paper_id")
    version = supporting_passage.get("arxiv_version")
    if paper_id != Q01_EXPECTED_PAPER_ID or version != Q01_EXPECTED_VERSION:
        raise ValueError(
            f"Fixture Error: Chunk '{Q01_EXPECTED_CHUNK_ID}' paper/version ({paper_id}, {version}) "
            f"does not match expected ({Q01_EXPECTED_PAPER_ID}, {Q01_EXPECTED_VERSION})."
        )

    page_number, section_title, locator_detail = _verify_c0007_structural_locator(supporting_passage)

    return {
        "abstained": False,
        "abstention_reason": "",
        "answer_text": (
            "The base Transformer model uses 8 parallel attention heads (h=8), with a query/key "
            "dimension of d_k=64, and an overall model dimension of d_model=512."
        ),
        "claims": [
            {
                "claim_id": "c1",
                "claim_text": "h=8, d_k=64, d_model=512",
                "cited_chunk_ids": [Q01_EXPECTED_CHUNK_ID],
                "citations": [
                    {
                        "chunk_id": Q01_EXPECTED_CHUNK_ID,
                        "paper_id": paper_id,
                        "arxiv_version": version,
                        "page_number": page_number,
                        "section_title": section_title,
                        "locator_detail": locator_detail,
                    }
                ],
            }
        ],
    }


def evaluate_q01_verification(
    prompt: str, stored_evidence: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Deterministic q01 integration-test verifier (not a general neural semantic verifier).
    Resolves cited chunks against the canonical index (data/indices/chunks_500.jsonl) as the
    authoritative source of truth, validates complete paragraph-level locators without
    permissive OR-fallbacks, verifies full mathematical support (h=8 and d_model/h=64 -> 512),
    and checks answer/claim prose consistency.
    """
    if "CANDIDATE RESPONSE TO VERIFY:" not in prompt or "SUPPORTING PASSAGES:" not in prompt:
        raise ValueError("Fixture Error: Verification prompt missing required delimiters.")

    canonical_index = load_canonical_corpus_index()
    candidate_json_str = prompt.split("CANDIDATE RESPONSE TO VERIFY:", 1)[1].split("SUPPORTING PASSAGES:", 1)[0]
    evidence_section = prompt.split("SUPPORTING PASSAGES:", 1)[1]

    cand = extract_json_object(candidate_json_str)
    prompt_passages = parse_passage_blocks(evidence_section)

    effective_passages: Dict[str, Dict[str, Any]] = {}
    for cid, pdata in prompt_passages.items():
        if cid not in canonical_index:
            continue
        merged = dict(canonical_index[cid])
        if pdata.get("text"):
            merged["text"] = pdata["text"]
        if stored_evidence and cid in stored_evidence and stored_evidence[cid].get("text"):
            merged["text"] = stored_evidence[cid]["text"]
        effective_passages[cid] = merged

    if cand.get("abstained"):
        return {
            "verified": False,
            "supported_claim_ids": [],
            "unsupported_claim_ids": [],
            "refusal_valid": False,
            "overlooked_chunk_ids": list(effective_passages.keys()),
            "feedback": "Refusal invalid: supporting evidence is present in retrieved passages.",
        }

    claims = cand.get("claims") or []
    if not claims:
        return {
            "verified": False,
            "supported_claim_ids": [],
            "unsupported_claim_ids": [],
            "refusal_valid": None,
            "overlooked_chunk_ids": [],
            "feedback": "No claims provided in candidate response.",
        }

    supported: List[str] = []
    unsupported: List[str] = []
    feedbacks: List[str] = []

    answer_text = str(cand.get("answer_text") or "")
    answer_prose_ok = _answer_consistent_with_q01_claims(answer_text)

    for c in claims:
        c_id = str(c.get("claim_id", "unknown"))
        cited_ids = list(c.get("cited_chunk_ids") or [])
        citations = list(c.get("citations") or [])

        if not cited_ids:
            unsupported.append(c_id)
            feedbacks.append("Missing cited_chunk_ids on claim.")
            continue

        claim_ok = True

        # Validate the structured claim before comparing it with prose. This makes the
        # primary diagnostic identify incorrect claim values when both the claim and
        # answer text contain the same incorrect number.
        claim_compact = re.sub(r"\s+", "", str(c.get("claim_text") or "").lower())
        h_m = re.search(r"h=(\d+)", claim_compact)
        dk_m = re.search(r"d_k=(\d+)", claim_compact)
        dm_m = re.search(r"d_model=(\d+)", claim_compact)

        if not h_m or not dk_m or not dm_m:
            claim_ok = False
            feedbacks.append("Missing required q01 attributes: h, d_k, and d_model must all be specified.")
        elif h_m.group(1) != "8" or dk_m.group(1) != "64" or dm_m.group(1) != "512":
            claim_ok = False
            feedbacks.append(
                f"Incorrect or swapped q01 attribute values: got h={h_m.group(1)}, "
                f"d_k={dk_m.group(1)}, d_model={dm_m.group(1)}."
            )

        if not claim_ok:
            unsupported.append(c_id)
            continue

        if not answer_prose_ok:
            claim_ok = False
            feedbacks.append("Answer text contains numerical assertions contradicting q01 claim values.")

        if claim_ok and cited_ids == [Q01_INSUFFICIENT_CHUNK_ID]:
            claim_ok = False
            feedbacks.append(
                f"{Q01_INSUFFICIENT_CHUNK_ID} alone is insufficient evidence for q01 numerical claims (h=8, d_k=64, d_model=512)."
            )

        if claim_ok:
            for cid in cited_ids:
                if cid not in prompt_passages:
                    claim_ok = False
                    feedbacks.append(f"Citation chunk_id '{cid}' not found in supplied supporting passages.")
                    break
                if cid not in canonical_index:
                    claim_ok = False
                    feedbacks.append(f"Citation chunk_id '{cid}' does not exist in canonical corpus index.")
                    break

        if claim_ok:
            if not citations:
                claim_ok = False
                feedbacks.append("Missing citations list on claim.")
            else:
                cit_chunk_ids = [cit.get("chunk_id") for cit in citations]
                if set(cit_chunk_ids) != set(cited_ids):
                    claim_ok = False
                    feedbacks.append("Citation-object chunk_id mismatch with cited_chunk_ids.")

        if claim_ok:
            for cit in citations:
                cid = cit.get("chunk_id")
                canonical_passage = effective_passages[cid]
                expected_paper = canonical_passage.get("paper_id")
                expected_ver = canonical_passage.get("arxiv_version")

                if cit.get("paper_id") != expected_paper:
                    claim_ok = False
                    feedbacks.append(
                        f"Citation metadata invalid: paper_id '{cit.get('paper_id')}' does not match '{expected_paper}'."
                    )
                    break
                if cit.get("arxiv_version") != expected_ver:
                    claim_ok = False
                    feedbacks.append(
                        f"Citation metadata invalid: arxiv_version '{cit.get('arxiv_version')}' does not match '{expected_ver}'."
                    )
                    break

                try:
                    exp_page, exp_sec, exp_loc = _verify_c0007_structural_locator(canonical_passage)
                except ValueError as ve:
                    claim_ok = False
                    feedbacks.append(str(ve))
                    break

                if cit.get("page_number") != exp_page:
                    claim_ok = False
                    feedbacks.append(
                        f"Citation metadata invalid: page_number {cit.get('page_number')} does not match expected page {exp_page}."
                    )
                    break
                sec = str(cit.get("section_title") or "").strip()
                if sec != exp_sec:
                    claim_ok = False
                    feedbacks.append(
                        f"Citation metadata invalid: section_title '{sec}' does not match '{exp_sec}'."
                    )
                    break
                loc = str(cit.get("locator_detail") or "").strip()
                if loc != exp_loc:
                    claim_ok = False
                    feedbacks.append(
                        f"Citation metadata invalid: locator_detail '{loc}' does not match '{exp_loc}'."
                    )
                    break
                if "paragraph_index" in cit and cit.get("paragraph_index") != Q01_EXPECTED_PARAGRAPH:
                    claim_ok = False
                    feedbacks.append(
                        f"Citation metadata invalid: paragraph_index {cit.get('paragraph_index')} does not match {Q01_EXPECTED_PARAGRAPH}."
                    )
                    break

        if claim_ok:
            combined_cited_text = "\n".join(effective_passages[cid].get("text", "") for cid in cited_ids)
            if not _passage_supports_q01_math(combined_cited_text):
                claim_ok = False
                feedbacks.append(
                    "Cited passage text does not establish full mathematical support (h = 8 and d_model / h = 64)."
                )

        if claim_ok:
            supported.append(c_id)
        else:
            unsupported.append(c_id)

    is_valid = len(unsupported) == 0 and len(supported) > 0
    return {
        "verified": is_valid,
        "supported_claim_ids": supported,
        "unsupported_claim_ids": unsupported,
        "refusal_valid": None,
        "overlooked_chunk_ids": [],
        "feedback": "; ".join(feedbacks),
    }


@app.post("/openai/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        req_data = await request.json()
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": f"Invalid JSON body: {exc}", "type": "invalid_request_error"}},
            status_code=400,
        )

    fault_header = request.headers.get("x-relay-fault", "none;req_id=unknown")
    stage = request.headers.get("x-relay-stage", "generate")

    if stage not in ("generate", "regenerate", "verify", "reverify"):
        return JSONResponse(
            {"error": {"message": f"Unknown stage: {stage}", "type": "invalid_request_error"}},
            status_code=400,
        )

    parts = fault_header.split(";")
    fault = parts[0].strip().lower()
    req_id = "unknown"
    for p in parts[1:]:
        if p.strip().startswith("req_id="):
            req_id = p.strip().split("=", 1)[1].strip()

    attempt_key = (req_id, stage)
    state.request_attempts[attempt_key] = state.request_attempts.get(attempt_key, 0) + 1
    attempt_num = state.request_attempts[attempt_key]

    # This fault flag represents a provider whose daily token budget is already
    # exhausted. It must be deterministic on the first request; tying it to
    # prior successful calls makes isolated tests incorrectly return HTTP 200.
    if fault == "429_tpd":
        state.tpd_consumed = 10_000
        return JSONResponse(
            {"error": {"message": "Tokens per day exceeded (TPD)", "type": "rate_limit_error"}},
            status_code=429,
        )

    now = time.time()
    while state.rpm_window and now - state.rpm_window[0] > 10.0:
        state.rpm_window.popleft()

    if fault == "429_rpm":
        if len(state.rpm_window) >= 10:
            retry_after = max(0.1, 10.0 - (now - state.rpm_window[0]))
            return JSONResponse(
                {"error": {"message": "Rate limit exceeded (RPM)", "type": "rate_limit_error"}},
                status_code=429,
                headers={"Retry-After": str(round(retry_after, 2))},
            )
        state.rpm_window.append(now)

    messages = req_data.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse(
            {"error": {"message": "Fixture Error: 'messages' must be a non-empty list.", "type": "invalid_request_error"}},
            status_code=400,
        )
    prompt = "".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
    if not prompt.strip():
        return JSONResponse(
            {"error": {"message": "Fixture Error: Prompt content is empty.", "type": "invalid_request_error"}},
            status_code=400,
        )

    prevalidated_fixture: Optional[Dict[str, Any]] = None
    if fault != "timeout_ttft":
        try:
            if stage in ("generate", "regenerate"):
                parsed_passages = parse_passage_blocks(prompt)
                prevalidated_fixture = build_q01_generation_fixture(parsed_passages)
                state.request_evidence[req_id] = parsed_passages
            else:
                stored_ev = state.request_evidence.get(req_id)
                prevalidated_fixture = evaluate_q01_verification(prompt, stored_evidence=stored_ev)
        except (ValueError, RuntimeError) as ve:
            return JSONResponse(
                {"error": {"message": str(ve), "type": "invalid_request_error"}},
                status_code=400,
            )

    async def event_generator():
        state.in_flight += 1
        try:
            if fault == "timeout_ttft":
                # Exceed the configured 15s TTFT timeout by a small, deterministic
                # margin. A 20s fixture sleep left abandoned ASGI generator tasks
                # alive after the client had already timed out, making repeated
                # timeout tests interfere with one another during server shutdown.
                # 16s still guarantees the client times out before the first token.
                try:
                    await asyncio.sleep(16.0)
                    yield 'data: {"choices": [{"delta": {"content": "too late"}}]}\n\n'
                except asyncio.CancelledError:
                    raise
                return

            # Optional load-sensitive mode used by the official traffic-profile tests.
            # It adds service delay as concurrent provider streams increase. Existing
            # benchmark scenarios keep their original fixed 20-ms initial delay.
            if fault == "load_sensitive":
                concurrent_extra_s = min(max(state.in_flight - 1, 0), 12) * 0.01
                await asyncio.sleep(0.02 + concurrent_extra_s)
            else:
                await asyncio.sleep(0.02)

            if fault == "disconnect_mid_stream" and attempt_num == 1 and stage in ("generate", "regenerate"):
                yield 'data: {"choices": [{"delta": {"content": "{"}}]}\n\n'
                yield 'data: {"choices": [{"delta": {"content": "\\"abstained\\": false,"}}]}\n\n'
                yield 'data: {"choices": [{"delta": {"content": "\\"answer_text\\": \\"Truncated json..."}}]}\n\n'
                await asyncio.sleep(0.05)
                raise ConnectionError("Simulated TCP socket reset mid-stream")

            payload_str = json.dumps(prevalidated_fixture)
            chunk_size = 32
            for i in range(0, len(payload_str), chunk_size):
                piece = payload_str[i : i + chunk_size]
                yield f'data: {{"choices": [{{"delta": {{"content": {json.dumps(piece)}}}}}]}}\n\n'
                await asyncio.sleep(0.01)

            yield 'data: {"choices": [], "usage": {"prompt_tokens": 1500, "completion_tokens": 150}}\n\n'
            yield "data: [DONE]\n\n"
        finally:
            state.in_flight -= 1

    return StreamingResponse(event_generator(), media_type="text/event-stream")

