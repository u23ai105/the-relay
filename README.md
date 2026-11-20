# The Relay

A verification-first Python RAG assistant and provider-scheduling runtime over three pinned arXiv papers. **The Relay** combines geometry-aware PDF extraction, hybrid BM25 + dense Reciprocal Rank Fusion (RRF) retrieval, optional cross-encoder reranking, two-tier citation and semantic verification, deadline-aware provider scheduling, and strict 64-column request-level telemetry.

---

## Architecture Overview

```text
User Query
  │
  ▼
[1. Deterministic Query Processing] (src/query_and_evidence.py)
  ├─ Technical tokenization & normalization (preserves math/IDs/locators)
  ├─ Paper routing hints & corpus concept alias mapping
  └─ Optional metadata-scoped VerifiedAnswerCache lookup
  │
  ▼
[2. Hybrid Retrieval & Optional Reranking] (src/retrieval.py)
  ├─ BM25Okapi + Dense Vector Search (128-dim SVD or SentenceTransformers)
  ├─ Reciprocal Rank Fusion (k = 60)
  └─ Separate Cross-Encoder Reranking stage (records isolated rerank_ms)
  │
  ▼
[3. Conservative Evidence Sufficiency Gate] (src/query_and_evidence.py)
  ├─ Weak lexical overlap ──► Conditional deterministic query expansion & re-retrieval
  └─ Out-of-domain / zero signal ──► Deterministic refusal (0 LLM calls)
  │
  ▼
[4. Global Provider Scheduler & SSE Client] (src/scheduler.py, src/groq_client.py)
  ├─ Local admission queue (max concurrency: 100, max depth: 500)
  ├─ Rolling-window QuotaManager (RPM / TPM / RPD / TPD reservation & reconciliation)
  ├─ Provider concurrency semaphore (max: 10) & shared 429 cooldown
  └─ Retry strategies: none | naive | aware (monotonic request-wide deadline checks)
  │
  ▼
[5. Structured Generation & Two-Tier Verification] (src/prompts_and_schemas.py, src/verification.py)
  ├─ JSON-structured GenerationOutput (atomic claims + granular citation locators)
  ├─ Tier 1: Deterministic Python validation (chunk ID, paper ID, version, page, section)
  ├─ Tier 2: Context-pruned semantic verification (cited chunks + boundary context, or all chunks on refusal)
  └─ Bounded correction loop (max_correction_attempts = 1) ──► Fail-safe abstention if unverified twice
  │
  ▼
[6. Verified Streaming Delivery & Telemetry] (src/pipeline.py, src/telemetry.py)
  ├─ Chunked async delivery (48 chars/chunk) of verified response + canonical citations
  └─ Validated 64-column non-overlapping latency, queue, retry, token, and cost telemetry
```

---

## Pinned Research Corpus

The corpus is pinned to exact arXiv versions and validated against `data/papers/provenance_manifest.json` using SHA-256 hashes and in-PDF arXiv stamps (`src/extraction.py`):

| arXiv ID | Version | Title |
| :--- | :---: | :--- |
| `1706.03762` | `v7` | *Attention Is All You Need* |
| `2410.00037` | `v1` | *Moshi: a speech-text foundation model for real-time dialogue* |
| `2507.07955` | `v2` | *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* |

---

## Key Capabilities

### Geometry-Aware Extraction & Deterministic Chunking
* **Column- & Layout-Aware PDF Parsing (`src/extraction.py`)**: Uses `pymupdf` to cluster visual rows left-to-right, detect two-column pages, distinguish raised mathematical exponents from superscript footnotes, track a bounded 4-level numbered section hierarchy (plus leaf sub-headers), and extract table/figure captions, grid tables, and equation numbers.
* **Lossless Paragraph-Aware Chunking (`src/chunking.py`)**: Produces deterministic `250`-token and `500`-token chunk sets (`chunks_250.jsonl`, `chunks_500.jsonl`) without mixing papers, splits oversized paragraphs cleanly along sentence/word boundaries, and attaches `45`-token `prev_boundary_context` and `next_boundary_context` snippets for downstream semantic verification.

### Hybrid Retrieval & Isolated Reranking
* **Hybrid RRF Engine (`src/retrieval.py`)**: Fuses `BM25Okapi` lexical rankings and L2-normalized dense vector rankings using Reciprocal Rank Fusion ($k=60$) with optional paper-ID score boosting.
* **Pluggable Backends**: Defaults to memory-safe `lightweight` backends (128-dim SVD/n-gram dense embeddings and a lexical/window/bigram/locator interaction reranker) with optional HuggingFace `sentence-transformers` support (`all-MiniLM-L6-v2` bi-encoder with `.npy` caching and `ms-marco-MiniLM-L-6-v2` cross-encoder).
* **Configured Retrieval Profiles (`configs/default.yaml`)**:

| Setting Name | Chunk Size | Candidates Fetched | Pieces Passed | Rerank | Method |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `250tok_4to4_rerank_no` | 250 tok | 4 | 4 | No | `hybrid_rrf` |
| `500tok_4to4_rerank_no` | 500 tok | 4 | 4 | No | `hybrid_rrf` |
| `500tok_12to4_rerank_yes` *(default)* | 500 tok | 12 | 4 | Yes | `hybrid_rrf` |
| `500tok_12to2_rerank_yes` | 500 tok | 12 | 2 | Yes | `hybrid_rrf` |

### Two-Tier Verification & Grounded Abstention
* **Tier 1 — Deterministic Citation Validation (`src/verification.py`)**: Enforces strict Pydantic schemas (`GenerationOutput`, `ClaimItem`, `CitationLocator`) and verifies that every cited `chunk_id` exists in the retrieved set with exact matching `paper_id`, `arxiv_version`, valid `page_number`, and `section_title`.
* **Tier 2 — Context-Pruned Semantic Verification**:
  * *Positive answers*: Prunes verifier input to only the cited chunks plus adjacent boundary context.
  * *Model refusals*: Passes all retrieved chunks so the verifier catches false refusals where evidence was overlooked.
* **Fail-Safe Abstention**: Triggers at most one bounded regeneration/re-verification cycle (`max_correction_attempts: 1`); if verification still fails, the pipeline safely abstains (`verification_failed_twice`).

### Provider Scheduler, SSE Client & Fault-Injecting Stand-In
* **Global Scheduler (`src/scheduler.py`)**: Coordinates local admission control, rolling-window RPM/TPM/RPD/TPD quota reservations (with post-call token reconciliation and cancellation on queue expiry), provider concurrency limits, and shared 429 cooldown windows.
* **Deadline-Aware SSE Client (`src/groq_client.py`)**: Executes single-attempt OpenAI-compatible SSE requests against Groq or the local stand-in with monotonic request-wide deadline propagation (`ContextVar`), explicit `connect` (`10s`), `ttft` (`15s`), `inter_chunk` (`8s`), and `per_call` (`30s`) timeouts, Groq 429 dimension classification (`rpm_limit`, `tpm_limit`, `rpd_limit`, `tpd_limit`, `rate_limit_unknown`), stage-aware retry feasibility checks, and safe discarding of truncated partial streams.
* **Local Stand-In Provider (`src/stand_in.py`)**: FastAPI OpenAI-compatible SSE server supporting deterministic fault injection via the `X-Relay-Fault` header:

| `X-Relay-Fault` Header | Simulated Behavior |
| :--- | :--- |
| `429_rpm` | HTTP 429 RPM rate limit with `Retry-After: 2.5` |
| `429_tpd` | HTTP 429 terminal daily token budget exhaustion |
| `503` | HTTP 503 transient service unavailable |
| `timeout_ttft` | 20s stall prior to HTTP 504 gateway timeout |
| `disconnect_mid_stream` | Flushes partial SSE chunk, then forces a hard socket reset |
| `hold_mid_stream` | Emits initial chunk and holds stream open indefinitely |

### Non-Overlapping 64-Column Telemetry & Evaluation Suite
* **Strict Latency Partition (`src/telemetry.py`)**: Every request row is validated by `validate_telemetry_row()` against a 64-column schema (the 12 mandatory core columns first, followed by 52 auxiliary diagnostics) enforcing:
  * `waiting_ms = local_queue_wait_ms + primary_provider_queue_wait_ms`
  * `working_ms = query_proc_ms + retrieval_ms + rerank_ms + evidence_eval_ms + primary_gen_ms + det_verify_ms + primary_sem_verify_ms + stream_delivery_ms`
  * `retrying_ms = failed_attempt_work_ms + retry_backoff_ms + retry_queue_wait_ms + correction_work_ms`
  * `total_ms == waiting_ms + working_ms + retrying_ms == (end_time - start_time) * 1000`
  * `first_token_ms + stream_delivery_ms == total_ms` (when verified tokens are streamed; otherwise `stream_delivery_ms == 0.0`)
* **18-Question Evaluation Rubric (`src/evaluation_questions.py`)**: Includes 9 standard answerable questions, 3 naive-retriever trap questions, 4 unanswerable in-domain traps, and 2 out-of-domain questions. Grades strictly separate infrastructure failures (`""`) from answer quality (`correct`, `wrong`, `refused_correctly`, `refused_wrongly`).

---

## Repository Structure

```text
u23ai105-the-relay/
├── configs/
│   └── default.yaml              # Provider, corpus, chunking, retrieval, and pipeline defaults
├── src/
│   ├── chunking.py               # Deterministic 250/500-token paragraph-aware chunker
│   ├── config.py                 # YAML config & .env loader
│   ├── evaluation_questions.py   # 18 evaluation questions & deterministic grading rubric
│   ├── extraction.py             # PyMuPDF geometry-aware extraction & SHA-256 verification
│   ├── groq_client.py            # Async SSE client, 429 classifier & monotonic deadline guard
│   ├── logger.py                 # UTC single-line JSON logger with credential redaction
│   ├── pipeline.py               # 6-stage RAGPipeline orchestrator
│   ├── prompts_and_schemas.py    # Pydantic I/O schemas & generation/verification prompts
│   ├── query_and_evidence.py     # Query normalization, alias expansion, gate & verified cache
│   ├── retrieval.py              # BM25, Dense, Hybrid RRF & Cross-Encoder reranking registry
│   ├── scheduler.py              # Admission queue, QuotaManager, concurrency & retry engine
│   ├── stand_in.py               # FastAPI fault-injecting OpenAI-compatible SSE stand-in
│   ├── telemetry.py              # 64-column schema validator & TelemetryCSVWriter
│   └── verification.py           # Tier 1 citation validator & Tier 2 context-pruning helpers
├── .env.example                  # Template for local GROQ_API_KEY (never commit .env)
├── pytest.ini                    # Strict pytest-asyncio configuration
├── requirements.txt              # Core runtime & test dependencies
├── requirements-retrieval.txt    # Optional sentence-transformers bi-/cross-encoder extras
├── README.md
└── SUBMISSION.md                 # Assignment packaging & evidence checklist
```

> **Runtime Data & Artifacts**: Keep the three pinned PDFs and `provenance_manifest.json` under `data/papers/`, generated chunk/embedding indices under `data/indices/`, raw per-experiment measurements under `results/raw/`, and the canonical request-level `results.csv` at the project root.

---

## Setup & Installation

Python **3.12** is the tested runtime environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To enable HuggingFace bi-encoder / cross-encoder retrieval backends instead of the default lightweight backends:

```bash
python -m pip install -r requirements-retrieval.txt
```

### Environment Variables

For live Groq requests only, copy `.env.example` to `.env` and configure your API key:

```bash
cp .env.example .env
# Add the following line to .env:
# GROQ_API_KEY=gsk_your_key_here
```

Never commit or package `.env`. All logs emitted through `src/logger.py` automatically redact `gsk_*` keys, `Bearer` tokens, and sensitive header keys.

### Running the Local Stand-In Server

Start the FastAPI stand-in provider on the default port configured in `configs/default.yaml`:

```bash
uvicorn src.stand_in:app --host 127.0.0.1 --port 8001
```

---

## Default Configuration (`configs/default.yaml`)

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `provider.real_base_url` | `https://api.groq.com/openai/v1` | Live Groq OpenAI-compatible endpoint |
| `provider.standin_base_url` | `http://127.0.0.1:8001/openai/v1` | Local FastAPI stand-in endpoint |
| `provider.default_model` | `openai/gpt-oss-20b` | Default generation & verification model |
| `provider.temperature` | `0.0` | Deterministic decoding |
| `provider.max_generation_tokens` | `1500` | Max completion tokens for answer generation |
| `provider.max_verification_tokens` | `400` | Max completion tokens for semantic verification |
| `provider.request_deadline_ms` | `45000.0` | End-to-end monotonic request deadline |
| `provider.max_provider_concurrency` | `10` | Max concurrent in-flight provider calls |
| `provider.max_transport_retries` | `2` | Max transport retry attempts per stage |
| `pipeline.retry_strategy` | `aware` | Retry policy (`none`, `naive`, or `aware`) |
| `pipeline.max_correction_attempts` | `1` | Max regeneration + re-verification loops |
| `pipeline.enable_cache` | `false` | Toggles `VerifiedAnswerCache` exact-match lookup |

---

## Testing & Experiment Scripts

Run the automated test suite (configured with `asyncio_mode = strict` in `pytest.ini`):

```bash
pytest
```

Follow `RUN_TESTS.md` for the prescribed batch execution order across the project's experiment scripts:

* `scripts/run_rag_smoke.py` — Local end-to-end `Q01` citation smoke test.
* `scripts/check_groq_live.py` — Single real Groq preflight request with retries disabled.
* `scripts/run_retrieval_eval.py` — Offline retrieval recall and cost evaluation across configured retrieval settings.
* `scripts/run_answer_quality_eval.py` — 15 unique end-to-end Groq answer quality checks (5 per retrieval setting), with retries disabled and stop-on-429 behavior.
* `scripts/run_required_load_profiles.py` — Executes the five required stand-in load profiles (steady, burst, sustained overload, halved RPM, and exhausted daily budget).
* `scripts/benchmark_provider.py` — Stand-in benchmark provider supporting opt-in load-sensitive response delay alongside prior fault scenarios.
* `scripts/run_load_experiments.py` — Earlier 450-request stand-in retry/fault-injection experiment (do not rerun unless intentionally creating a separate batch).

---

## Evidence & Telemetry Handling

* Preserve all raw experiment outputs under `results/raw/` without deleting or overwriting previous batches.
* Assemble the canonical root `results.csv` strictly from validated request-level rows, retaining the 12 mandatory columns in positions 1–12 (`request_id`, `scenario`, `source`, `start_time`, `first_token_ms`, `total_ms`, `status`, `retries`, `tokens_in`, `tokens_out`, `retrieval_setting`, `answer_grade`) and labeling `real` vs. `stand-in` requests accurately.
* See `SUBMISSION.md` for the complete packaging and verification checklist.
