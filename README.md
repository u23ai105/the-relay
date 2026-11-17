# The Relay

A Python RAG assistant over the three pinned arXiv papers in `data/papers/`. The application retrieves cited passages, generates a structured answer, verifies support, and records request-level latency, queue, retry, and token telemetry.

## Environment

Python 3.12 is the tested project environment. Create or activate the existing virtual environment, install the repository's `requirements.txt`, and keep the three PDFs plus `data/papers/provenance_manifest.json` and `data/indices/` in place.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the real Groq preflight only, copy `.env.example` to `.env` and set `GROQ_API_KEY`. Never commit or submit `.env`.

## Test scripts

See `RUN_TESTS.md` for the prescribed batch order. The project includes:

- `scripts/run_rag_smoke.py`: one local end-to-end Q01 citation smoke test.
- `scripts/check_groq_live.py`: at most one real Groq preflight request, with retries disabled.
- `scripts/run_retrieval_eval.py`: offline retrieval recall/cost evaluation for the configured retrieval settings.
- `scripts/run_answer_quality_eval.py`: 15 unique end-to-end Groq answer checks, five per retrieval setting, with retries disabled and stop-on-429 behavior.
- `scripts/run_required_load_profiles.py`: the five assignment-required stand-in load profiles.
- `scripts/benchmark_provider.py`: updated with opt-in load-sensitive response delay for those traffic profiles; existing fault scenarios keep their prior behavior.
- `scripts/run_load_experiments.py`: the earlier 450-request stand-in retry/fault experiment. Do not rerun it unless deliberately creating a new separate batch.

## Evidence

Keep all raw measurements under `results/raw/`. Do not delete or replace previous experiment files. The final `results.csv` must be assembled from request-level measurements and must retain the twelve mandatory columns specified in the assignment brief.
