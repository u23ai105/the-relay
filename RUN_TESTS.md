# The Relay — test execution order

Run these commands from the project root:

```bash
cd /Users/muzammilmohammad/Projects/the-relay
mkdir -p results/raw
```

Use `.venv/bin/python` so the commands use the project's virtual environment.

## Batch 1 — local Q01 smoke and test suite

```bash
.venv/bin/python scripts/run_rag_smoke.py 2>&1 | tee results/raw/batch1_q01_smoke.log
.venv/bin/python -m pytest -q 2>&1 | tee results/raw/batch1_pytest.log
```

The smoke test must pass the exact Q01 citation check: chunk `1706.03762v7_s500_c0007`, page 5, section `3.2.2 Multi-Head Attention`, Paragraph 12.

## Batch 2 — one real Groq preflight

Create `.env` from `.env.example`, set your own `GROQ_API_KEY`, then run:

```bash
.venv/bin/python scripts/check_groq_live.py 2>&1 | tee results/raw/batch2_groq_preflight.log
```

This script is designed to make at most ONE real inference request with retries disabled. If it returns HTTP 429, stop. Do not rerun it to probe the limit repeatedly. The JSON evidence is written to `results/raw/groq_live_preflight.json`.

## Batch 3 — offline retrieval evaluation

```bash
.venv/bin/python scripts/run_retrieval_eval.py 2>&1 | tee results/raw/batch3_retrieval_eval.log
```

This is a local/offline retrieval diagnostic. It reports passage recall, retrieved evidence tokens, and retrieval/reranking latency. It does NOT by itself produce generated-answer quality, refusal accuracy, or end-to-end LLM latency for Table 3; do not present it as doing so.

## Batch 3b — answer quality across three retrieval settings (Groq)

```bash
.venv/bin/python scripts/run_answer_quality_eval.py --pause-seconds 1.0 2>&1 | tee results/raw/batch3b_answer_quality.log
```

This performs 15 distinct authored questions in total: five questions for each of three retrieval settings. It is a real-provider evaluation; each logical pipeline question can require a generation call and a semantic-verification call. Transport retries are disabled, calls run sequentially, and the script stops on the first HTTP 429 or authentication/permission error. Preserve the timestamped CSV/JSON outputs. The question sets are distinct across settings rather than a paired same-question comparison; disclose that limitation in the report.

If Batch 2 preflight returned HTTP 429, do not start this batch.

## Batch 4 — five assignment-required stand-in load profiles

The test kit includes updated `scripts/benchmark_provider.py` and a new `scripts/run_required_load_profiles.py`. The provider update adds an opt-in `load_sensitive` fixture mode: concurrent streams receive additional simulated service delay. Existing 450-run fault scenarios are unchanged unless this header is explicitly used.

Run:

```bash
.venv/bin/python scripts/run_required_load_profiles.py 2>&1 | tee results/raw/batch4_official_load_profiles.log
```

This makes NO Groq calls. It writes a timestamped CSV and JSON summary under `results/raw/` for:

1. `steady`
2. `burst`
3. `sustained_overload`
4. `per_minute_limit_halved`
5. `daily_budget_exhausted`

The RPM-halved and TPD-exhausted cases lower the local scheduler's quota caps. These are stand-in simulations, not measured Groq limits. The traffic-shape cases use load-dependent stand-in latency plus scheduler queueing. Report Table 1 only from genuine Groq documentation/headers and the single live preflight evidence.

## Preserve existing results

Do not rerun `scripts/run_load_experiments.py` just to obtain the five official profiles. That script runs the earlier 450-request matrix (baseline/RPM/disconnect/TPD/timeout) with three retry policies. Preserve its existing CSV/JSON and the new timestamped official-profile outputs as separate evidence until final reconciliation.

## Stop conditions

- If Batch 1 fails, stop before Batch 2 and inspect the failure.
- If Batch 2 returns 429, stop real-provider probing; retain its JSON/log.
- Batch 3 and Batch 4 are local and can be run without a Groq key.
- Do not create the final report until raw outputs have been reviewed and every table value can be recomputed from the final request-level `results.csv`.
