# The Relay — Batch Evidence Review

Evidence reviewed: `batch_evidence_review.zip`, containing Batch 1–4 logs and the raw JSON/CSV files generated on 2026-11-13.

## Batch 1 — local correctness

- Q01 smoke: pipeline status `ok`; verified gate passed; `abstained=false`; answer grade `correct`.
- Exact final citation: *Attention Is All You Need* (arXiv:1706.03762v7), page 5, Section 3.2.2 Multi-Head Attention, Paragraph 12, chunk `1706.03762v7_s500_c0007`.
- Pytest: 62 passed, 7 deprecation warnings, 75.00 seconds.

## Batch 2 — Groq preflight

- Model: `openai/gpt-oss-20b`.
- One request: HTTP 200; latency 830.77 ms; first token 829.69 ms; input 107 tokens; output 53 tokens.
- Response headers: `x-ratelimit-limit-requests=1000`, `x-ratelimit-remaining-requests=999`; `x-ratelimit-limit-tokens=8000`, `x-ratelimit-remaining-tokens=7829`.
- Reset headers: requests `1m26.4s`; tokens `1.282s`.
- `retry-after` was unavailable on this successful HTTP 200 response. The JSON's separate `retry_after: 1282.0` must not be presented as an HTTP Retry-After header; it numerically matches the token reset of 1.282 seconds converted to milliseconds.
- This one success does not empirically establish RPM, TPD, concurrent in-flight capacity, or recovery after a 429. Use provider documentation for published limits and clearly label them as published, not experimentally discovered.

## Batch 3 — retrieval-only evaluation

All four configurations ran on 18 authored questions. Passage recall uses the 12 questions marked answerable; the six other questions are refusal diagnostics.

| Setting | Top-1 passage recall | Top-k passage recall | Avg evidence tokens | Combined retrieval/rerank median | Combined p95 |
|---|---:|---:|---:|---:|---:|
| 250 / 4→4 / no rerank | 8/12 (66.7%) | 11/12 (91.7%) | 834.7 | 0.89 ms | 2.18 ms |
| 500 / 4→4 / no rerank | 9/12 (75.0%) | 12/12 (100.0%) | 1,644.4 | 0.55 ms | 1.35 ms |
| 500 / 12→4 / heuristic rerank | 9/12 (75.0%) | 12/12 (100.0%) | 1,758.5 | 2.48 ms | 5.66 ms |
| 500 / 12→2 / heuristic rerank | 9/12 (75.0%) | 9/12 (75.0%) | 891.1 | 2.41 ms | 5.42 ms |

These are retrieval metrics, not answer correctness or end-to-end model response latencies. The reranker is explicitly reported as a lightweight heuristic.

## Batch 3b — answer-quality evaluation

- Planned: 15 unique questions, five allocated to each of three configurations.
- Completed: 2 logical pipeline requests, both allocated to `250tok_4to4_rerank_no`.
- Q01: request status `ok`, but grade `refused_wrongly`; the pipeline abstained even though the reference answer was in the paper. The answer used 3,725 input and 615 output tokens (4,340 combined, mixed accounting mode); successful-request TTFT 3,135.57 ms; total 3,492.97 ms.
- Q04: HTTP 429, classified as `tpm_limit`; no retry; no valid final answer; 1,924 input and 497 output tokens recorded before the request was rate-limited.
- The run stopped immediately on 429 as intended. Do not count the infrastructure-failed Q04 as an answer-grade result. Only one answer was actually graded, and it was an incorrect refusal.
- Table 3 answer-quality metrics are therefore incomplete. Retrieval recall must not be substituted for generated-answer correctness.

## Batch 4 — official local stand-in profiles

All 108 requests were sent to the local stateful stand-in, not Groq. Retry strategy was `none`.

| Scenario | Requests | Duration | Offered/min | First token p50 / p95 / p99 (ms) | Total p50 / p95 / p99 (ms) | Success | Rate-limited | Timeout/error | Retries/request | Peak queue | Tokens/answer |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| steady | 12 | 22.335 s | 32.71 | 310.81 / 321.49 / 321.49 | 314.38 / 560.42 / 560.42 | 100.0% | 0.0% | 0.0% | 0.000 | 1 | 3,300 |
| burst | 24 | 2.762 s | 5,976.18 | 2,229.50 / 2,505.95 / 2,518.14 | 2,235.18 / 2,508.34 / 2,520.62 | 100.0% | 0.0% | 0.0% | 0.000 | 22 | 3,300 |
| sustained_overload (first attempt) | 40 | 8.156 s | 306.51 | 332.65 / 340.11 / 342.75 | 335.57 / 343.92 / 346.51 | 100.0% | 0.0% | 0.0% | 0.000 | 1 | 3,300 |
| per_minute_limit_halved | 24 | 0.465 s | 6,095.56 | n/a (0 successes) | n/a (0 successes) | 0.0% | 100.0% | 0.0% | 0.000 | 5 | n/a |
| daily_budget_exhausted | 8 | 0.080 s | 6,397.85 | n/a (0 successes) | n/a (0 successes) | 0.0% | 100.0% | 0.0% | 0.000 | 1 | n/a |

No requests timed out or ended with generic errors. The first three scenarios had provider concurrency caps of 5, 3 and 2 respectively. The two quota scenarios were local scheduler simulations: RPM 15 (half of the configured 30-RPM baseline) and TPD 1. They are not measurements of Groq's live enforcement behavior.

### Required correction: sustained overload

The first `sustained_overload` profile did not create sustained queueing. It offered 306.51 requests/min (about 5.1 requests/s) while the observed mean end-to-end stand-in service time was about 333 ms at concurrency 2, corresponding to an approximate two-slot throughput near 6 requests/s. The recorded peak queue was only 1. It is therefore not defensible to describe this run as a demonstrated overload.

A corrected runner is supplied separately. It changes only this profile to 80 requests at 0.10-second intervals (10 requests/s) with provider concurrency still capped at 2. Run only that profile; keep all original outputs untouched. Replace the old `scripts/run_required_load_profiles.py` with the corrected file and execute:

```bash
.venv/bin/python scripts/run_required_load_profiles.py --profile sustained_overload 2>&1 | tee results/raw/batch4_sustained_overload_corrected.log
```

Then use the corrected sustained-overload row in the final report and final results consolidation, while retaining the original first run as an earlier attempted profile in the engineering history. If the corrected run still has no queue build-up, do not label it overload; report what the measurement shows and adjust again only if time permits.

## Other notes for the final report

- The Table 2b cost value for local stand-in runs should be `not applicable — no provider billing`, rather than implying that the stand-in incurred actual Groq charges. A reference-cost estimate may be shown separately if derived from Groq's current published token price.
- The project config has reference prices of $0.10/M input and $0.50/M output. Groq's current model page for GPT-OSS-20B lists $0.075/M input and $0.30/M output; use the official model page if estimating equivalent hosted-service cost, and label it as reference pricing rather than free-tier cash spend.
- The one-request live preflight confirms a successful connection, but does not establish the concurrent-request limit or actual recovery after a 429.
- Preserve all raw files unchanged. Do not rerun the 450-request benchmark.
