# Submission status — The Relay

This checklist is based on the official four-page assignment brief and the source ZIP inspected on 13 November 2026.

| Requirement | Current state in this package | Required next action |
|---|---|---|
| Working pipeline, at least four steps, one retrieval step, branch/loop and tool call | Source modules and tests are present; local smoke script is present | Run final local smoke and preserve output |
| Exact three paper versions | All three PDFs and `provenance_manifest.json` present | Preserve them in final source package |
| Stand-in that can slow, fail, rate-limit and stream | Stand-in and benchmark provider code are present | Document calibration against real observations and major differences |
| Real Groq measurements | No live-preflight JSON in supplied ZIP | Configure local `.env`; execute the one-request preflight once; record documented and observed limits |
| At least three traffic shapes | Existing 450-row data is fault-injection matrix, not the required traffic shapes | Run at least steady, burst and sustained overload; also complete required halved-RPM and exhausted-daily-budget rows |
| Required Table 1 | No completed real-provider limit table in ZIP | Derive from documentation and actual headers/responses; include method and confidence |
| Required Tables 2a/2b | Existing diagnostic JSON/CSV available but wrong scenario set for official table | Recompute from all final request-level rows |
| Retrieval Table 3 | Four offline retrieval settings available; passage recall only | Add answer-level grades, citation correctness, correct refusals, tokens per answer, first-token and total-time measurements |
| Two-page write-up | Not included in this source ZIP | Prepare final two-page PDF with exact brief headings and all required tables |
| Vaani feedback | Not included | Visit Vaani and write genuine observations; use `evidence/vaani_feedback_template.md` |
| Assistant conversation history | `conversation/` was empty | Add genuine unedited export, or real prompts if export is unavailable; complete reflection template |
| Canonical `results.csv` | Prepared at project root with the required 12 columns; currently 450 stand-in records only | Append validated rows from missing official runs; do not treat interim file as final |
| Complete reproducible source | Main source, tests, scripts, config, papers and indexes present | Re-run tests, exclude secrets/caches, include final logs/docs |

## Existing 450-request run — interpretation

The included corrected batch contains 450 requests, all with `source=stand-in`, in these categories: baseline, RPM contention, mid-stream disconnect, TPD exhaustion, and slow-provider/TTFT timeout, across three retry strategies. These are real recorded stand-in experiment results. They do not establish the official steady, burst, sustained-overload, per-minute-limit-halved, and daily-budget-exhausted profiles. Keep the categories distinct.

## Final packaging checklist

- [ ] Two-page PDF only (writing pages; code and supporting files separate)
- [ ] Exact brief heading order
- [ ] Tables 1, 2a, 2b, and 3 fully populated; write `not measured` where not measured
- [ ] Table numbers recomputable from final root `results.csv`
- [ ] One row per request; exact 12 mandatory columns in first 12 positions
- [ ] Real and stand-in requests correctly labelled
- [ ] Raw per-request results and aggregate evidence retained
- [ ] Complete source, requirements, config, corpus, indexes and README included
- [ ] Genuine assistant conversation export/prompts and reflection included
- [ ] Genuine Vaani feedback included
- [ ] No `.env`, API keys, credentials, `.venv`, caches or temporary files
