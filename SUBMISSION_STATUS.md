# Submission status — The Relay

## Verified in this archive
- Complete source tree, tests, configuration, three pinned paper PDFs, and SHA-256 provenance manifest.
- Q01 smoke test passed with the exact p.5 / Section 3.2.2 / Paragraph 12 citation.
- Full local test suite: 62 passed, 7 dependency deprecation warnings.
- One live Groq preflight: HTTP 200, 830.77 ms, 107 input / 53 output tokens; response headers preserved.
- Offline retrieval evaluation: four settings, 18 questions per setting.
- Request-level raw results from the corrected 450-request fault batch, both official-profile runs, both answer-quality attempts, and the one live preflight.
- Consolidated root `results.csv`: 651 unique request rows. The twelve assignment-required columns are first; auxiliary telemetry fields follow so report aggregates can be recomputed.
- Five official local stand-in load profiles: steady (12), burst (24), corrected sustained overload (80), per-minute limit halved (24), daily budget exhausted (8).
- Two-page report, Vaani feedback, and retained assistant work-history PDF/reflection.

## Material limitations — not fully rubric-complete
- The live answer-quality evaluation planned 15 questions but stopped on HTTP 429 after attempting 10; 9 were graded and 1 was rate-limited. A third live retrieval setting was not reached. The assignment asks for answer quality on at least 15 questions and at least three retrieval settings. Do not claim this requirement is fully met.
- In-flight Groq capacity and time-to-recovery after 429 were not measured. RPM and TPD are documented limits, not experimentally exhausted.
- Vaani feedback is based on the public website and developer documentation; no authenticated agent deployment or live call was tested.
- The assistant reflection paraphrases prompt themes; the attached retained interaction/work-history PDF should be checked by the submitter to ensure it is the full conversation history intended for this requirement.

## Data integrity
- Corrected results.csv was rebuilt from 450 + 108 + 80 + 2 + 10 + 1 request records (651 total).
- The first 40-request sustained-load attempt is named `sustained_overload_initial_underloaded` so it is not confused with the corrected 80-request official scenario.
- All request IDs are unique; status and answer-grade values conform to the brief.
- No `.env`, API key, `.venv`, Python cache, `.DS_Store`, or backup files are included.

**Verdict:** This is the most complete honest package supported by the available evidence, but it does not fully satisfy the minimum 15-answer / three-live-settings quality-evaluation requirement. Submit only with that limitation understood and disclosed.
