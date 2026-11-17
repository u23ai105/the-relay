# How I worked with the assistant

The following three prompt themes are paraphrased from the working history rather than presented as verbatim quotations.

1. **Audit the original assignment before planning more tests.** I asked for the official requirements and submission items to be checked first, then for a batch-wise test plan. This mattered because the earlier 450-request experiment used five fault scenarios, while the original brief separately required steady, burst, sustained overload, halved per-minute limit, and daily-budget exhaustion. Reading the original brief changed the test plan instead of allowing the wrong scenarios to be relabelled.

2. **Give me runnable files and exact storage paths, not templates or abstract instructions.** This mattered because the deadline was close. I asked for the scripts, where to put them, and the commands to execute on my Mac. I also rejected report and feedback templates as submission artefacts; the final report and feedback should be real completed files.

3. **Stop on provider limits and reconcile the raw evidence.** I asked to use the real Groq service sparingly and to keep the stand-in load tests local. The answer-quality run stopped on HTTP 429 without retries. I did not create extra accounts or keep retrying to force completion.

**One thing the assistant got wrong and how I caught it.** An earlier report and test plan treated the existing 450-request benchmark as if it covered the official required load profiles. I supplied the original four-page assignment PDF and compared its exact Table 2a scenario names. That exposed the mismatch. The test plan was corrected, and I ran the five named profiles separately. The first sustained-overload attempt still did not build a queue (peak queue 1); I used the observed result to increase offered load and ran an 80-request corrected attempt.

**One decision I overruled.** I kept the evaluation within the provider's free-tier constraints and accepted an incomplete answer-quality run rather than bypassing the HTTP 429 limit or using another account. I also chose completed submission files over placeholder templates.

**What is distinctly mine.** I selected and checked the three exact paper versions and their provenance, ran the tests and real preflight from my Mac, inspected the citation returned by Q01, decided which local traffic profiles to run, noticed that the first sustained-overload attempt had not actually overloaded the stand-in, and preserved the provider's 429 stop rather than hiding it. The trade-offs and final claims should be judged against those measurements, not against the assistant's suggested wording.
