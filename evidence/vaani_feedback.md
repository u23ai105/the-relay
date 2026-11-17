# Your feedback on Vaani

I reviewed Vaani's public website, including the homepage, platform overview, and developer documentation.

**What stood out:** Vaani presents itself as more than a speech-generation API: the platform connects agent building, natural turn-taking and multilingual interaction, tool integration, call execution, and post-call improvement. I particularly liked the attention to real phone-conversation details such as pauses, interruptions, language changes, and carrying actions through to connected tools. The Teacher AI workflow's reviewable prompt changes and explicit approval step also stood out because they give the builder a chance to inspect changes before applying them.

**What confused me:** The website introduces several related surfaces—Build, Humanise, Connect, Run, Improve, Scale, Teacher AI, Memories, call screening, and developer integrations. The capabilities are interesting, but it takes some effort to understand which part a new developer should start with and how the pieces fit together into the first working deployment. The developer guide does explain the first API call, but the distinction between a call being *dispatched* and actually being *answered or completed* is an important operational detail that could be made more prominent earlier in the onboarding flow.

**One thing I would change:** I would add a single guided “Make your first test call” path from the main landing page. It would show the required prerequisites (API key, agent ID, and test number), guide the developer through triggering a call, and then show where to inspect the call result or receive it through a webhook. I would make the dispatched, answered, and completed states visibly distinct. This would connect the high-level platform story to a concrete first success and reduce the uncertainty about what to do after the initial API request.

---

**Scope of this feedback:** This assessment is based on the publicly accessible website and developer pages. I did not create an authenticated Vaani account, configure an agent, or place a live call, so these comments are about the public product presentation and onboarding rather than hands-on runtime behaviour.

**Pages reviewed:**
- https://www.vaani.dev/
- https://www.vaani.dev/platform
- https://www.vaani.dev/developers
