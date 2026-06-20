---
name: Bug report
about: Something doesn't work as documented
title: ""
labels: bug
---

**What happened**
A clear description of the bug.

**Expected**
What you expected to happen.

**Reproduce**
Minimal steps or code. If routing-related, include the `decide()` inputs
(`sensitive`, `kind`, `model`) and what `llm-localfirst route ... ` printed.

**Environment**
- llm-localfirst version:
- Python version:
- Local server (Ollama / vLLM / LM Studio) + model:
- Installed extras (`openai` / `anthropic` / `mcp` / `pydantic-ai`):

**Privacy guarantee?**
If this involves a `sensitive=True` call reaching the cloud, please flag it clearly —
that's a security-relevant bug and we'll prioritize it.
