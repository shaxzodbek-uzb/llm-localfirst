# llm-localfirst

**Local-first LLM routing — keep sensitive data and bulk text labor on your own
models, and call the cloud only for the hard part.**

[![PyPI](https://img.shields.io/pypi/v/llm-localfirst.svg)](https://pypi.org/project/llm-localfirst/)
[![CI](https://github.com/shaxzodbek-uzb/llm-localfirst/actions/workflows/ci.yml/badge.svg)](https://github.com/shaxzodbek-uzb/llm-localfirst/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/llm-localfirst.svg)](https://pypi.org/project/llm-localfirst/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Most LLM routers optimize **which cloud provider** to call for cost or failover.
`llm-localfirst` inverts the default: it runs on **your own local model first**
(Ollama / vLLM / LM Studio) and reaches for the cloud only when the work genuinely
needs it. It adds two things mainstream routers don't:

1. **🔒 Privacy routing that fails closed.** A call you flag `sensitive=True` is pinned
   to a local model and is **never** allowed to fall back to the cloud. If the local
   model is down, the call **raises** — it does not quietly ship your prompt to a
   third-party API.
2. **🤝 Manager–worker delegation.** Give a cloud "director" agent a drop-in tool that
   offloads token-heavy, low-risk text labor (summarize / draft / translate / reformat /
   extract / classify) to a fast **local** worker — cutting cloud spend and keeping bulk
   data on your hardware. (Extracted from a production Pydantic AI agent.)

Plus a model **allowlist guard** (arbitrary model strings are rejected — an SSRF/cost
blast-radius control), cached reachability probing, an **MCP server** wrapper, and a CLI.

---

## The privacy guarantee, in five lines

```python
from llm_localfirst import Router, LocalUnavailable

router = Router.from_env()
try:
    out = router.complete("Redact all PII from this record.",
                          source=customer_record, sensitive=True)
except LocalUnavailable:
    # Local model is down. We did NOT send the record to the cloud. You decide.
    ...
```

`sensitive=True` means *this data must not leave the box*. The router would rather fail
than leak. That asymmetry — sensitive calls fail closed, ordinary bulk calls fall back to
cloud — **is** the product.

---

## Install

```bash
pip install llm-localfirst              # the routing brain — zero provider SDKs
pip install "llm-localfirst[openai]"    # + talk to local Ollama/vLLM/LM Studio (and cloud OpenAI)
pip install "llm-localfirst[anthropic]" # + Claude (the default cloud fallback / reason model)
pip install "llm-localfirst[all]"       # everything (also: mcp, pydantic-ai)
```

| Extra         | Adds                          | Needed for                                   |
| ------------- | ----------------------------- | -------------------------------------------- |
| *(none)*      | `pydantic-settings`           | `router.decide(...)` — pure routing, no calls |
| `openai`      | `openai`                      | running calls on a local OpenAI-compatible server (or cloud OpenAI) |
| `anthropic`   | `anthropic`                   | the default cloud fallback / `reason` model (Claude) |
| `mcp`         | `mcp`                         | `llm-localfirst mcp` (expose the router over MCP) |
| `pydantic-ai` | `pydantic-ai-slim`            | the manager-worker `attach_worker` integration |

The decision path (`decide()`) imports **no** provider SDK, so you can inspect routing —
and run the whole test suite — with nothing but the core installed.

---

## 60-second quickstart (Ollama)

```bash
ollama pull qwen2.5:7b          # any OpenAI-compatible local server works
pip install "llm-localfirst[openai,anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...   # only needed for the cloud fallback / reason path
```

```python
from llm_localfirst import Router, Kind

router = Router.from_env()

# 1) Inspect routing WITHOUT spending a token.
print(router.decide(kind=Kind.BULK))     # -> local  (cheap + private)
print(router.decide(kind="reason"))      # -> cloud  (the hard part)
print(router.decide(sensitive=True))     # -> local  (pinned; never cloud)

# 2) Actually run it. Bulk work prefers local, and falls back to cloud only if local is down.
print(router.complete("Summarize this in one sentence.",
                      source=long_text, kind=Kind.BULK).text)
```

Or from the shell:

```bash
llm-localfirst doctor                      # show config, the allowlist, and local up/down
llm-localfirst route "summarize this" --kind bulk
llm-localfirst route "redact this" --sensitive    # exits non-zero if local is down (fail-closed)
```

---

## How routing decides

`decide()` probes whether your local model is reachable (cached), then applies these
rules in order:

| Call                          | Local up        | Local down                          |
| ----------------------------- | --------------- | ----------------------------------- |
| `sensitive=True`              | **local**       | **raises `LocalUnavailable`** (fail-closed) |
| explicit `model="<cloud>"` + `sensitive=True` | — | **raises `PrivacyViolation`**       |
| `kind="reason"`               | cloud           | cloud                               |
| `kind="bulk"` / `"auto"` (default) | local      | cloud fallback (`fell_back=True`)   |
| explicit `model="<name>"`     | that allowlisted model (cloud blocked only when sensitive) |

Any explicit `model` must be a name on the allowlist; an arbitrary string (or a stray
URL) raises `ModelNotAllowed`. That allowlist is the SSRF / cost guard — a caller can
never point the router at a new endpoint or an expensive model it wasn't configured with.

---

## Manager–worker delegation (Pydantic AI)

Let a cloud director keep the planning and tool-calls, and offload the grunt text work to
a local worker:

```python
from pydantic_ai import Agent
from llm_localfirst import Router
from llm_localfirst.integrations.pydantic_ai import attach_worker

router = Router.from_env()
director = Agent("anthropic:claude-haiku-4-5", system_prompt="...")

# Adds a `delegate_to_worker(task, source)` tool that routes to your LOCAL model.
# attach_worker REFUSES a non-local worker, so delegated source text can't leak.
attach_worker(director, router, worker_model="local",
              on_delegate=lambda task, result: ...)  # optional observability hook
```

The director calls `delegate_to_worker` for summaries, drafts, translations,
reformatting, and extraction; those run on your GPU instead of burning cloud tokens.
See [`examples/manager_worker.py`](examples/manager_worker.py).

---

## MCP-native

Expose the router to any MCP client (Claude Desktop, IDEs, agents) as two tools —
`route` (dry decision) and `complete`:

```bash
pip install "llm-localfirst[mcp]"
llm-localfirst mcp        # serves over stdio
```

---

## How it compares

`llm-localfirst` is **not** a general multi-provider gateway, and it isn't trying to be.
To be clear and fair: **LiteLLM and Bifrost can already route to local models** (Ollama,
vLLM) — local capability is not the differentiator. The differentiators are the
**fail-closed privacy pin**, the **manager-worker delegation tool**, and a **local-first
default posture**.

| Capability                                            | llm-localfirst | LiteLLM | OpenRouter | llmrouter-lib |
| ----------------------------------------------------- | :------------: | :-----: | :--------: | :-----------: |
| Route to local models (Ollama/vLLM)                   | ✅             | ✅      | ❌         | ➖            |
| **Default posture is local-first**                    | ✅             | ❌ (cloud proxy) | ❌  | ➖            |
| **Sensitive calls fail closed — never fall back to cloud** | ✅        | ❌      | ❌         | ❌            |
| **Manager-worker delegation tool (cloud→local)**      | ✅             | ❌      | ❌         | ❌            |
| Allowlist guard (reject arbitrary model strings)      | ✅             | ➖      | ➖         | ➖            |
| Many cloud providers / load-balancing / caching       | ➖ (by design) | ✅      | ✅         | ➖            |

If you want a broad cloud gateway with dozens of providers, use LiteLLM. If you want your
**private** data to stay local by construction and your **bulk** work to run on your own
hardware, that's this library.

---

## What this is NOT

- **Not a multi-cloud gateway.** It ships one local backend + Claude (+ optional OpenAI).
  Add more by registering them on the allowlist; it won't grow a hundred provider shims.
- **Not a content classifier.** *You* tag a call `sensitive=True` (or pick a `kind`). It
  does not guess whether your text is private — it enforces what you declare.
- **Not load-balancing / semantic caching / cost analytics.** Those are gateway features;
  this is a routing *policy* with a privacy guarantee.
- **Not a prompt firewall.** It controls *where* a call runs, not what's in it.

---

## Configuration

All settings are read from the environment (prefix `LF_`) or a `.env` file. See
[`.env.example`](.env.example). Highlights:

| Variable                  | Default                       | Meaning                                  |
| ------------------------- | ----------------------------- | ---------------------------------------- |
| `LF_LOCAL_BASE_URL`       | `http://localhost:11434/v1`   | local OpenAI-compatible endpoint         |
| `LF_LOCAL_MODEL_ID`       | `qwen2.5:7b`                  | local model id                           |
| `LF_FALLBACK_MODEL`       | `haiku`                       | cloud model for non-sensitive fallback   |
| `LF_REASON_MODEL`         | `haiku`                       | cloud model for `kind="reason"`          |
| `LF_SENSITIVE_FAIL_CLOSED`| `true`                        | keep sensitive calls from ever leaking   |
| `LF_PROBE_TTL`            | `30.0`                        | seconds to cache the reachability probe  |

---

## Development

```bash
uv venv && uv pip install -e '.[dev]'
ruff check . && pytest
```

The routing brain (policy, registry, router, reachability) is covered 100% offline — no
network and no provider SDKs required. Contributions welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT © Shaxzodbek Qambaraliyev / Blaze. See [LICENSE](LICENSE).
