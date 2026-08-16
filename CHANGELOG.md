# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-08-16

### Added
- **Usage accounting.** Every completion now records its token counts against a
  `Ledger`, queryable per target or per model (`calls`, `tokens`, `cost`,
  `by_model`, `snapshot`). A ledger is always created, because accounting costs
  nothing and a router that can't say what it spent isn't much of a router.
- **A fail-closed cloud spend ceiling.** `Budget(max_cloud_calls=,
  max_cloud_tokens=, max_cloud_cost=)` is enforced at the dispatch boundary: once
  spent, a cloud-bound call raises `BudgetExceeded` rather than quietly
  overspending. Local calls are never gated — they cost nothing, and gating them
  would defeat the point of running local.
- Configuration via `LF_MAX_CLOUD_CALLS`, `LF_MAX_CLOUD_TOKENS`,
  `LF_MAX_CLOUD_COST` and `LF_PRICES`.
- `llm-localfirst complete --usage` prints the tally for a call on **stderr**, so
  redirecting stdout still captures only the completion. `doctor` now reports the
  configured budget and names any allowlisted cloud model that has no price.
- MCP: a third tool, `usage`, so a cloud director can see what it has spent
  delegating to the local worker.

### Notes
- **No price table ships with this package.** Prices are yours to set via
  `LF_PRICES` (per million tokens, `[input, output]`). A hard-coded table goes
  stale silently, and a stale number is worse than no number — so a cost ceiling
  with an unpriced cloud model in the allowlist is refused at construction time
  rather than never triggering.
- Token counts are read from whatever the backend returns and normalised across
  the common field spellings; a backend that reports nothing contributes zero and
  is counted separately as an unpriced call.

### Fixed
- `__version__` still read `0.1.0` in the 0.1.1 release. It now tracks the
  packaged version.

## [0.1.1] — 2026-06-20

### Added
- **`py.typed` marker (PEP 561)** — the package now ships its type information, so
  downstream `mypy` / `pyright` users get full type checking against the public API.
- A terminal **demo** in the README showing the fail-closed privacy guarantee live.
- `mypy` added to the dev extra and the CI pipeline (the package type-checks clean).

### Changed
- README demo image uses an absolute URL so it renders on the PyPI project page too.

## [0.1.0] — 2026-06-20

Initial release.

### Added
- **Local-first router** (`Router`) with `decide()` (pure routing, no LLM call),
  `acomplete()` (async), and `complete()` (sync wrapper).
- **Fail-closed privacy routing**: `sensitive=True` calls are pinned to a local model
  and raise `LocalUnavailable` rather than ever falling back to the cloud. Enforced in
  the policy and re-asserted at the router's dispatch boundary (defense in depth).
- **Routing policy** (`Policy`, `Kind`, `Decision`): `bulk`/`auto` prefer local with
  cloud fallback; `reason` prefers cloud; explicit overrides are allowlist-checked and a
  sensitive cloud override raises `PrivacyViolation`.
- **Allowlist guard** (`Registry`, `ModelRef`, `default_registry`): only registered model
  names resolve — arbitrary strings raise `ModelNotAllowed` (SSRF/cost protection).
- **Cached reachability probe** (`Reachability`) over stdlib `urllib`; never raises.
- **Backends**: `OpenAICompatBackend` (Ollama / vLLM / LM Studio / cloud OpenAI) and
  `AnthropicBackend` (Claude), with lazy provider imports.
- **Manager-worker integration** (`integrations.pydantic_ai.attach_worker`): a
  `delegate_to_worker` tool that offloads bulk text labor to a local worker; refuses a
  non-local worker so delegated text can't leak.
- **MCP server** (`integrations.mcp.build_mcp_server`): `route` and `complete` tools.
- **CLI** (`llm-localfirst`): `doctor`, `route`, `complete`, `mcp`.
- **Config** (`Settings`, env prefix `LF_`).

[Unreleased]: https://github.com/shaxzodbek-uzb/llm-localfirst/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shaxzodbek-uzb/llm-localfirst/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/shaxzodbek-uzb/llm-localfirst/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/shaxzodbek-uzb/llm-localfirst/releases/tag/v0.1.0
