# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/shaxzodbek-uzb/llm-localfirst/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shaxzodbek-uzb/llm-localfirst/releases/tag/v0.1.0
