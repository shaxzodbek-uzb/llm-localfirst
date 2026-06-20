# Contributing to llm-localfirst

Thanks for your interest! This is a small, focused library — the goal is a sharp,
well-tested routing policy with a privacy guarantee, not a sprawling gateway.

## Dev setup

```bash
git clone https://github.com/shaxzodbek-uzb/llm-localfirst
cd llm-localfirst
uv venv && uv pip install -e '.[dev]'
```

(`pip install -e '.[dev]'` works too if you don't use [uv](https://docs.astral.sh/uv/).)

## Run the checks

```bash
ruff check .       # lint (line length 100; E, F, I, UP, B)
ruff format .      # optional: auto-format
pytest             # the full suite — must pass offline, no provider SDKs
```

The whole test suite runs **offline**: no network, and none of the optional provider
SDKs (`openai`, `anthropic`, `mcp`, `pydantic-ai`) are required. Tests fake the network
probe and inject fake backends. Please keep it that way — if you add a test that needs a
live provider, gate it so the default `pytest` run stays hermetic.

## The one rule that matters

The headline guarantee is: **a `sensitive=True` call must never reach a cloud target.**
If you touch `policy.py`, `router.py`, or the integrations, make sure the relevant tests
still hold (`test_policy.py::test_sensitive_never_yields_cloud`,
`test_router.py::test_router_guard_blocks_leaky_policy`) and add a test for any new path
that could route a call. Defense in depth is intentional: the policy enforces it, *and*
the router re-asserts it at the dispatch boundary.

## Design principles

- **Core stays dependency-light.** `import llm_localfirst` and `router.decide(...)` must
  work with only `pydantic-settings`. Provider SDKs are optional extras, imported lazily
  inside the backends/integrations.
- **The allowlist is a guard, not a convenience.** Don't add a path that resolves an
  arbitrary model string.
- **Small surface.** New public API should earn its place. When in doubt, prefer fewer
  moving parts and stdlib over a dependency.

## Pull requests

1. Open an issue first for anything non-trivial so we can agree on the shape.
2. Keep PRs focused; include tests.
3. Ensure `ruff check .` and `pytest` are green (CI runs them on Python 3.10–3.13).
4. Update the README / `CHANGELOG.md` if you change behavior or public API.

By contributing you agree your contributions are licensed under the MIT License.
