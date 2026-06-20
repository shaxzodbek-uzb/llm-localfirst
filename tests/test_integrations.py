"""Tests for the integrations — the manager-worker privacy guard and lazy-import errors.

``pydantic_ai`` and ``mcp`` are not installed in the ``[dev]`` venv. For the worker
tests we inject a stub ``pydantic_ai`` module so the guard logic runs; for MCP we assert
the lazy-import error contract.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from llm_localfirst import BackendError, ModelNotAllowed, Policy, PrivacyViolation, Registry, Router


class _StubReachability:
    def __init__(self, up: bool = True) -> None:
        self.up = up

    def check(self, base_url: str | None, *, timeout: float = 2.0) -> bool:
        return self.up


class _FakeAgent:
    """Minimal pydantic-ai Agent stand-in: captures the tool registered via tool_plain."""

    def __init__(self) -> None:
        self.registered = None

    def tool_plain(self, fn):  # noqa: ANN001 - decorator shim
        self.registered = fn
        return fn


def _router(registry: Registry, policy: Policy, fake_backends) -> Router:
    return Router(
        registry=registry,
        policy=policy,
        reachability=_StubReachability(True),
        backends=dict(fake_backends),
    )


@pytest.fixture
def _stub_pydantic_ai(monkeypatch: pytest.MonkeyPatch):
    """Make `import pydantic_ai` succeed so attach_worker's own logic is exercised."""
    monkeypatch.setitem(sys.modules, "pydantic_ai", types.ModuleType("pydantic_ai"))


def test_attach_worker_registers_local_tool(
    _stub_pydantic_ai, registry, policy, fake_backends
) -> None:
    from llm_localfirst.integrations.pydantic_ai import attach_worker

    agent = _FakeAgent()
    attach_worker(agent, _router(registry, policy, fake_backends), worker_model="local")
    assert agent.registered is not None
    assert agent.registered.__name__ == "delegate_to_worker"
    assert "local worker" in (agent.registered.__doc__ or "")


def test_attach_worker_routes_to_local_backend(
    _stub_pydantic_ai, registry, policy, fake_backends
) -> None:
    from llm_localfirst.integrations.pydantic_ai import attach_worker

    agent = _FakeAgent()
    seen: list[tuple[str, str]] = []
    attach_worker(
        agent,
        _router(registry, policy, fake_backends),
        worker_model="local",
        on_delegate=lambda task, result: seen.append((task, result)),
    )
    out = asyncio.run(agent.registered("summarize this", "some source"))
    assert out == "local-reply"
    assert fake_backends["openai_compat"].called is True
    assert fake_backends["anthropic"].called is False  # bulk delegation stays local
    assert seen and seen[0][0] == "summarize this"  # on_delegate hook fired


def test_attach_worker_rejects_cloud_worker(
    _stub_pydantic_ai, registry, policy, fake_backends
) -> None:
    """A worker wired to a cloud model must be refused at attach time — delegated
    source text must never silently leave the box."""
    from llm_localfirst.integrations.pydantic_ai import attach_worker

    with pytest.raises(PrivacyViolation):
        attach_worker(_FakeAgent(), _router(registry, policy, fake_backends), worker_model="haiku")


def test_attach_worker_rejects_unknown_worker(
    _stub_pydantic_ai, registry, policy, fake_backends
) -> None:
    from llm_localfirst.integrations.pydantic_ai import attach_worker

    with pytest.raises(ModelNotAllowed):
        attach_worker(_FakeAgent(), _router(registry, policy, fake_backends), worker_model="nope")


def test_attach_worker_without_pydantic_ai_raises(
    monkeypatch: pytest.MonkeyPatch, registry, policy, fake_backends
) -> None:
    """With pydantic-ai absent, attach_worker raises a helpful BackendError."""
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)  # forces ImportError on import
    from llm_localfirst.integrations.pydantic_ai import attach_worker

    with pytest.raises(BackendError, match="pydantic-ai"):
        attach_worker(_FakeAgent(), _router(registry, policy, fake_backends))


def test_build_mcp_server_without_mcp_raises() -> None:
    """mcp is not installed in the dev venv -> a helpful BackendError, not ImportError."""
    from llm_localfirst.integrations.mcp import build_mcp_server

    with pytest.raises(BackendError, match="MCP"):
        build_mcp_server()
