"""Tests for router.py — decision wiring + backend dispatch (SPEC cases 8-9).

Uses the conftest fixtures: ``registry``, ``policy``, ``fake_backends`` (a
provider->FakeBackend map). A ``StubReachability`` replaces the network probe so the
suite stays offline. The critical assertions: a sensitive call only ever touches the
local backend, and the dispatch-boundary guard refuses to leak even if the policy is buggy.
"""

from __future__ import annotations

import pytest

from llm_localfirst import (
    BackendError,
    Decision,
    Kind,
    LocalUnavailable,
    Policy,
    PrivacyViolation,
    Registry,
    Router,
)


class StubReachability:
    """Reachability stand-in with a fixed up/down answer; counts calls."""

    def __init__(self, up: bool) -> None:
        self.up = up
        self.calls = 0

    def check(self, base_url: str | None, *, timeout: float = 2.0) -> bool:
        self.calls += 1
        return self.up


def _router(registry: Registry, policy: Policy, fake_backends, *, up: bool) -> Router:
    return Router(
        registry=registry,
        policy=policy,
        reachability=StubReachability(up),
        backends=dict(fake_backends),
    )


# --- Case 8: decide() wires reachability + policy ---------------------------------------


def test_decide_local_up(registry, policy, fake_backends) -> None:
    r = _router(registry, policy, fake_backends, up=True)
    d = r.decide(sensitive=False, kind=Kind.AUTO)
    assert d.target == "local" and d.fell_back is False


def test_decide_local_down_falls_back(registry, policy, fake_backends) -> None:
    r = _router(registry, policy, fake_backends, up=False)
    d = r.decide(sensitive=False, kind=Kind.BULK)
    assert d.target == "cloud" and d.fell_back is True


def test_decide_coerces_kind_string(registry, policy, fake_backends) -> None:
    r = _router(registry, policy, fake_backends, up=True)
    d = r.decide(kind="reason")  # string, not Kind
    assert d.target == "cloud"


def test_decide_invalid_kind_string_raises(registry, policy, fake_backends) -> None:
    r = _router(registry, policy, fake_backends, up=True)
    with pytest.raises(ValueError):
        r.decide(kind="nonsense")


# --- Case 9: acomplete() dispatches to the right backend; sensitive stays local ---------


async def test_acomplete_bulk_local_dispatches_local(registry, policy, fake_backends) -> None:
    r = _router(registry, policy, fake_backends, up=True)
    result = await r.acomplete("summarize", kind=Kind.BULK)
    assert result.text == "local-reply"
    assert fake_backends["openai_compat"].called is True
    assert fake_backends["anthropic"].called is False


async def test_acomplete_sensitive_only_touches_local(registry, policy, fake_backends) -> None:
    r = _router(registry, policy, fake_backends, up=True)
    result = await r.acomplete("patient record ...", sensitive=True)
    assert result.text == "local-reply"
    assert fake_backends["openai_compat"].called is True
    # The cloud backend must never be invoked for a sensitive call.
    assert fake_backends["anthropic"].called is False


async def test_acomplete_sensitive_local_down_raises_and_calls_nothing(
    registry, policy, fake_backends
) -> None:
    r = _router(registry, policy, fake_backends, up=False)
    with pytest.raises(LocalUnavailable):
        await r.acomplete("secret", sensitive=True)
    assert fake_backends["openai_compat"].called is False
    assert fake_backends["anthropic"].called is False


async def test_acomplete_bulk_fallback_dispatches_cloud(registry, policy, fake_backends) -> None:
    r = _router(registry, policy, fake_backends, up=False)
    result = await r.acomplete("summarize", kind=Kind.BULK)
    assert result.text == "cloud-reply"
    assert fake_backends["anthropic"].called is True
    assert fake_backends["openai_compat"].called is False


async def test_acomplete_appends_source(registry, policy, fake_backends) -> None:
    r = _router(registry, policy, fake_backends, up=True)
    await r.acomplete("translate", source="hello world", kind=Kind.BULK)
    call = fake_backends["openai_compat"].calls[0]
    assert "--- Source text ---" in call.prompt
    assert "hello world" in call.prompt
    assert call.prompt.startswith("translate")


async def test_acomplete_forwards_opts_and_system(registry, policy, fake_backends) -> None:
    r = _router(registry, policy, fake_backends, up=True)
    await r.acomplete("x", kind=Kind.BULK, system="be terse", temperature=0.1)
    call = fake_backends["openai_compat"].calls[0]
    assert call.system == "be terse"
    assert call.opts.get("temperature") == 0.1


async def test_acomplete_missing_backend_raises(registry, policy) -> None:
    # Only a cloud backend registered, but a bulk/local call needs openai_compat.
    from tests.conftest import FakeBackend  # type: ignore

    r = Router(
        registry=registry,
        policy=policy,
        reachability=StubReachability(True),
        backends={"anthropic": FakeBackend("anthropic")},
    )
    with pytest.raises(BackendError):
        await r.acomplete("x", kind=Kind.BULK)


# --- Defence in depth: a buggy policy still cannot leak a sensitive call ----------------


class _LeakyPolicy:
    """A deliberately broken policy that returns a CLOUD decision for a sensitive call,
    to prove the Router's dispatch-boundary guard catches it."""

    local_name = "local"

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def decide(self, *, sensitive, kind, model, local_up) -> Decision:
        cloud = self._registry.get("haiku")
        return Decision(model=cloud, target="cloud", reason="(bug)", fell_back=False)


async def test_router_guard_blocks_leaky_policy(registry, fake_backends) -> None:
    r = Router(
        registry=registry,
        policy=_LeakyPolicy(registry),
        reachability=StubReachability(True),
        backends=dict(fake_backends),
    )
    with pytest.raises(PrivacyViolation):
        await r.acomplete("secret", sensitive=True)
    # No backend was called — the guard fired before dispatch.
    assert fake_backends["anthropic"].called is False
    assert fake_backends["openai_compat"].called is False


# --- Sync wrapper ----------------------------------------------------------------------


def test_complete_sync_wrapper(registry, policy, fake_backends) -> None:
    r = _router(registry, policy, fake_backends, up=True)
    result = r.complete("hi", kind=Kind.BULK)
    assert result.text == "local-reply"


async def test_complete_sync_inside_loop_raises(registry, policy, fake_backends) -> None:
    r = _router(registry, policy, fake_backends, up=True)
    with pytest.raises(RuntimeError):
        r.complete("hi", kind=Kind.BULK)  # called from within the running test loop


# --- from_env() wiring -----------------------------------------------------------------


def test_from_env_builds_default_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pure construction: no network, no provider SDKs. Ensure env doesn't override.
    for var in ("LF_FALLBACK_MODEL", "LF_REASON_MODEL", "LF_SENSITIVE_FAIL_CLOSED"):
        monkeypatch.delenv(var, raising=False)
    r = Router.from_env()
    assert {"local", "opus", "sonnet", "haiku"} <= set(r.registry.names())
    assert r.policy.local_name == "local"
    assert r.policy.fallback_name == "haiku"
    assert r.policy.sensitive_fail_closed is True
    # Stock backends are present for both providers.
    assert set(r.backends) == {"openai_compat", "anthropic"}
