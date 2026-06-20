"""Shared pytest fixtures for the llm-localfirst test suite.

Everything here is offline-safe: no network, no provider SDKs. The only third-party
import the suite needs is ``pydantic-settings`` (a core dependency) plus the ``[dev]``
extra (pytest / pytest-asyncio / anyio). Provider libraries (anthropic, openai, mcp,
pydantic_ai) are intentionally NOT required — the tests exercise the pure routing
brain and dispatch via fakes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm_localfirst import (
    CompletionResult,
    ModelRef,
    Policy,
    Registry,
)

# --------------------------------------------------------------------------------------
# Fixture registry: a fake local model plus stub cloud models. base_url for the local
# model is deliberately a non-routable sentinel so a stray real probe would just fail.
# --------------------------------------------------------------------------------------

LOCAL_BASE_URL = "http://localhost:9/v1"
CLOUD_BASE_URL = "https://api.example-cloud.test/v1"


@pytest.fixture
def models() -> dict[str, ModelRef]:
    """Mapping of allowlist keys -> ModelRef used to build the fixture Registry."""
    return {
        "local": ModelRef(
            name="local",
            target="local",
            provider="openai_compat",
            model_id="qwen2.5:7b",
            base_url=LOCAL_BASE_URL,
            api_key_env="LF_LOCAL_API_KEY",
        ),
        "haiku": ModelRef(
            name="haiku",
            target="cloud",
            provider="anthropic",
            model_id="claude-haiku-4-5",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        "sonnet": ModelRef(
            name="sonnet",
            target="cloud",
            provider="anthropic",
            model_id="claude-sonnet-4-6",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        "opus": ModelRef(
            name="opus",
            target="cloud",
            provider="anthropic",
            model_id="claude-opus-4-8",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        "gpt": ModelRef(
            name="gpt",
            target="cloud",
            provider="openai_compat",
            model_id="gpt-4o-mini",
            base_url=CLOUD_BASE_URL,
            api_key_env="OPENAI_API_KEY",
        ),
    }


@pytest.fixture
def registry(models: dict[str, ModelRef]) -> Registry:
    """A Registry built from the fixture models (the SSRF/cost allowlist under test)."""
    return Registry(models)


@pytest.fixture
def policy(registry: Registry) -> Policy:
    """A Policy with the spec defaults (local/haiku/haiku, fail-closed)."""
    return Policy(
        registry,
        local_name="local",
        fallback_name="haiku",
        reason_name="haiku",
        sensitive_fail_closed=True,
    )


# --------------------------------------------------------------------------------------
# A fake Backend (satisfies the Backend Protocol structurally) that records calls
# instead of talking to any provider. Used to assert dispatch behaviour without network.
# --------------------------------------------------------------------------------------


@dataclass
class _Call:
    model: ModelRef
    prompt: str
    system: str | None
    opts: dict


class FakeBackend:
    """Records every ``complete`` call and returns a canned CompletionResult.

    Structurally compatible with ``llm_localfirst.backends.base.Backend``; importing the
    real Protocol is unnecessary because the Router dispatches by ``provider`` string and
    duck-types ``complete``.
    """

    def __init__(self, provider: str, *, reply: str = "ok") -> None:
        self.provider = provider
        self.reply = reply
        self.calls: list[_Call] = []

    async def complete(
        self,
        model: ModelRef,
        prompt: str,
        *,
        system: str | None = None,
        **opts: object,
    ) -> CompletionResult:
        self.calls.append(_Call(model=model, prompt=prompt, system=system, opts=dict(opts)))
        return CompletionResult(text=self.reply, model=model, usage={"calls": len(self.calls)})

    @property
    def called(self) -> bool:
        return bool(self.calls)


@pytest.fixture
def fake_backends() -> dict[str, FakeBackend]:
    """A provider->FakeBackend mapping covering both stock providers."""
    return {
        "openai_compat": FakeBackend("openai_compat", reply="local-reply"),
        "anthropic": FakeBackend("anthropic", reply="cloud-reply"),
    }
