"""Light tests for the backends — the lazy-import error contract.

These run in the ``[dev]``-only venv where no provider SDK is installed, so calling a
backend must raise a helpful :class:`BackendError` (with a pip-install hint) rather than
a bare ``ImportError``. Live provider calls are out of scope (covered by dispatch tests
in test_router.py via the FakeBackend).
"""

from __future__ import annotations

import pytest

from llm_localfirst import BackendError, ModelRef
from llm_localfirst.backends import AnthropicBackend, Backend, OpenAICompatBackend

LOCAL = ModelRef(
    name="local", target="local", provider="openai_compat",
    model_id="qwen2.5:7b", base_url="http://localhost:11434/v1", api_key_env=None,
)
CLOUD = ModelRef(
    name="haiku", target="cloud", provider="anthropic",
    model_id="claude-haiku-4-5", api_key_env="ANTHROPIC_API_KEY",
)


def test_backends_declare_provider() -> None:
    assert OpenAICompatBackend().provider == "openai_compat"
    assert AnthropicBackend().provider == "anthropic"


def test_backends_satisfy_protocol() -> None:
    assert isinstance(OpenAICompatBackend(), Backend)
    assert isinstance(AnthropicBackend(), Backend)


async def test_openai_compat_without_sdk_raises_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the lazy `import openai` to fail even if the SDK happens to be present.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):  # noqa: ANN001 - test shim
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BackendError, match="openai"):
        await OpenAICompatBackend().complete(LOCAL, "hi")


async def test_anthropic_without_sdk_raises_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):  # noqa: ANN001 - test shim
        if name == "anthropic":
            raise ImportError("no anthropic")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BackendError, match="anthropic"):
        await AnthropicBackend().complete(CLOUD, "hi")
