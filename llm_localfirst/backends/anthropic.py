"""Anthropic (Claude) backend.

Drives Claude models via the ``anthropic`` SDK's ``messages.create`` API. The key
is read from the environment variable named by ``model.api_key_env`` (default
``ANTHROPIC_API_KEY``).

The ``anthropic`` SDK is imported lazily so the module can be imported, and the
:meth:`Router.decide` path can run, with no provider library installed.
"""

from __future__ import annotations

import os

from ..errors import BackendError
from ..registry import ModelRef
from .base import CompletionResult

_INSTALL_HINT = (
    "Anthropic backend requires the 'anthropic' package. "
    "Install it with: pip install 'llm-localfirst[anthropic]'"
)

_DEFAULT_MAX_TOKENS = 1024


class AnthropicBackend:
    """Backend for Anthropic's Claude models (``messages.create``)."""

    provider = "anthropic"

    def _client(self, model: ModelRef):
        """Build an ``anthropic.AsyncAnthropic`` client for ``model`` (lazy import)."""
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
            raise BackendError(_INSTALL_HINT) from exc

        api_key = os.environ.get(model.api_key_env) if model.api_key_env else None
        return anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        model: ModelRef,
        prompt: str,
        *,
        system: str | None = None,
        **opts: object,
    ) -> CompletionResult:
        """Call ``messages.create`` and return the concatenated text blocks."""
        client = self._client(model)

        # The Anthropic API requires max_tokens; let callers override via opts.
        opts.setdefault("max_tokens", _DEFAULT_MAX_TOKENS)
        if system:
            opts["system"] = system

        try:
            response = await client.messages.create(
                model=model.model_id,
                messages=[{"role": "user", "content": prompt}],
                **opts,
            )
        except Exception as exc:  # pragma: no cover - provider/network failure
            raise BackendError(f"anthropic completion failed: {exc}") from exc

        text = "".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", "") == "text"
        )

        usage = None
        if getattr(response, "usage", None) is not None:
            usage = (
                response.usage.model_dump()
                if hasattr(response.usage, "model_dump")
                else dict(response.usage)
            )

        return CompletionResult(text=text, model=model, usage=usage)
