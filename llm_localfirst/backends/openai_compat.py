"""OpenAI-compatible backend.

Drives any server that speaks the OpenAI chat-completions API — that covers the
local engines this library is built for (Ollama, vLLM, LM Studio, llama.cpp's
server, …) as well as cloud OpenAI itself. The endpoint is taken from
``model.base_url`` and the key (when needed) from ``model.api_key_env``.

The ``openai`` SDK is imported lazily so the module can be imported, and the
:meth:`Router.decide` path can run, with no provider library installed.
"""

from __future__ import annotations

import os

from ..errors import BackendError
from ..registry import ModelRef
from .base import CompletionResult

_INSTALL_HINT = (
    "OpenAI-compatible backend requires the 'openai' package. "
    "Install it with: pip install 'llm-localfirst[openai]'"
)


class OpenAICompatBackend:
    """Backend for OpenAI-compatible chat-completions endpoints.

    Works against local servers (Ollama / vLLM / LM Studio) and cloud OpenAI.
    For local servers an API key is usually unnecessary; when no key is found in
    the environment a harmless placeholder (``"not-needed"``) is sent so the SDK
    is satisfied.
    """

    provider = "openai_compat"

    def _client(self, model: ModelRef):
        """Build an ``openai.AsyncOpenAI`` client for ``model`` (lazy import)."""
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
            raise BackendError(_INSTALL_HINT) from exc

        api_key = None
        if model.api_key_env:
            api_key = os.environ.get(model.api_key_env)
        # Local OpenAI-compatible servers typically ignore the key; the SDK still
        # insists on a non-empty value, so fall back to a placeholder.
        return openai.AsyncOpenAI(base_url=model.base_url, api_key=api_key or "not-needed")

    async def complete(
        self,
        model: ModelRef,
        prompt: str,
        *,
        system: str | None = None,
        **opts: object,
    ) -> CompletionResult:
        """Call ``chat.completions`` and return the first choice's text."""
        client = self._client(model)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.chat.completions.create(
                model=model.model_id,
                messages=messages,
                **opts,
            )
        except Exception as exc:  # pragma: no cover - provider/network failure
            raise BackendError(f"openai_compat completion failed: {exc}") from exc

        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""

        usage = None
        if getattr(response, "usage", None) is not None:
            usage = (
                response.usage.model_dump()
                if hasattr(response.usage, "model_dump")
                else dict(response.usage)
            )

        return CompletionResult(text=text, model=model, usage=usage)
