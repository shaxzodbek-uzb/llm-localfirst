"""Provider backends: thin async adapters that actually run a completion.

Each backend imports its provider SDK lazily, so importing this package — and the
whole :meth:`~llm_localfirst.Router.decide` path — works with no provider library
installed. The Router dispatches to a backend by matching ``ModelRef.provider``.
"""

from __future__ import annotations

from .anthropic import AnthropicBackend
from .base import Backend, CompletionResult
from .openai_compat import OpenAICompatBackend

__all__ = ["Backend", "CompletionResult", "AnthropicBackend", "OpenAICompatBackend"]
