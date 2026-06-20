"""llm-localfirst — a local-first LLM router.

Run on your own local model first, and call the cloud only when the work
genuinely needs it. Provides privacy routing (fail-closed), manager-worker
delegation, an allowlist guard, cached reachability probing, an MCP server
wrapper, and a CLI.

The public API re-exported here is the stable surface; provider SDKs
(anthropic / openai / mcp / pydantic_ai) are optional extras that are imported
lazily, so ``import llm_localfirst`` never requires them.
"""

from __future__ import annotations

from .backends.base import CompletionResult
from .config import Settings
from .errors import (
    BackendError,
    LocalFirstError,
    LocalUnavailable,
    ModelNotAllowed,
    PrivacyViolation,
)
from .policy import Decision, Kind, Policy
from .reachability import Reachability
from .registry import ModelRef, Registry, default_registry
from .router import Router

__version__ = "0.1.0"

__all__ = [
    "Router",
    "Policy",
    "Decision",
    "Kind",
    "Registry",
    "ModelRef",
    "Reachability",
    "Settings",
    "default_registry",
    "CompletionResult",
    "LocalFirstError",
    "ModelNotAllowed",
    "LocalUnavailable",
    "PrivacyViolation",
    "BackendError",
]
