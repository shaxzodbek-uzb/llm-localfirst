"""Exception hierarchy for llm-localfirst.

All errors raised by the library subclass :class:`LocalFirstError`, so callers can
catch the whole family with a single ``except LocalFirstError``.
"""

from __future__ import annotations

__all__ = [
    "LocalFirstError",
    "ModelNotAllowed",
    "LocalUnavailable",
    "PrivacyViolation",
    "BackendError",
    "BudgetExceeded",
]


class LocalFirstError(Exception):
    """Base class for every error raised by llm-localfirst."""


class ModelNotAllowed(LocalFirstError):
    """Raised when a model name is not in the allowlist (SSRF/cost guard)."""


class LocalUnavailable(LocalFirstError):
    """Raised when a sensitive (or fail-closed) call needs the local model but it is unreachable."""


class PrivacyViolation(LocalFirstError):
    """Raised when a sensitive call would be routed to the cloud (e.g. explicit cloud override)."""


class BackendError(LocalFirstError):
    """Wraps a provider-call failure."""


class BudgetExceeded(LocalFirstError):
    """Raised when a cloud call is blocked because the spend ceiling is used up.

    Like :class:`PrivacyViolation`, this fails closed: the call does not quietly
    proceed, and it does not silently downgrade to a different model.
    """
