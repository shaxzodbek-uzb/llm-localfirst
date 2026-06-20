"""Cached, never-raising reachability probe for local OpenAI-compatible endpoints.

:class:`Reachability` performs a cheap ``GET <base_url>/models`` (the OpenAI-compatible
health endpoint exposed by Ollama / vLLM / LM Studio) and caches the boolean result per
``base_url`` for a short TTL. It uses only :mod:`urllib` from the stdlib — no extra
dependency — and a monotonic clock, and **never raises**: any error means "down".

The Router calls this so that :class:`~llm_localfirst.policy.Policy` can stay a pure,
I/O-free function over a ``local_up`` boolean.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request

__all__ = ["Reachability"]


class Reachability:
    """Caches whether an OpenAI-compatible ``base_url`` is currently reachable."""

    def __init__(self, ttl: float = 30.0) -> None:
        """Create a probe whose results are cached for ``ttl`` seconds per ``base_url``."""
        self.ttl = ttl
        # cache key -> (result, monotonic_expiry)
        self._cache: dict[str, tuple[bool, float]] = {}

    def check(self, base_url: str | None, *, timeout: float = 2.0) -> bool:
        """Return whether ``<base_url>/models`` answered, using the per-url TTL cache.

        Performs a cheap ``GET`` of the OpenAI-compatible ``/models`` health endpoint.
        The result is cached per ``base_url`` for ``ttl`` seconds. Any error (missing
        ``base_url``, connection refused, timeout, non-2xx status) yields ``False``.
        This method never raises.

        Args:
            base_url: The endpoint base, e.g. ``"http://localhost:11434/v1"``.
            timeout: Per-request socket timeout in seconds.

        Returns:
            ``True`` if the endpoint responded with a success status, else ``False``.
        """
        if not base_url:
            return False

        now = time.monotonic()
        cached = self._cache.get(base_url)
        if cached is not None and cached[1] > now:
            return cached[0]

        result = self._probe(base_url, timeout=timeout)
        self._cache[base_url] = (result, now + self.ttl)
        return result

    def clear(self) -> None:
        """Drop all cached results, forcing the next :meth:`check` to re-probe."""
        self._cache.clear()

    @staticmethod
    def _probe(base_url: str, *, timeout: float) -> bool:
        """Issue a single ``GET <base_url>/models`` request; return success as a bool.

        Swallows every exception (network, URL, timeout, anything) and returns ``False``.
        """
        url = base_url.rstrip("/") + "/models"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed scheme
                status = getattr(resp, "status", None)
                if status is None:
                    status = resp.getcode()
                return 200 <= int(status) < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False
        except Exception:  # pragma: no cover - defensive: never let a probe raise
            return False
