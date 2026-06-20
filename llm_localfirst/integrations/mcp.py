"""MCP-native wrapper: expose the router as an MCP server.

Two tools are exposed over MCP — ``route`` (dry decision, no LLM call beyond a
cached reachability probe) and ``complete`` (run and return text). The ``mcp``
SDK is imported lazily inside :func:`build_mcp_server` so importing this module
never requires the optional extra.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..errors import BackendError
from ..router import Router


def build_mcp_server(router: Router | None = None, *, name: str = "llm-localfirst") -> Any:
    """Build an MCP server exposing the local-first router.

    Returns an ``mcp.server.fastmcp.FastMCP`` instance with two tools:

    - ``route(prompt, sensitive=False, kind="auto")`` -> the routing
      :class:`~llm_localfirst.Decision` rendered as a plain ``dict``
      (model name/target, reason, ``fell_back``). Performs only a cached
      reachability probe, no LLM call.
    - ``complete(prompt, source="", sensitive=False, kind="auto")`` -> the
      completion text from the chosen backend.

    Args:
        router: The :class:`~llm_localfirst.Router` to wrap. If ``None``, one is
            built via :meth:`Router.from_env`.
        name: The MCP server name advertised to clients.

    Raises:
        BackendError: If the ``mcp`` SDK is not installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised via stub in tests
        raise BackendError(
            "The MCP SDK is not installed. Install it with "
            "\"pip install 'llm-localfirst[mcp]'\"."
        ) from exc

    if router is None:
        router = Router.from_env()

    server = FastMCP(name)

    @server.tool()
    def route(prompt: str, sensitive: bool = False, kind: str = "auto") -> dict:
        """Decide where a prompt would run (local vs cloud) without calling an LLM.

        Returns the routing decision: which allowlisted model and target were
        chosen, the human-readable reason, and whether it fell back off the
        preferred target.
        """
        decision = router.decide(sensitive=sensitive, kind=kind, model=None)
        data = asdict(decision)
        # Flatten the nested ModelRef to its allowlist name for a clean payload.
        data["model"] = decision.model.name
        return data

    @server.tool()
    async def complete(
        prompt: str,
        source: str = "",
        sensitive: bool = False,
        kind: str = "auto",
    ) -> str:
        """Run a completion through the local-first router and return its text.

        Sensitive calls are pinned to the local model (fail-closed); bulk/auto
        calls prefer local and fall back to the cloud only when local is down.
        """
        result = await router.acomplete(
            prompt,
            source=source,
            sensitive=sensitive,
            kind=kind,
        )
        return result.text

    return server
