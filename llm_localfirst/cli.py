"""Command-line interface for llm-localfirst.

Subcommands:

- ``doctor``   — print settings summary, registry (names + targets), and local
  reachability (up/down).
- ``route``    — print the routing decision for a prompt (no LLM call; only a
  cached reachability probe).
- ``complete`` — run a completion and print the resulting text.
- ``mcp``      — start the MCP server on stdio.

Uses only :mod:`argparse` (no extra dependency). Exits non-zero with a clean,
single-line message when a :class:`~llm_localfirst.LocalFirstError` is raised.
"""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .errors import LocalFirstError
from .policy import Kind
from .registry import default_registry
from .router import Router


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-localfirst",
        description=(
            "Local-first LLM router: keep sensitive data and bulk text labor on "
            "your own models; call the cloud only for the hard part."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Show settings, registry, and local reachability.")

    p_route = sub.add_parser("route", help="Show the routing decision for a prompt.")
    p_route.add_argument("prompt", help="The prompt to route.")
    p_route.add_argument(
        "--sensitive",
        action="store_true",
        help="Treat the call as sensitive (pinned local, fail-closed).",
    )
    p_route.add_argument(
        "--kind",
        choices=[k.value for k in Kind],
        default="auto",
        help="Routing kind (default: auto).",
    )

    p_complete = sub.add_parser("complete", help="Run a completion and print the text.")
    p_complete.add_argument("prompt", help="The prompt to complete.")
    p_complete.add_argument("--source", default="", help="Source text appended to the prompt.")
    p_complete.add_argument(
        "--sensitive",
        action="store_true",
        help="Treat the call as sensitive (pinned local, fail-closed).",
    )
    p_complete.add_argument(
        "--kind",
        choices=[k.value for k in Kind],
        default="auto",
        help="Routing kind (default: auto).",
    )
    p_complete.add_argument(
        "--model",
        default=None,
        help="Explicit allowlisted model name to use (overrides routing).",
    )

    sub.add_parser("mcp", help="Start the MCP server on stdio.")

    return parser


def _cmd_doctor() -> None:
    settings = Settings()
    registry = default_registry(settings)
    router = Router.from_env()

    print("llm-localfirst doctor")
    print("=====================")
    print("Settings:")
    print(f"  local_base_url        = {settings.local_base_url}")
    print(f"  local_model_id        = {settings.local_model_id}")
    print(f"  openai_base_url       = {settings.openai_base_url}")
    print(f"  openai_model_id       = {settings.openai_model_id}")
    print(f"  default_kind          = {settings.default_kind}")
    print(f"  fallback_model        = {settings.fallback_model}")
    print(f"  reason_model          = {settings.reason_model}")
    print(f"  sensitive_fail_closed = {settings.sensitive_fail_closed}")
    print(f"  probe_ttl             = {settings.probe_ttl}")

    print("Registry (allowlist):")
    for name in registry.names():
        ref = registry.get(name)
        print(f"  {name:<10} target={ref.target:<6} provider={ref.provider} id={ref.model_id}")

    local = registry.get("local")
    up = router.reachability.check(local.base_url)
    print(f"Local reachability ({local.base_url}): {'up' if up else 'down'}")


def _cmd_route(args: argparse.Namespace) -> None:
    router = Router.from_env()
    decision = router.decide(sensitive=args.sensitive, kind=args.kind, model=None)
    print(f"model     = {decision.model.name}")
    print(f"target    = {decision.target}")
    print(f"reason    = {decision.reason}")
    print(f"fell_back = {decision.fell_back}")


def _cmd_complete(args: argparse.Namespace) -> None:
    router = Router.from_env()
    result = router.complete(
        args.prompt,
        source=args.source,
        sensitive=args.sensitive,
        kind=args.kind,
        model=args.model,
    )
    print(result.text)


def _cmd_mcp() -> None:
    from .integrations.mcp import build_mcp_server

    server = build_mcp_server()
    server.run()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            _cmd_doctor()
        elif args.command == "route":
            _cmd_route(args)
        elif args.command == "complete":
            _cmd_complete(args)
        elif args.command == "mcp":
            _cmd_mcp()
        else:  # pragma: no cover - argparse enforces a known command
            parser.error(f"unknown command: {args.command}")
    except LocalFirstError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
