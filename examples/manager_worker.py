"""Manager-worker delegation with pydantic-ai and llm-localfirst.

A cloud "director" agent plans and owns the tool calls, but offloads
token-heavy, low-risk text labor (summarize / draft / translate / reformat /
extract / classify) to a fast LOCAL "worker" via a drop-in pydantic-ai tool.

``attach_worker`` registers a ``delegate_to_worker(task, source)`` tool on the
director agent. Calls to that tool are routed by ``Router`` as ``kind='bulk'``,
which prefers the local model — saving cloud tokens and keeping the bulk text
on your own hardware.

Prerequisites
-------------
* A local OpenAI-compatible server (Ollama default ``http://localhost:11434/v1``).
* The pydantic-ai and openai extras::

      pip install "llm-localfirst[pydantic-ai,openai,anthropic]"

* A cloud key for the director, e.g.::

      export ANTHROPIC_API_KEY="sk-ant-..."
"""

from __future__ import annotations

import asyncio

from llm_localfirst import Router


def on_delegate(task: str, result: str) -> None:
    """Observability hook: called every time the director offloads to the worker.

    This is where a confidence / eval gate could plug in to score or veto the
    worker's output before the director uses it.
    """
    preview = result[:80].replace("\n", " ")
    print(f"  [delegated] task={task[:60]!r} -> {preview!r}...")


async def main() -> None:
    # Lazy import so the example file is importable without pydantic-ai.
    from pydantic_ai import Agent

    from llm_localfirst.integrations.pydantic_ai import attach_worker

    router = Router.from_env()

    # The director runs on a cloud model and owns planning + tool calls.
    director = Agent(
        "anthropic:claude-haiku-4-5",
        system_prompt=(
            "You are a director agent. For any heavy text work — summarizing, "
            "drafting, translating, reformatting, extracting, or classifying — "
            "call delegate_to_worker and let the local worker do it. Keep "
            "decisions and planning for yourself."
        ),
    )

    # Register the local-worker delegation tool on the director.
    attach_worker(director, router, worker_model="local", on_delegate=on_delegate)

    article = (
        "Local-first LLM routing inverts the usual cloud-router default: it runs "
        "on your own model first and reaches for the cloud only when the work "
        "genuinely needs it. Sensitive calls are pinned to local and fail closed, "
        "so private prompts never leak to a third-party provider. Bulk text labor "
        "is offloaded to a fast local worker to cut cost and keep data in-house."
    )

    result = await director.run(
        "Summarize this article in two sentences, then translate the summary "
        f"to Spanish:\n\n{article}"
    )
    print("\n[director output]\n" + result.output)


if __name__ == "__main__":
    asyncio.run(main())
