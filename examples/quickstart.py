"""60-second quickstart for llm-localfirst.

Prerequisites
-------------
1. A local OpenAI-compatible server (Ollama is the default endpoint
   ``http://localhost:11434/v1``). Pull a small model, e.g.::

       ollama pull qwen2.5:7b

2. The provider extras — ``openai`` drives the local backend; ``anthropic`` is
   used by the cloud fallback / reason path, so install both to exercise the whole
   example (``[openai]`` alone suffices only while the local server stays up)::

       pip install "llm-localfirst[openai,anthropic]"

3. A cloud key for the fallback / reason path::

       export ANTHROPIC_API_KEY="sk-ant-..."

What this shows
---------------
* ``Router.from_env()`` builds the whole stack from ``LF_*`` env vars / ``.env``.
* ``router.decide(...)`` is a pure routing decision — it probes local
  reachability but makes NO LLM call, so you can inspect where a request would
  go before spending a token.
* ``router.complete(...)`` actually runs the completion on the chosen backend.
* A ``sensitive=True`` call is pinned to the local model and fails closed: if
  the local model is down it raises ``LocalUnavailable`` instead of leaking the
  prompt to the cloud.
"""

from __future__ import annotations

from llm_localfirst import Kind, LocalUnavailable, Router


def main() -> None:
    # Build router, policy, registry, reachability and backends from the
    # environment (LF_* vars + ANTHROPIC_API_KEY / OPENAI_API_KEY). No provider
    # SDK is needed just to call decide().
    router = Router.from_env()

    # 1) Inspect routing WITHOUT calling an LLM. ----------------------------
    # Bulk text labor prefers the local model (cheap + private).
    bulk = router.decide(kind=Kind.BULK)
    print(f"[bulk]      -> {bulk.target:<5} model={bulk.model.name!r} "
          f"reason={bulk.reason!r} fell_back={bulk.fell_back}")

    # Hard reasoning prefers the cloud.
    reason = router.decide(kind="reason")
    print(f"[reason]    -> {reason.target:<5} model={reason.model.name!r} "
          f"reason={reason.reason!r}")

    # Sensitive work is pinned to local and never allowed to reach the cloud.
    private = router.decide(sensitive=True)
    print(f"[sensitive] -> {private.target:<5} model={private.model.name!r} "
          f"reason={private.reason!r}")
    assert private.target == "local", "sensitive must stay local"

    # 2) Run a real completion. --------------------------------------------
    # Non-sensitive bulk work: routes to the local worker if it is up, and
    # falls back to the cloud only if the local model is unreachable.
    result = router.complete(
        "Summarize the following text in one sentence.",
        source="Local-first routing keeps private data and bulk text labor on "
        "your own models, and calls the cloud only for the hard part.",
        kind=Kind.BULK,
    )
    print(f"\n[summary] ({result.model.name}): {result.text}")

    # 3) Demonstrate the fail-closed privacy guarantee. --------------------
    # A sensitive call will RAISE LocalUnavailable if the local model is down,
    # rather than silently sending the prompt to a cloud provider.
    try:
        secret = router.complete(
            "Redact all personal data from this customer record.",
            source="Name: Jane Doe, Card: 4111 1111 1111 1111",
            sensitive=True,
        )
        print(f"\n[redacted] ({secret.model.name}): {secret.text}")
    except LocalUnavailable:
        print("\n[redacted] local model is down -> refused to use the cloud "
              "(fail-closed). Start your local server and retry.")


if __name__ == "__main__":
    main()
