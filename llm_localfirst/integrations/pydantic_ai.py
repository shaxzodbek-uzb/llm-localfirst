"""pydantic-ai integration: the manager-worker delegation tool.

A cloud "director" agent keeps planning, decisions, and tool-calls for itself,
and offloads token-heavy, low-risk text labor (summarize / draft / translate /
reformat / extract / classify) to a fast *local* worker via a drop-in tool.

The provider SDK (``pydantic_ai``) is imported lazily inside :func:`attach_worker`
so importing this module never requires the optional extra.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import BackendError, PrivacyViolation
from ..policy import Kind
from ..router import Router

# What the model sees — mirrors the production agent's tool description.
_DELEGATE_DOC = (
    "Offload heavy text work to the fast local worker — summarizing, drafting, "
    "translating, reformatting, extracting, classifying — anything token-heavy "
    "over text you already have. Keep decisions/planning/tool-calls for yourself."
)


def attach_worker(
    agent: Any,
    router: Router,
    *,
    worker_model: str = "local",
    on_delegate: Callable[[str, str], Any] | None = None,
) -> None:
    """Register a ``delegate_to_worker`` tool on a pydantic-ai ``Agent``.

    The tool routes work to the local worker as :attr:`Kind.BULK` (privacy-
    preserving, cost-saving) via :meth:`Router.acomplete`. This lets a cloud
    "director" agent offload bulk text labor while retaining planning and
    tool-calls for itself.

    Args:
        agent: A ``pydantic_ai.Agent`` (the "director").
        router: The :class:`~llm_localfirst.Router` that dispatches the call.
        worker_model: Allowlist name of the local worker model. Defaults to
            ``"local"``.
        on_delegate: Optional hook called as ``on_delegate(task, result)`` after
            each delegation, for observability — a place where a confidence/eval
            gate can plug in.

    Raises:
        BackendError: If ``pydantic_ai`` is not installed.
        ModelNotAllowed: If ``worker_model`` is not in the router's allowlist.
        PrivacyViolation: If ``worker_model`` resolves to a non-local target — the
            worker must be local so delegated source text never leaves the box.
    """
    try:
        import pydantic_ai  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via stub in tests
        raise BackendError(
            "pydantic-ai is not installed. Install it with "
            "\"pip install 'llm-localfirst[pydantic-ai]'\"."
        ) from exc

    # Structural privacy guarantee: a worker that the director hands bulk source text
    # to must be LOCAL. Reject a cloud worker at wiring time rather than leaking later.
    worker_ref = router.registry.get(worker_model)  # raises ModelNotAllowed if unknown
    if worker_ref.target != "local":
        raise PrivacyViolation(
            f"worker_model {worker_model!r} resolves to a {worker_ref.target} target; "
            "the manager-worker tool must delegate to a LOCAL model so delegated text "
            "never leaves the box"
        )

    @agent.tool_plain
    async def delegate_to_worker(task: str, source: str = "") -> str:
        result = await router.acomplete(
            task,
            source=source,
            kind=Kind.BULK,
            model=worker_model,
        )
        text = result.text
        if on_delegate is not None:
            on_delegate(task, text)
        return text

    # Carry the model-facing description onto the registered tool callable.
    delegate_to_worker.__doc__ = _DELEGATE_DOC
