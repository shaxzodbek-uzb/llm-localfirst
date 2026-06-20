"""Model allowlist: :class:`ModelRef`, :class:`Registry`, and :func:`default_registry`.

The registry is the **SSRF / cost blast-radius guard**: only the short names registered
here may ever be resolved to a concrete provider/model. Arbitrary model strings supplied
by a caller are rejected with :class:`ModelNotAllowed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .errors import ModelNotAllowed

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing config at module load
    from .config import Settings

__all__ = ["Target", "ModelRef", "Registry", "default_registry"]

Target = Literal["local", "cloud"]


@dataclass(frozen=True)
class ModelRef:
    """An immutable, allowlisted reference to one resolvable model.

    Attributes:
        name: Short allowlist key, e.g. ``"local"`` or ``"haiku"``.
        target: ``"local"`` or ``"cloud"`` — drives privacy/fallback routing.
        provider: Backend provider id, ``"openai_compat"`` or ``"anthropic"``.
        model_id: Concrete model id passed to the provider SDK.
        base_url: Endpoint for ``openai_compat`` (local endpoint or cloud); ``None`` otherwise.
        api_key_env: Name of the env var the backend reads the API key from, if any.
    """

    name: str
    target: Target
    provider: str
    model_id: str
    base_url: str | None = None
    api_key_env: str | None = None


class Registry:
    """An immutable allowlist of :class:`ModelRef` keyed by short name."""

    def __init__(self, models: dict[str, ModelRef]) -> None:
        """Build a registry from a mapping of ``name -> ModelRef``."""
        self._models: dict[str, ModelRef] = dict(models)

    def __contains__(self, name: str) -> bool:
        """Return ``True`` iff ``name`` is an allowlisted model key."""
        return name in self._models

    def is_allowed(self, name: str | None) -> bool:
        """Return ``True`` iff ``name`` is a non-``None`` allowlisted key."""
        return name is not None and name in self._models

    def get(self, name: str) -> ModelRef:
        """Resolve a name to its :class:`ModelRef`.

        Raises:
            ModelNotAllowed: If ``name`` is not in the allowlist.
        """
        try:
            return self._models[name]
        except KeyError:
            raise ModelNotAllowed(
                f"model {name!r} is not allowlisted; allowed: {self.names()}"
            ) from None

    def names(self) -> list[str]:
        """Return the allowlisted model names, in insertion order."""
        return list(self._models)

    def by_target(self, target: Target) -> list[ModelRef]:
        """Return all models whose ``target`` matches ``target``."""
        return [m for m in self._models.values() if m.target == target]


def default_registry(settings: Settings | None = None) -> Registry:
    """Build the default allowlist from settings.

    Registers one local model (``"local"``) plus the Claude models
    (``opus``/``sonnet``/``haiku``) and, when an OpenAI cloud key env is configured,
    an optional cloud OpenAI model (``"gpt"``). The resulting allowlist is the
    SSRF/cost guard — only these names may ever be resolved.

    Args:
        settings: Optional :class:`~llm_localfirst.config.Settings`. When ``None`` a
            fresh ``Settings()`` (reading ``LF_*`` env / ``.env``) is constructed.

    Returns:
        A :class:`Registry` containing the default models.
    """
    if settings is None:
        from .config import Settings  # lazy: keep module import provider/config free

        settings = Settings()

    models: dict[str, ModelRef] = {
        "local": ModelRef(
            name="local",
            target="local",
            provider="openai_compat",
            model_id=settings.local_model_id,
            base_url=settings.local_base_url,
            api_key_env=settings.local_api_key_env,
        ),
        "opus": ModelRef(
            name="opus",
            target="cloud",
            provider="anthropic",
            model_id="claude-opus-4-8",
            api_key_env=settings.anthropic_api_key_env,
        ),
        "sonnet": ModelRef(
            name="sonnet",
            target="cloud",
            provider="anthropic",
            model_id="claude-sonnet-4-6",
            api_key_env=settings.anthropic_api_key_env,
        ),
        "haiku": ModelRef(
            name="haiku",
            target="cloud",
            provider="anthropic",
            model_id="claude-haiku-4-5",
            api_key_env=settings.anthropic_api_key_env,
        ),
    }

    if settings.openai_api_key_env:
        models["gpt"] = ModelRef(
            name="gpt",
            target="cloud",
            provider="openai_compat",
            model_id=settings.openai_model_id,
            base_url=settings.openai_base_url,
            api_key_env=settings.openai_api_key_env,
        )

    return Registry(models)
