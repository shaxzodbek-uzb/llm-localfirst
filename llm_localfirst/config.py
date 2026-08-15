"""Configuration for llm-localfirst.

All settings are read from the environment (prefix ``LF_``) or an ``.env`` file via
``pydantic-settings``. This is the only hard dependency of the routing brain, so the
``decide()`` path never needs a provider SDK installed.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from env (``LF_*``) and ``.env``.

    The defaults target a stock local Ollama install plus Claude in the cloud, which
    is the recommended setup for the privacy-routing and manager-worker patterns.

    Attributes:
        local_base_url: OpenAI-compatible base URL of the local model server
            (Ollama by default; vLLM and LM Studio expose the same shape).
        local_model_id: Concrete model id served locally.
        local_api_key_env: Env var the local backend reads its key from (usually
            unused for local servers, which accept any key).
        anthropic_api_key_env: Env var holding the Anthropic API key.
        openai_api_key_env: Env var holding the cloud OpenAI API key.
        openai_base_url: Base URL for the cloud OpenAI provider.
        openai_model_id: Concrete cloud OpenAI model id.
        default_kind: Default :class:`~llm_localfirst.policy.Kind` value as a string.
        fallback_model: Allowlist key used when a non-sensitive call must leave
            local because the local model is down.
        reason_model: Allowlist key used for ``Kind.REASON`` work.
        sensitive_fail_closed: When ``True``, sensitive calls fail closed (raise)
            rather than ever touching the cloud when local is unreachable.
        probe_ttl: Seconds to cache a local-reachability probe result.
        prices: Per-million-token prices keyed by allowlist name, as
            ``[input, output]`` — e.g. ``LF_PRICES='{"haiku": [0.8, 4.0]}'``. Empty
            by default: this library does not ship a price table, because a stale
            hard-coded number is worse than no number.
        max_cloud_calls: Optional ceiling on cloud calls per process.
        max_cloud_tokens: Optional ceiling on total cloud tokens per process.
        max_cloud_cost: Optional ceiling on estimated cloud spend per process.
            Requires a price for every allowlisted cloud model.
    """

    model_config = SettingsConfigDict(env_prefix="LF_", env_file=".env", extra="ignore")

    local_base_url: str = "http://localhost:11434/v1"
    local_model_id: str = "qwen2.5:7b"
    local_api_key_env: str = "LF_LOCAL_API_KEY"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model_id: str = "gpt-4o-mini"
    default_kind: str = "auto"
    fallback_model: str = "haiku"
    reason_model: str = "haiku"
    sensitive_fail_closed: bool = True
    probe_ttl: float = 30.0
    prices: dict[str, tuple[float, float]] = {}
    max_cloud_calls: int | None = None
    max_cloud_tokens: int | None = None
    max_cloud_cost: float | None = None
