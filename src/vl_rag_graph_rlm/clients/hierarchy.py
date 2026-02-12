"""Provider hierarchy client with automatic fallback."""

import logging
import os
from typing import Any

from vl_rag_graph_rlm.clients.base import BaseLM
from vl_rag_graph_rlm.types import ModelUsageSummary, UsageSummary

logger = logging.getLogger(__name__)

# Default provider hierarchy order.
# Editable via PROVIDER_HIERARCHY env var (comma-separated).
# If openai_compatible or anthropic_compatible have API keys configured,
# they are automatically prepended (user set up a custom endpoint on purpose).
DEFAULT_HIERARCHY = [
    "modalresearch",
    "sambanova",
    "nebius",
    "groq",
    "cerebras",
    "zai",
    "zenmux",
    "openrouter",
    "gemini",
    "deepseek",
    "openai",
    "anthropic",
    "mistral",
    "fireworks",
    "together",
    "azure_openai",
]

# Map provider names to their API key env var names
PROVIDER_KEY_MAP = {
    "modalresearch": "MODAL_RESEARCH_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "nebius": "NEBIUS_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "zai": "ZAI_API_KEY",
    "zenmux": "ZENMUX_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
    "anthropic_compatible": "ANTHROPIC_COMPATIBLE_API_KEY",
}


def get_hierarchy() -> list[str]:
    """Get provider hierarchy from env var or default.

    If PROVIDER_HIERARCHY is set, that order is used as-is.
    Otherwise, openai_compatible and anthropic_compatible are
    automatically prepended if they have API keys configured
    (the user intentionally set up a custom endpoint).

    Returns:
        Ordered list of provider names to try.
    """
    env_val = os.getenv("PROVIDER_HIERARCHY", "").strip()
    if env_val:
        return [p.strip().lower() for p in env_val.split(",") if p.strip()]

    hierarchy = list(DEFAULT_HIERARCHY)

    # Prepend generic SDK providers if configured — they take priority
    # because the user explicitly set up a custom endpoint.
    for generic in reversed(["openai_compatible", "anthropic_compatible"]):
        env_key = PROVIDER_KEY_MAP.get(generic, f"{generic.upper()}_API_KEY")
        key_val = os.getenv(env_key, "")
        if key_val and not key_val.startswith("your_"):
            hierarchy.insert(0, generic)

    return hierarchy


def get_available_providers(hierarchy: list[str] | None = None) -> list[str]:
    """Filter hierarchy to providers that have API keys configured.

    Args:
        hierarchy: Provider order to check. Uses default if None.

    Returns:
        Ordered list of providers with valid API keys set.
    """
    if hierarchy is None:
        hierarchy = get_hierarchy()

    available = []
    for provider in hierarchy:
        env_key = PROVIDER_KEY_MAP.get(provider, f"{provider.upper()}_API_KEY")
        key_val = os.getenv(env_key, "")
        if key_val and not key_val.startswith("your_"):
            available.append(provider)
    return available


def resolve_auto_provider() -> str:
    """Resolve the best available provider from the hierarchy.

    Returns:
        First provider in the hierarchy with a valid API key.

    Raises:
        RuntimeError: If no providers have API keys configured.
    """
    available = get_available_providers()
    if not available:
        raise RuntimeError(
            "No providers have API keys configured. "
            "Set at least one provider's API key in .env or environment. "
            "Run 'vrlmrag --list-providers' to see options."
        )
    return available[0]


class HierarchyClient(BaseLM):
    """Client that tries providers in hierarchy order with automatic fallback.

    On each completion call, attempts the current provider. If it fails
    (rate limit, auth error, network error, etc.), automatically falls
    through to the next available provider in the hierarchy.

    The hierarchy is configurable via the PROVIDER_HIERARCHY env var
    (comma-separated provider names) or defaults to:
        sambanova → nebius → groq → cerebras → zai → zenmux →
        openrouter → gemini → deepseek → openai → anthropic →
        mistral → fireworks → together → azure_openai

    If openai_compatible or anthropic_compatible have API keys set,
    they are automatically prepended (user set up a custom endpoint).

    Providers without API keys are automatically skipped.

    Examples:
        >>> client = HierarchyClient()
        >>> result = client.completion("Hello")  # tries hierarchy in order

        >>> # Start from a specific provider
        >>> client = HierarchyClient(start_provider="groq")
    """

    def __init__(
        self,
        start_provider: str | None = None,
        model_name: str | None = None,
        **kwargs,
    ):
        """Initialize the hierarchy client.

        Args:
            start_provider: Optional provider to start from in the hierarchy.
                If set, the hierarchy begins at this provider (skipping earlier ones).
                If None, uses the full hierarchy from the beginning.
            model_name: Optional model override. If None, each provider uses its default.
            **kwargs: Additional kwargs passed to each provider client.
        """
        # Avoid circular import
        from vl_rag_graph_rlm.clients import get_client

        self._get_client = get_client
        self._kwargs = kwargs
        self._model_override = model_name

        hierarchy = get_available_providers()
        if not hierarchy:
            raise RuntimeError(
                "No providers have API keys configured. "
                "Set at least one provider's API key in .env or environment."
            )

        # If start_provider specified, slice hierarchy from that point
        if start_provider:
            start_provider = start_provider.lower()
            if start_provider in hierarchy:
                idx = hierarchy.index(start_provider)
                hierarchy = hierarchy[idx:]
            else:
                # start_provider not available, use full hierarchy
                logger.warning(
                    "Provider '%s' not available (no API key), using full hierarchy",
                    start_provider,
                )

        self._hierarchy = hierarchy
        self._active_provider: str | None = None
        self._active_client: BaseLM | None = None
        self._failed_providers: set[str] = set()

        # Initialize with first provider
        self._activate_next()

        super().__init__(model_name=self.model_name)

    def _activate_next(self) -> None:
        """Activate the next available provider in the hierarchy."""
        for provider in self._hierarchy:
            if provider in self._failed_providers:
                continue

            try:
                kwargs = dict(self._kwargs)
                if self._model_override:
                    kwargs["model_name"] = self._model_override

                client = self._get_client(provider, **kwargs)
                self._active_provider = provider
                self._active_client = client
                self.model_name = client.model_name
                logger.info(
                    "Hierarchy: activated provider '%s' (model: %s)",
                    provider,
                    client.model_name,
                )
                return
            except Exception as e:
                logger.warning(
                    "Hierarchy: could not initialize '%s': %s", provider, e
                )
                self._failed_providers.add(provider)
                continue

        raise RuntimeError(
            f"All providers in hierarchy exhausted. "
            f"Tried: {self._hierarchy}, "
            f"Failed: {self._failed_providers}"
        )

    @property
    def active_provider(self) -> str:
        """The currently active provider name."""
        return self._active_provider or ""

    @property
    def hierarchy(self) -> list[str]:
        """The full hierarchy (available providers only)."""
        return list(self._hierarchy)

    @property
    def failed_providers(self) -> set[str]:
        """Set of providers that have failed."""
        return set(self._failed_providers)

    def _try_with_fallback(self, method_name: str, *args, **kwargs) -> Any:
        """Execute a method with automatic fallback through the hierarchy."""
        last_error = None

        while self._active_client is not None:
            try:
                method = getattr(self._active_client, method_name)
                result = method(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                failed = self._active_provider
                self._failed_providers.add(failed)
                logger.warning(
                    "Hierarchy: provider '%s' failed (%s: %s), trying next...",
                    failed,
                    type(e).__name__,
                    str(e)[:120],
                )

                # Try to activate next provider
                try:
                    self._activate_next()
                    logger.info(
                        "Hierarchy: fell through to '%s' (model: %s)",
                        self._active_provider,
                        self.model_name,
                    )
                except RuntimeError:
                    break

        raise RuntimeError(
            f"All providers in hierarchy exhausted. Last error: {last_error}"
        )

    def completion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """Make a completion call with automatic provider fallback."""
        return self._try_with_fallback("completion", prompt, model=model)

    async def acompletion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        **kwargs,
    ) -> str:
        """Make an async completion call with automatic provider fallback."""
        last_error = None

        while self._active_client is not None:
            try:
                result = await self._active_client.acompletion(
                    prompt, model=model, **kwargs
                )
                return result
            except Exception as e:
                last_error = e
                failed = self._active_provider
                self._failed_providers.add(failed)
                logger.warning(
                    "Hierarchy: provider '%s' failed (%s: %s), trying next...",
                    failed,
                    type(e).__name__,
                    str(e)[:120],
                )

                try:
                    self._activate_next()
                except RuntimeError:
                    break

        raise RuntimeError(
            f"All providers in hierarchy exhausted. Last error: {last_error}"
        )

    def get_usage_summary(self) -> UsageSummary:
        """Get usage summary from the active client."""
        if self._active_client:
            return self._active_client.get_usage_summary()
        return UsageSummary(models={}, total_prompt_tokens=0, total_completion_tokens=0)

    def get_last_usage(self) -> ModelUsageSummary:
        """Get last usage from the active client."""
        if self._active_client:
            return self._active_client.get_last_usage()
        return ModelUsageSummary(
            model="none", prompt_tokens=0, completion_tokens=0, total_tokens=0, calls=0
        )

    def __repr__(self) -> str:
        available = len(self._hierarchy)
        failed = len(self._failed_providers)
        return (
            f"HierarchyClient(active='{self._active_provider}', "
            f"model='{self.model_name}', "
            f"available={available}, failed={failed})"
        )
