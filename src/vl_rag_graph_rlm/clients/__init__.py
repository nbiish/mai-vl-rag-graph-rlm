"""Client factory for creating LM clients."""

from typing import Any

from vl_rag_graph_rlm.clients.anthropic import AnthropicClient
from vl_rag_graph_rlm.clients.base import BaseLM
from vl_rag_graph_rlm.clients.gemini import GeminiClient
from vl_rag_graph_rlm.clients.litellm import LiteLLMClient
from vl_rag_graph_rlm.clients.openai_compatible import (
    OpenAIClient,
    OpenRouterClient,
    ZenMuxClient,
    ZaiClient,
)
from vl_rag_graph_rlm.types import ProviderType


def get_client(provider: ProviderType | str, **kwargs) -> BaseLM:
    """
    Factory function to create LM clients by provider name.

    Args:
        provider: Provider name ('openai', 'openrouter', 'zenmux', 'zai', 'anthropic', 'gemini', 'litellm')
        **kwargs: Provider-specific arguments (api_key, model_name, etc.)

    Returns:
        BaseLM instance

    Examples:
        >>> # OpenRouter
        >>> client = get_client('openrouter', api_key='...', model_name='anthropic/claude-3.5-sonnet')

        >>> # ZenMux
        >>> client = get_client('zenmux', api_key='...', model_name='gpt-4o')

        >>> # z.ai
        >>> client = get_client('zai', api_key='...', model_name='claude-3-opus')

        >>> # OpenAI
        >>> client = get_client('openai', api_key='...', model_name='gpt-4o')

        >>> # Anthropic
        >>> client = get_client('anthropic', api_key='...', model_name='claude-3-5-sonnet-20241022')

        >>> # Gemini
        >>> client = get_client('gemini', api_key='...', model_name='gemini-1.5-pro')

        >>> # LiteLLM (universal)
        >>> client = get_client('litellm', model_name='gpt-4o')
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "openrouter":
        return OpenRouterClient(**kwargs)
    elif provider == "zenmux":
        return ZenMuxClient(**kwargs)
    elif provider == "zai":
        return ZaiClient(**kwargs)
    elif provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "gemini":
        return GeminiClient(**kwargs)
    elif provider == "litellm":
        return LiteLLMClient(**kwargs)
    else:
        # Try LiteLLM for unknown providers (supports 100+ providers)
        return LiteLLMClient(**kwargs)


__all__ = [
    "BaseLM",
    "get_client",
    "OpenAIClient",
    "OpenRouterClient",
    "ZenMuxClient",
    "ZaiClient",
    "AnthropicClient",
    "GeminiClient",
    "LiteLLMClient",
]
