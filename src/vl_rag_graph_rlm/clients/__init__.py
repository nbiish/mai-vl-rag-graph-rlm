"""Client factory for creating LM clients."""

from typing import Any

from vl_rag_graph_rlm.clients.anthropic import AnthropicClient, AnthropicCompatibleClient
from vl_rag_graph_rlm.clients.base import BaseLM
from vl_rag_graph_rlm.clients.gemini import GeminiClient
from vl_rag_graph_rlm.clients.litellm import LiteLLMClient
from vl_rag_graph_rlm.clients.ollama import OllamaClient
from vl_rag_graph_rlm.clients.hierarchy import (
    HierarchyClient,
    get_hierarchy,
    get_available_providers,
    resolve_auto_provider,
    DEFAULT_HIERARCHY,
    PROVIDER_KEY_MAP,
)
from vl_rag_graph_rlm.clients.openai_compatible import (
    AzureOpenAIClient,
    CerebrasClient,
    DeepSeekClient,
    FireworksClient,
    GenericOpenAIClient,
    GroqClient,
    MistralClient,
    NebiusClient,
    OpenAIClient,
    OpenRouterClient,
    SambaNovaClient,
    TogetherClient,
    ZenMuxClient,
    ZaiClient,
)
from vl_rag_graph_rlm.types import ProviderType


def get_client(provider: ProviderType | str, **kwargs) -> BaseLM:
    """
    Factory function to create LM clients by provider name.

    Args:
        provider: Provider name (see examples below)
        **kwargs: Provider-specific arguments (api_key, model_name, etc.)

    Returns:
        BaseLM instance

    Examples:
        >>> # OpenAI (Official)
        >>> client = get_client('openai', api_key='...', model_name='gpt-4o')

        >>> # Generic OpenAI-Compatible (custom base URL)
        >>> client = get_client('openai_compatible', base_url='https://api.example.com/v1', model_name='...')

        >>> # Azure OpenAI
        >>> client = get_client('azure_openai', model_name='gpt-4o')

        >>> # Anthropic (Official Claude)
        >>> client = get_client('anthropic', api_key='...', model_name='claude-3-5-sonnet-20241022')

        >>> # Generic Anthropic-Compatible
        >>> client = get_client('anthropic_compatible', base_url='https://api.example.com', model_name='...')

        >>> # OpenRouter
        >>> client = get_client('openrouter', api_key='...', model_name='kimi/kimi-k2.5')

        >>> # ZenMux (uses provider/model format)
        >>> client = get_client('zenmux', api_key='...', model_name='moonshotai/kimi-k2.5')

        >>> # z.ai (tries Coding Plan endpoint first, falls back to normal)
        >>> client = get_client('zai', api_key='...', model_name='glm-4.7')

        >>> # Groq (ultra-fast LPU inference)
        >>> client = get_client('groq', api_key='...', model_name='moonshotai/kimi-k2-instruct-0905')

        >>> # Mistral AI
        >>> client = get_client('mistral', api_key='...', model_name='mistral-large-latest')

        >>> # Fireworks AI
        >>> client = get_client('fireworks', api_key='...', model_name='accounts/fireworks/models/llama-v3p1-70b-instruct')

        >>> # Together AI
        >>> client = get_client('together', api_key='...', model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo')

        >>> # DeepSeek
        >>> client = get_client('deepseek', api_key='...', model_name='deepseek-chat')

        >>> # SambaNova Cloud (fast DeepSeek inference, 128K context)
        >>> client = get_client('sambanova', api_key='...', model_name='DeepSeek-V3.2')

        >>> # Nebius Token Factory (MiniMax, GLM-4.7, DeepSeek, Llama)
        >>> client = get_client('nebius', api_key='...', model_name='MiniMaxAI/MiniMax-M2.1')

        >>> # Cerebras (ultra-fast wafer-scale inference)
        >>> client = get_client('cerebras', api_key='...', model_name='zai-glm-4.7')

        >>> # Gemini
        >>> client = get_client('gemini', api_key='...', model_name='gemini-1.5-pro')

        >>> # LiteLLM (universal - supports 100+ providers)
        >>> client = get_client('litellm', model_name='gpt-4o')

        >>> # Auto (uses provider hierarchy — tries providers in order)
        >>> client = get_client('auto')  # tries sambanova → nebius → groq → ...
    """
    provider = provider.lower()

    if provider == "auto":
        return HierarchyClient(**kwargs)
    elif provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "openai_compatible":
        return GenericOpenAIClient(**kwargs)
    elif provider == "azure_openai":
        return AzureOpenAIClient(**kwargs)
    elif provider == "openrouter":
        return OpenRouterClient(**kwargs)
    elif provider == "zenmux":
        return ZenMuxClient(**kwargs)
    elif provider == "zai":
        return ZaiClient(**kwargs)
    elif provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "anthropic_compatible":
        return AnthropicCompatibleClient(**kwargs)
    elif provider == "gemini":
        return GeminiClient(**kwargs)
    elif provider == "groq":
        return GroqClient(**kwargs)
    elif provider == "mistral":
        return MistralClient(**kwargs)
    elif provider == "fireworks":
        return FireworksClient(**kwargs)
    elif provider == "together":
        return TogetherClient(**kwargs)
    elif provider == "deepseek":
        return DeepSeekClient(**kwargs)
    elif provider == "sambanova":
        return SambaNovaClient(**kwargs)
    elif provider == "nebius":
        return NebiusClient(**kwargs)
    elif provider == "cerebras":
        return CerebrasClient(**kwargs)
    elif provider == "ollama":
        return OllamaClient(**kwargs)
    elif provider == "litellm":
        return LiteLLMClient(**kwargs)
    else:
        # Try LiteLLM for unknown providers (supports 100+ providers)
        return LiteLLMClient(**kwargs)


__all__ = [
    "BaseLM",
    "get_client",
    "HierarchyClient",
    "get_hierarchy",
    "get_available_providers",
    "resolve_auto_provider",
    "DEFAULT_HIERARCHY",
    "PROVIDER_KEY_MAP",
    "OpenAIClient",
    "GenericOpenAIClient",
    "AzureOpenAIClient",
    "OpenRouterClient",
    "ZenMuxClient",
    "ZaiClient",
    "AnthropicClient",
    "AnthropicCompatibleClient",
    "GeminiClient",
    "LiteLLMClient",
    "GroqClient",
    "MistralClient",
    "FireworksClient",
    "TogetherClient",
    "DeepSeekClient",
    "SambaNovaClient",
    "NebiusClient",
    "CerebrasClient",
    "OllamaClient",
]
