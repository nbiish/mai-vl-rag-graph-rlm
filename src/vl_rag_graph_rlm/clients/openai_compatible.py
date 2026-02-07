"""OpenAI-compatible client (works with OpenAI, OpenRouter, ZenMux, z.ai)."""

import os
import time
from collections import defaultdict
from typing import Any

import openai
from dotenv import load_dotenv

from vl_rag_graph_rlm.clients.base import BaseLM
from vl_rag_graph_rlm.types import ModelUsageSummary, UsageSummary

load_dotenv()


class OpenAICompatibleClient(BaseLM):
    """
    Universal client for OpenAI-compatible APIs.
    Works with: OpenAI, OpenRouter, ZenMux, z.ai, and other OpenAI-compatible endpoints.
    """

    # Default base URLs for known providers
    DEFAULT_URLS = {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "zenmux": "https://zenmux.ai/api/v1",
        "zai": "https://api.z.ai/api/coding/paas/v4",
    }

    # Recommended cheap SOTA models by provider
    RECOMMENDED_MODELS = {
        "zenmux": {
            "default": "baidu/ernie-5.0-thinking",
            "coding": "bytedance/doubao-seed-1.8",
            "fast": "z-ai/glm-4.7-flash",
        },
        "openrouter": {
            "default": "minimax/minimax-m2.1",
            "nebius": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "coding": "z-ai/glm-4.7",
            "free": "solar-pro/solar-pro-3:free",
            "fast": "google/gemini-3-flash-preview",
        },
        "zai": {
            "default": "glm-4.7",
            "coding": "glm-4.7-coding",
            "fast": "glm-4.7-flash",
        },
        "openai": {
            "default": "gpt-4o-mini",
            "coding": "gpt-4o",
            "fast": "gpt-4o-mini",
        },
    }

    # Environment variable names for API keys
    API_KEY_ENV = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "zenmux": "ZENMUX_API_KEY",
        "zai": "ZAI_API_KEY",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        provider: str = "openai",
        **kwargs,
    ):
        """
        Initialize the client.

        Args:
            api_key: API key (optional, falls back to env var)
            model_name: Model name (e.g., "gpt-4o", "anthropic/claude-3.5-sonnet")
            base_url: Custom base URL (optional)
            provider: Provider name for defaults ("openai", "openrouter", "zenmux", "zai")
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(model_name=model_name, **kwargs)

        self.provider = provider.lower()

        # Resolve API key
        if api_key is None:
            env_var = self.API_KEY_ENV.get(self.provider, f"{self.provider.upper()}_API_KEY")
            api_key = os.getenv(env_var)

        # Resolve base URL
        if base_url is None:
            base_url = self.DEFAULT_URLS.get(self.provider, "https://api.openai.com/v1")

        self.base_url = base_url
        self.api_key = api_key

        if not self.api_key:
            raise ValueError(
                f"API key required for {self.provider}. "
                f"Set {self.API_KEY_ENV.get(self.provider, f'{self.provider.upper()}_API_KEY')} "
                f"environment variable or pass api_key explicitly."
            )

        # Initialize OpenAI clients
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, **kwargs)
        self.async_client = openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, **kwargs
        )

        # Usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self._last_usage: ModelUsageSummary | None = None

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make a synchronous completion call."""
        messages = self._prepare_messages(prompt)
        model = model or self.model_name

        if not model:
            raise ValueError("Model name is required")

        start_time = time.time()
        response = self.client.chat.completions.create(model=model, messages=messages)
        self._track_usage(response, model, time.time() - start_time)

        return response.choices[0].message.content

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None, **kwargs) -> str:
        """Make an asynchronous completion call."""
        messages = self._prepare_messages(prompt)
        model = model or self.model_name

        if not model:
            raise ValueError("Model name is required")

        start_time = time.time()
        response = await self.async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        self._track_usage(response, model, time.time() - start_time)

        return response.choices[0].message.content

    def _prepare_messages(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, str]]:
        """Convert prompt to OpenAI message format."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            return prompt  # type: ignore
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _track_usage(self, response: openai.ChatCompletion, model: str, execution_time: float):
        """Track usage statistics."""
        self.model_call_counts[model] += 1

        usage = getattr(response, "usage", None)
        if usage:
            self.model_input_tokens[model] += usage.prompt_tokens
            self.model_output_tokens[model] += usage.completion_tokens

            self._last_usage = ModelUsageSummary(
                total_calls=1,
                total_input_tokens=usage.prompt_tokens,
                total_output_tokens=usage.completion_tokens,
            )
        else:
            self._last_usage = ModelUsageSummary(
                total_calls=1, total_input_tokens=0, total_output_tokens=0
            )

    def get_usage_summary(self) -> UsageSummary:
        """Get aggregated usage summary."""
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        """Get usage for the last call."""
        if self._last_usage is None:
            return ModelUsageSummary(total_calls=0, total_input_tokens=0, total_output_tokens=0)
        return self._last_usage


class OpenAIClient(OpenAICompatibleClient):
    """Convenience class for OpenAI API."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            provider="openai",
            **kwargs,
        )


class OpenRouterClient(OpenAICompatibleClient):
    """Convenience class for OpenRouter API.
    
    Recommended cheap SOTA models:
        - minimax/minimax-m2.1: Minimax 2.1, excellent reasoning
        - kimi/kimi-k2.5: Excellent reasoning, very cheap
        - z-ai/glm-4.7: Great for coding
        - solar-pro/solar-pro-3:free: Free tier
        - google/gemini-3-flash-preview: Fast with 1M context
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        super().__init__(
            api_key=api_key,
            model_name=model_name or "minimax/minimax-m2.1",
            provider="openrouter",
            **kwargs,
        )


class ZenMuxClient(OpenAICompatibleClient):
    """Convenience class for ZenMux API.

    ZenMux is a unified API gateway supporting OpenAI and Anthropic protocols.
    Models use provider/model-name format (e.g., "moonshotai/kimi-k2.5").

    OpenAI protocol endpoint: https://zenmux.ai/api/v1
    Anthropic protocol endpoint: https://zenmux.ai/api/anthropic

    Recommended models (OpenAI protocol):
        - moonshotai/kimi-k2.5: Excellent reasoning, very capable
        - baidu/ernie-5.0-thinking: Best for reasoning
        - bytedance/doubao-seed-1.8: Best for coding
        - z-ai/glm-4.7-flash: Fast, cheap responses

    Docs: https://docs.zenmux.ai/guide/quickstart.html
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        super().__init__(
            api_key=api_key,
            model_name=model_name or "moonshotai/kimi-k2.5",
            provider="zenmux",
            **kwargs,
        )


class ZaiClient(OpenAICompatibleClient):
    """Client for z.ai API (Zhipu AI) with Coding Plan priority.

    z.ai offers two API endpoints:
        1. Coding Plan (flat-rate $3-15/mo): https://api.z.ai/api/coding/paas/v4
        2. Normal (pay-per-token):           https://open.bigmodel.cn/api/paas/v4

    By default, the Coding Plan endpoint is tried first. If it fails (e.g.,
    no active subscription), the client automatically falls back to the normal
    endpoint. Set ZAI_CODING_PLAN=false to skip the Coding Plan endpoint.

    Coding Plan models:
        - glm-4.7: Flagship model, complex tasks (recommended)
        - glm-4.5-air: Lightweight, faster responses

    Normal API models:
        - glm-4.7: Flagship model, excellent reasoning
        - glm-4.7-flash: Fast, cost-effective

    Docs: https://docs.z.ai/devpack/tool/others
    """

    ZAI_CODING_PLAN_URL = "https://api.z.ai/api/coding/paas/v4"
    ZAI_NORMAL_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        # Determine if Coding Plan should be tried first
        self._use_coding_plan = os.getenv("ZAI_CODING_PLAN", "true").lower() in ("true", "1", "yes")
        self._coding_plan_failed = False

        # Start with Coding Plan URL if enabled, otherwise normal
        base_url = self.ZAI_CODING_PLAN_URL if self._use_coding_plan else self.ZAI_NORMAL_URL

        super().__init__(
            api_key=api_key,
            model_name=model_name or "glm-4.7",
            base_url=base_url,
            provider="zai",
            **kwargs,
        )

        # Pre-create fallback client for normal endpoint (lazy — only used on failure)
        self._fallback_client: openai.OpenAI | None = None
        self._fallback_async_client: openai.AsyncOpenAI | None = None

    def _get_fallback_client(self) -> openai.OpenAI:
        """Lazily create fallback client pointing to normal z.ai endpoint."""
        if self._fallback_client is None:
            self._fallback_client = openai.OpenAI(
                api_key=self.api_key, base_url=self.ZAI_NORMAL_URL
            )
        return self._fallback_client

    def _get_fallback_async_client(self) -> openai.AsyncOpenAI:
        """Lazily create async fallback client pointing to normal z.ai endpoint."""
        if self._fallback_async_client is None:
            self._fallback_async_client = openai.AsyncOpenAI(
                api_key=self.api_key, base_url=self.ZAI_NORMAL_URL
            )
        return self._fallback_async_client

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make a completion call, trying Coding Plan first then falling back."""
        messages = self._prepare_messages(prompt)
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required")

        start_time = time.time()

        # Try primary endpoint (Coding Plan if enabled)
        if self._use_coding_plan and not self._coding_plan_failed:
            try:
                response = self.client.chat.completions.create(model=model, messages=messages)
                self._track_usage(response, model, time.time() - start_time)
                return response.choices[0].message.content
            except Exception:
                self._coding_plan_failed = True
                # Fall through to normal endpoint

        # Fallback to normal endpoint
        fallback = self._get_fallback_client() if self._use_coding_plan else self.client
        response = fallback.chat.completions.create(model=model, messages=messages)
        self._track_usage(response, model, time.time() - start_time)
        return response.choices[0].message.content

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None, **kwargs) -> str:
        """Make an async completion call, trying Coding Plan first then falling back."""
        messages = self._prepare_messages(prompt)
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required")

        start_time = time.time()

        # Try primary endpoint (Coding Plan if enabled)
        if self._use_coding_plan and not self._coding_plan_failed:
            try:
                response = await self.async_client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
                self._track_usage(response, model, time.time() - start_time)
                return response.choices[0].message.content
            except Exception:
                self._coding_plan_failed = True

        # Fallback to normal endpoint
        fallback = self._get_fallback_async_client() if self._use_coding_plan else self.async_client
        response = await fallback.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        self._track_usage(response, model, time.time() - start_time)
        return response.choices[0].message.content


class GenericOpenAIClient(OpenAICompatibleClient):
    """
    Generic client for any OpenAI-compatible API endpoint.

    Use this for custom providers or self-hosted models that implement
    the OpenAI API specification.

    Args:
        api_key: API key for the provider
        model_name: Model name to use
        base_url: Full base URL for the API (e.g., "https://api.groq.com/openai/v1")

    Examples:
        >>> # Generic OpenAI-compatible endpoint
        >>> client = GenericOpenAIClient(
        ...     api_key="your-key",
        ...     model_name="llama-3.1-70b",
        ...     base_url="https://api.example.com/v1"
        ... )
    """

    DEFAULT_URLS = {
        "openai_compatible": "https://api.openai.com/v1",  # Placeholder, should be overridden
    }

    API_KEY_ENV = {
        "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        **kwargs
    ):
        # Resolve API key from env if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")

        # Resolve base URL from env if not provided
        if base_url is None:
            base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")

        if not base_url:
            raise ValueError(
                "base_url is required for GenericOpenAIClient. "
                "Pass it explicitly or set OPENAI_COMPATIBLE_BASE_URL environment variable."
            )

        # Get model from env if not provided
        if model_name is None:
            model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

        if not model_name:
            raise ValueError(
                "model_name is required for GenericOpenAIClient. "
                "Pass it explicitly or set OPENAI_COMPATIBLE_MODEL environment variable."
            )

        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            provider="openai_compatible",
            **kwargs,
        )


class AzureOpenAIClient(OpenAICompatibleClient):
    """
    Client for Azure OpenAI Service.

    Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables,
    or explicit api_key and base_url parameters.

    Examples:
        >>> # Using environment variables
        >>> client = AzureOpenAIClient(model_name="gpt-4o")

        >>> # Explicit configuration
        >>> client = AzureOpenAIClient(
        ...     api_key="your-azure-key",
        ...     base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
        ...     model_name="gpt-4o"
        ... )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        api_version: str = "2024-02-01",
        **kwargs
    ):
        # Resolve Azure credentials
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = base_url or os.getenv("AZURE_OPENAI_ENDPOINT")

        if not api_key:
            raise ValueError(
                "Azure OpenAI API key required. "
                "Set AZURE_OPENAI_API_KEY environment variable or pass api_key explicitly."
            )

        if not endpoint:
            raise ValueError(
                "Azure OpenAI endpoint required. "
                "Set AZURE_OPENAI_ENDPOINT environment variable or pass base_url explicitly."
            )

        # Construct full base URL with API version
        # Endpoint should be like: https://your-resource.openai.azure.com/
        # We need: https://your-resource.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={version}
        base_url = endpoint.rstrip("/")
        if "/openai/deployments/" not in base_url:
            # Assume endpoint is just the resource URL
            if model_name:
                base_url = f"{base_url}/openai/deployments/{model_name}"

        # Add API version as query param to base_url
        if "?" not in base_url:
            base_url = f"{base_url}?api-version={api_version}"

        super().__init__(
            api_key=api_key,
            model_name=model_name or "gpt-4o",
            base_url=base_url,
            provider="azure_openai",
            **kwargs,
        )


# ============================================================
# Popular OpenAI-Compatible Provider Clients
# ============================================================

class GroqClient(OpenAICompatibleClient):
    """
    Client for Groq API (ultra-fast inference).

    Recommended models:
        - llama-3.3-70b-versatile: Latest Llama 3.3 70B (recommended)
        - llama-3.1-8b: Fast, cost-effective
        - mixtral-8x7b-32768: Large context

    Get API key: https://console.groq.com
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        super().__init__(
            api_key=api_key,
            model_name=model_name or "llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            provider="groq",
            **kwargs,
        )


class MistralClient(OpenAICompatibleClient):
    """
    Client for Mistral AI API.

    Recommended models:
        - mistral-large-latest: Most capable
        - mistral-medium: Balanced
        - mistral-small: Fast, cheap

    Get API key: https://console.mistral.ai
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        super().__init__(
            api_key=api_key,
            model_name=model_name or "mistral-large-latest",
            base_url="https://api.mistral.ai/v1",
            provider="mistral",
            **kwargs,
        )


class FireworksClient(OpenAICompatibleClient):
    """
    Client for Fireworks AI API.

    Recommended models:
        - accounts/fireworks/models/llama-v3p1-70b-instruct: Llama 3.1 70B
        - accounts/fireworks/models/mixtral-8x22b-instruct: Mixtral 8x22B

    Get API key: https://fireworks.ai
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        super().__init__(
            api_key=api_key,
            model_name=model_name or "accounts/fireworks/models/llama-v3p1-70b-instruct",
            base_url="https://api.fireworks.ai/inference/v1",
            provider="fireworks",
            **kwargs,
        )


class TogetherClient(OpenAICompatibleClient):
    """
    Client for Together AI API.

    Recommended models:
        - meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo: Fast Llama 3.1
        - mistralai/Mixtral-8x22B-Instruct-v0.1: Mixtral 8x22B

    Get API key: https://api.together.ai
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        super().__init__(
            api_key=api_key,
            model_name=model_name or "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            base_url="https://api.together.xyz/v1",
            provider="together",
            **kwargs,
        )


class DeepSeekClient(OpenAICompatibleClient):
    """
    Client for DeepSeek API.

    Recommended models:
        - deepseek-chat: DeepSeek-V3 general purpose
        - deepseek-reasoner: DeepSeek-R1 reasoning model

    Get API key: https://platform.deepseek.com
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        super().__init__(
            api_key=api_key,
            model_name=model_name or "deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            provider="deepseek",
            **kwargs,
        )


class SambaNovaClient(OpenAICompatibleClient):
    """
    Client for SambaNova Cloud API.

    Production models (128K context):
        - DeepSeek-V3.2: Latest DeepSeek V3 (200+ tok/sec)
        - DeepSeek-V3-0324: DeepSeek V3 March 2024 (250+ tok/sec)
        - DeepSeek-R1-0528: Reasoning model
        - DeepSeek-R1-Distill-Llama-70B: Distilled reasoning

    Rate limits (Free tier): 20 RPM, 40 RPD, 200K TPD
    Rate limits (Developer tier): 60 RPM, 12K RPD

    SambaNova Cloud provides fastest inference for open source models
    with OpenAI-compatible API.

    Get API key: https://cloud.sambanova.ai
    Docs: https://docs.sambanova.ai/cloud/docs/get-started/supported-models
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        api_key = api_key or os.getenv("SAMBANOVA_API_KEY")
        super().__init__(
            api_key=api_key,
            model_name=model_name or "DeepSeek-V3.2",
            base_url="https://api.sambanova.ai/v1",
            provider="sambanova",
            **kwargs,
        )


class NebiusClient(OpenAICompatibleClient):
    """
    Client for Nebius Token Factory API.

    Recommended models:
        - MiniMaxAI/MiniMax-M2.1: MiniMax M2.1 (default)
        - z-ai/GLM-4.7: Z.AI's flagship for agentic coding and reasoning
        - deepseek-ai/DeepSeek-R1-0528: DeepSeek R1 reasoning
        - meta-llama/Meta-Llama-3.1-70B-Instruct: Llama 3.1 70B

    Nebius Token Factory provides OpenAI-compatible access to various
    models including MiniMax, GLM-4.7, DeepSeek, and Llama series.

    Get API key: https://tokenfactory.nebius.com
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        api_key = api_key or os.getenv("NEBIUS_API_KEY")
        super().__init__(
            api_key=api_key,
            model_name=model_name or "MiniMaxAI/MiniMax-M2.1",
            base_url="https://api.tokenfactory.nebius.com/v1",
            provider="nebius",
            **kwargs,
        )


class CerebrasClient(OpenAICompatibleClient):
    """
    Client for Cerebras Inference API.

    Recommended models:
        - llama-3.3-70b: Llama 3.3 70B (recommended production model)
        - llama3.1-8b: Llama 3.1 8B (fast, cost-effective)
        - qwen-3-32b: Qwen 3 32B
        - zai-glm-4.7: Z.AI GLM 4.7 (preview)

    Cerebras provides ultra-fast inference on custom wafer-scale hardware
    with OpenAI-compatible API.

    Get API key: https://cloud.cerebras.ai
    Docs: https://inference-docs.cerebras.ai
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        super().__init__(
            api_key=api_key,
            model_name=model_name or "llama-3.3-70b",
            base_url="https://api.cerebras.ai/v1",
            provider="cerebras",
            **kwargs,
        )
