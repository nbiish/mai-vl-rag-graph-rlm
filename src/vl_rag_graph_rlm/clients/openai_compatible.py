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
        "zenmux": "https://api.zenmux.ai/v1",
        "zai": "https://open.bigmodel.cn/api/paas/v4",
    }

    # Recommended cheap SOTA models by provider
    RECOMMENDED_MODELS = {
        "zenmux": {
            "default": "ernie-5.0-thinking-preview",
            "coding": "dubao-seed-1.8",
            "fast": "glm-4.7-flash",
        },
        "openrouter": {
            "default": "kimi/kimi-k2.5",
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

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make an asynchronous completion call."""
        messages = self._prepare_messages(prompt)
        model = model or self.model_name

        if not model:
            raise ValueError("Model name is required")

        start_time = time.time()
        response = await self.async_client.chat.completions.create(
            model=model, messages=messages
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
        - kimi/kimi-k2.5: Excellent reasoning, very cheap
        - z-ai/glm-4.7: Great for coding
        - solar-pro/solar-pro-3:free: Free tier
        - google/gemini-3-flash-preview: Fast with 1M context
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        super().__init__(
            api_key=api_key,
            model_name=model_name or "kimi/kimi-k2.5",
            provider="openrouter",
            **kwargs,
        )


class ZenMuxClient(OpenAICompatibleClient):
    """Convenience class for ZenMux API.
    
    Recommended models:
        - ernie-5.0-thinking-preview: Best for reasoning
        - dubao-seed-1.8: Best for coding
        - glm-4.7-flash: Fast, cheap responses
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        super().__init__(
            api_key=api_key,
            model_name=model_name or "ernie-5.0-thinking-preview",
            provider="zenmux",
            **kwargs,
        )


class ZaiClient(OpenAICompatibleClient):
    """Convenience class for z.ai API (Zhipu AI).
    
    Recommended models:
        - glm-4.7: Flagship model, excellent reasoning
        - glm-4.7-coding: Optimized for code generation
        - glm-4.7-flash: Fast, cost-effective
        
    Note: z.ai also offers flat-rate Coding Plans ($3-15/mo) via their native API.
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None, **kwargs):
        super().__init__(
            api_key=api_key,
            model_name=model_name or "glm-4.7",
            provider="zai",
            **kwargs,
        )
