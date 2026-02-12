"""Anthropic Claude client."""

import logging
import os
import time
from collections import defaultdict
from typing import Any

import anthropic
from dotenv import load_dotenv

from vl_rag_graph_rlm.clients.base import BaseLM
from vl_rag_graph_rlm.types import ModelUsageSummary, UsageSummary

logger = logging.getLogger(__name__)

load_dotenv()


class AnthropicClient(BaseLM):
    """Client for Anthropic Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "claude-3-5-sonnet-20241022",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key explicitly."
            )

        self._anthropic_kwargs = kwargs
        self.client = anthropic.Anthropic(api_key=self.api_key, **kwargs)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key, **kwargs)

        # Resolve fallback API key: ANTHROPIC_API_KEY_FALLBACK env var.
        self._fallback_api_key: str | None = os.getenv("ANTHROPIC_API_KEY_FALLBACK")
        self._fallback_key_client: anthropic.Anthropic | None = None
        self._fallback_key_async_client: anthropic.AsyncAnthropic | None = None
        self._using_fallback_key = False

        # Usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self._last_usage: ModelUsageSummary | None = None

    def _get_fallback_key_client(self) -> anthropic.Anthropic:
        """Lazily create sync client using the fallback API key."""
        if self._fallback_key_client is None:
            self._fallback_key_client = anthropic.Anthropic(
                api_key=self._fallback_api_key, **self._anthropic_kwargs
            )
        return self._fallback_key_client

    def _get_fallback_key_async_client(self) -> anthropic.AsyncAnthropic:
        """Lazily create async client using the fallback API key."""
        if self._fallback_key_async_client is None:
            self._fallback_key_async_client = anthropic.AsyncAnthropic(
                api_key=self._fallback_api_key, **self._anthropic_kwargs
            )
        return self._fallback_key_async_client

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make a synchronous completion call with fallback key retry."""
        model = model or self.model_name
        messages = self._prepare_messages(prompt)

        start_time = time.time()
        try:
            response = self.client.messages.create(
                model=model,
                messages=messages,
                max_tokens=4096,
            )
            self._track_usage(response, model, time.time() - start_time)
            return response.content[0].text
        except Exception as primary_err:
            if not self._fallback_api_key or self._using_fallback_key:
                raise
            logger.warning(
                "anthropic: primary key failed (%s: %s), retrying with fallback key",
                type(primary_err).__name__, str(primary_err)[:120],
            )
            self._using_fallback_key = True
            fb_client = self._get_fallback_key_client()
            start_time = time.time()
            response = fb_client.messages.create(
                model=model,
                messages=messages,
                max_tokens=4096,
            )
            self._track_usage(response, model, time.time() - start_time)
            self.client = fb_client
            self.api_key = self._fallback_api_key
            return response.content[0].text

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make an asynchronous completion call with fallback key retry."""
        model = model or self.model_name
        messages = self._prepare_messages(prompt)

        start_time = time.time()
        try:
            response = await self.async_client.messages.create(
                model=model,
                messages=messages,
                max_tokens=4096,
            )
            self._track_usage(response, model, time.time() - start_time)
            return response.content[0].text
        except Exception as primary_err:
            if not self._fallback_api_key or self._using_fallback_key:
                raise
            logger.warning(
                "anthropic: primary key failed (%s: %s), retrying with fallback key (async)",
                type(primary_err).__name__, str(primary_err)[:120],
            )
            self._using_fallback_key = True
            fb_client = self._get_fallback_key_async_client()
            start_time = time.time()
            response = await fb_client.messages.create(
                model=model,
                messages=messages,
                max_tokens=4096,
            )
            self._track_usage(response, model, time.time() - start_time)
            self.async_client = fb_client
            self.api_key = self._fallback_api_key
            return response.content[0].text

    def _prepare_messages(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, str]]:
        """Convert prompt to Anthropic message format."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # Convert OpenAI format to Anthropic format
            anthropic_messages = []
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Anthropic doesn't use 'system' in messages
                if role == "system":
                    anthropic_messages.append({"role": "user", "content": f"System: {content}"})
                else:
                    anthropic_messages.append({"role": role, "content": content})
            return anthropic_messages
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _track_usage(self, response: Any, model: str, execution_time: float):
        """Track usage statistics."""
        self.model_call_counts[model] += 1

        if hasattr(response, "usage"):
            self.model_input_tokens[model] += response.usage.input_tokens
            self.model_output_tokens[model] += response.usage.output_tokens

            self._last_usage = ModelUsageSummary(
                total_calls=1,
                total_input_tokens=response.usage.input_tokens,
                total_output_tokens=response.usage.output_tokens,
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
        if self._last_usage is None:
            return ModelUsageSummary(total_calls=0, total_input_tokens=0, total_output_tokens=0)
        return self._last_usage


class AnthropicCompatibleClient(AnthropicClient):
    """
    Client for generic Anthropic-compatible APIs.

    Use this for custom providers or proxies that implement
    the Anthropic API specification with a custom base URL.

    Args:
        api_key: API key for the provider
        model_name: Model name to use
        base_url: Full base URL for the API

    Examples:
        >>> # Generic Anthropic-compatible endpoint
        >>> client = AnthropicCompatibleClient(
        ...     api_key="your-key",
        ...     model_name="claude-3-5-sonnet",
        ...     base_url="https://api.example.com/v1"
        ... )

        >>> # Using environment variables
        >>> # Set ANTHROPIC_COMPATIBLE_API_KEY, ANTHROPIC_COMPATIBLE_BASE_URL, ANTHROPIC_COMPATIBLE_MODEL
        >>> client = AnthropicCompatibleClient()
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        # Resolve from environment if not provided
        api_key = api_key or os.getenv("ANTHROPIC_COMPATIBLE_API_KEY")
        base_url = base_url or os.getenv("ANTHROPIC_COMPATIBLE_BASE_URL")
        model_name = model_name or os.getenv("ANTHROPIC_COMPATIBLE_MODEL")

        if not api_key:
            raise ValueError(
                "API key required for AnthropicCompatibleClient. "
                "Set ANTHROPIC_COMPATIBLE_API_KEY environment variable or pass api_key explicitly."
            )

        if not base_url:
            raise ValueError(
                "base_url is required for AnthropicCompatibleClient. "
                "Set ANTHROPIC_COMPATIBLE_BASE_URL environment variable or pass base_url explicitly."
            )

        if not model_name:
            raise ValueError(
                "model_name is required for AnthropicCompatibleClient. "
                "Set ANTHROPIC_COMPATIBLE_MODEL environment variable or pass model_name explicitly."
            )

        # Store values before calling parent init
        self.api_key = api_key
        self.base_url = base_url

        # Initialize the base class
        super().__init__(model_name=model_name, **kwargs)

        # Re-initialize Anthropic clients with custom base URL
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs
        )
        self.async_client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs
        )

        # Usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self._last_usage: ModelUsageSummary | None = None
