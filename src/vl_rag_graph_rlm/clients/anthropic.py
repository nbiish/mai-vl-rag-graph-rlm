"""Anthropic Claude client."""

import os
import time
from collections import defaultdict
from typing import Any

import anthropic
from dotenv import load_dotenv

from vl_rag_graph_rlm.clients.base import BaseLM
from vl_rag_graph_rlm.types import ModelUsageSummary, UsageSummary

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

        self.client = anthropic.Anthropic(api_key=self.api_key, **kwargs)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key, **kwargs)

        # Usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self._last_usage: ModelUsageSummary | None = None

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make a synchronous completion call."""
        model = model or self.model_name
        messages = self._prepare_messages(prompt)

        start_time = time.time()
        response = self.client.messages.create(
            model=model,
            messages=messages,
            max_tokens=4096,
        )
        self._track_usage(response, model, time.time() - start_time)

        return response.content[0].text

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make an asynchronous completion call."""
        model = model or self.model_name
        messages = self._prepare_messages(prompt)

        start_time = time.time()
        response = await self.async_client.messages.create(
            model=model,
            messages=messages,
            max_tokens=4096,
        )
        self._track_usage(response, model, time.time() - start_time)

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
