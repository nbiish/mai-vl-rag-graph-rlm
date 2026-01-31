"""LiteLLM universal client - supports 100+ providers."""

import os
import time
from collections import defaultdict
from typing import Any

import litellm
from dotenv import load_dotenv

from vl_rag_graph_rlm.clients.base import BaseLM
from vl_rag_graph_rlm.types import ModelUsageSummary, UsageSummary

load_dotenv()


class LiteLLMClient(BaseLM):
    """
    Universal client using LiteLLM.
    Supports 100+ providers: OpenAI, Anthropic, Gemini, Azure, Ollama, etc.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gpt-4o",
        api_base: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.extra_kwargs = kwargs

        # Usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self._last_usage: ModelUsageSummary | None = None

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make a synchronous completion call."""
        model = model or self.model_name
        messages = self._prepare_messages(prompt)

        call_kwargs = {
            "model": model,
            "messages": messages,
        }

        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.api_base:
            call_kwargs["api_base"] = self.api_base

        call_kwargs.update(self.extra_kwargs)

        start_time = time.time()
        response = litellm.completion(**call_kwargs)
        self._track_usage(response, model, time.time() - start_time)

        return response.choices[0].message.content

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make an asynchronous completion call."""
        model = model or self.model_name
        messages = self._prepare_messages(prompt)

        call_kwargs = {
            "model": model,
            "messages": messages,
        }

        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.api_base:
            call_kwargs["api_base"] = self.api_base

        call_kwargs.update(self.extra_kwargs)

        start_time = time.time()
        response = await litellm.acompletion(**call_kwargs)
        self._track_usage(response, model, time.time() - start_time)

        return response.choices[0].message.content

    def _prepare_messages(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, str]]:
        """Convert prompt to message format."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            return prompt  # type: ignore
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _track_usage(self, response: Any, model: str, execution_time: float):
        """Track usage statistics."""
        self.model_call_counts[model] += 1

        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)

            self.model_input_tokens[model] += input_tokens
            self.model_output_tokens[model] += output_tokens

            self._last_usage = ModelUsageSummary(
                total_calls=1,
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
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
