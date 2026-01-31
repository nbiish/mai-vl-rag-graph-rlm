"""Google Gemini client."""

import os
import time
from collections import defaultdict
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

from vl_rag_graph_rlm.clients.base import BaseLM
from vl_rag_graph_rlm.types import ModelUsageSummary, UsageSummary

load_dotenv()


class GeminiClient(BaseLM):
    """Client for Google Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-1.5-pro",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable or pass api_key explicitly."
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

        # Usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self._last_usage: ModelUsageSummary | None = None

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make a synchronous completion call."""
        content = self._prepare_content(prompt)

        start_time = time.time()
        response = self.model.generate_content(content)
        self._track_usage(response, model or self.model_name, time.time() - start_time)

        return response.text

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make an asynchronous completion call."""
        content = self._prepare_content(prompt)

        start_time = time.time()
        response = await self.model.generate_content_async(content)
        self._track_usage(response, model or self.model_name, time.time() - start_time)

        return response.text

    def _prepare_content(self, prompt: str | list[dict[str, Any]]) -> str:
        """Convert prompt to Gemini content format."""
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, list):
            # Convert message list to single string
            parts = []
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            return "\n".join(parts)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _track_usage(self, response: Any, model: str, execution_time: float):
        """Track usage statistics."""
        self.model_call_counts[model] += 1

        # Gemini doesn't always provide token counts
        if hasattr(response, "usage_metadata"):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
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
