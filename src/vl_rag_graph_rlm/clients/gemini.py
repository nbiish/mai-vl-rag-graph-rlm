"""Google Gemini client — uses the google-genai SDK (replaces deprecated google-generativeai)."""

import os
import time
from collections import defaultdict
from typing import Any

from google import genai
from dotenv import load_dotenv

from vl_rag_graph_rlm.clients.base import BaseLM
from vl_rag_graph_rlm.types import ModelUsageSummary, UsageSummary

load_dotenv()


class GeminiClient(BaseLM):
    """Client for Google Gemini API (google-genai SDK)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY "
                "environment variable or pass api_key explicitly."
            )

        self.client = genai.Client(api_key=self.api_key)

        # Usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self._last_usage: ModelUsageSummary | None = None

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make a synchronous completion call."""
        content = self._prepare_content(prompt)
        effective_model = model or self.model_name

        start_time = time.time()
        response = self.client.models.generate_content(
            model=effective_model,
            contents=content,
        )
        self._track_usage(response, effective_model, time.time() - start_time)

        return response.text

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make an asynchronous completion call."""
        content = self._prepare_content(prompt)
        effective_model = model or self.model_name

        start_time = time.time()
        response = await self.client.aio.models.generate_content(
            model=effective_model,
            contents=content,
        )
        self._track_usage(response, effective_model, time.time() - start_time)

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

        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            um = response.usage_metadata
            input_tokens = getattr(um, "prompt_token_count", 0) or 0
            output_tokens = getattr(um, "candidates_token_count", 0) or 0
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
