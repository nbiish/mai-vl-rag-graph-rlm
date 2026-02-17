"""vLLM client for high-throughput self-hosted LLM inference.

Provides OpenAI-compatible API access to vLLM servers for
high-performance local or self-hosted LLM inference.
"""

import logging
import os
from typing import Any, Optional

import openai

from vl_rag_graph_rlm.clients.base import BaseLM
from vl_rag_graph_rlm.types import ModelUsageSummary, UsageSummary

logger = logging.getLogger(__name__)


class VLLMClient(BaseLM):
    """Client for vLLM high-throughput inference server.
    
    vLLM provides high-performance LLM serving with features like:
    - PagedAttention for efficient memory management
    - Continuous batching for high throughput
    - Tensor parallelism for multi-GPU serving
    
    Example:
        >>> from vl_rag_graph_rlm.clients.vllm import VLLMClient
        >>> client = VLLMClient(base_url="http://localhost:8000/v1")
        >>> response = client.completion("Explain quantum computing")
    
    References:
        - vLLM: https://github.com/vllm-project/vllm
        - OpenAI-compatible API: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """
    
    def __init__(
        self,
        api_key: str = "not-needed",  # vLLM doesn't require API keys but OpenAI client expects one
        model_name: str = "default",
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.timeout = timeout
        
        # Initialize OpenAI-compatible client
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
        )
        
        # Usage tracking
        self._call_count = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._last_usage: Optional[ModelUsageSummary] = None
        
        logger.info(f"vLLM client initialized: {self.base_url}")
    
    def completion(self, prompt: str | list[dict[str, Any]], model: Optional[str] = None) -> str:
        """Make a synchronous completion call to vLLM server."""
        effective_model = model or self.model_name
        
        # Build messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        
        try:
            response = self.client.chat.completions.create(
                model=effective_model,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
            )
            
            self._call_count += 1
            
            # Track usage
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            else:
                # Fallback: estimate tokens
                input_tokens = sum(len(m["content"]) // 4 for m in messages)
                output_tokens = len(response.choices[0].message.content) // 4
            
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            
            self._last_usage = ModelUsageSummary(
                total_calls=1,
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
            )
            
            return response.choices[0].message.content
            
        except openai.APIConnectionError as e:
            logger.error(f"Cannot connect to vLLM at {self.base_url}. Is the server running?")
            raise ConnectionError(
                f"Cannot connect to vLLM at {self.base_url}. "
                "Make sure vLLM server is running. "
                "Start with: python -m vllm.entrypoints.openai.api_server --model <model>"
            ) from e
        except Exception as e:
            logger.error(f"vLLM API error: {e}")
            raise
    
    async def acompletion(self, prompt: str | list[dict[str, Any]], model: Optional[str] = None) -> str:
        """Make an asynchronous completion call."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.completion, prompt, model)
    
    def get_usage_summary(self) -> UsageSummary:
        """Get aggregated usage summary."""
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self._call_count,
                    total_input_tokens=self._input_tokens,
                    total_output_tokens=self._output_tokens,
                )
            }
        )
    
    def get_last_usage(self) -> ModelUsageSummary:
        if self._last_usage is None:
            return ModelUsageSummary(total_calls=0, total_input_tokens=0, total_output_tokens=0)
        return self._last_usage
    
    @staticmethod
    def check_server(base_url: Optional[str] = None) -> bool:
        """Check if vLLM server is running."""
        import requests
        
        url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        health_url = url.replace("/v1", "/health")
        
        try:
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def list_models(base_url: Optional[str] = None) -> list[str]:
        """List available models from vLLM server."""
        import requests
        
        url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        
        try:
            response = requests.get(f"{url}/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to list vLLM models: {e}")
            return []
