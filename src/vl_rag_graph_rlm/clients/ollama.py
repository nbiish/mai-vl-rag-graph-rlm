"""Ollama provider for local LLM inference OR Claude API mode.

Supports TWO modes of operation:
1. LOCAL MODE: Uses local Ollama installation (http://localhost:11434)
2. API MODE: Uses Claude models via Ollama's API compatibility layer

Set OLLAMA_MODE=api to enable API mode (requires OLLAMA_API_KEY).
"""

import os
import time
from collections import defaultdict
from typing import Any, Optional

from vl_rag_graph_rlm.clients.base import BaseLM
from vl_rag_graph_rlm.types import ModelUsageSummary, UsageSummary


class OllamaClient(BaseLM):
    """Client for Ollama - supports both local inference and Claude API mode.
    
    LOCAL MODE (default):
    - Requires Ollama running locally (default: http://localhost:11434)
    - Install from: https://ollama.com
    - No API keys needed
    
    API MODE (Claude):
    - Set OLLAMA_MODE=api and provide OLLAMA_API_KEY
    - Uses Claude models through Ollama interface
    - Acts like an API provider (not local inference)
    
    Typical local models:
    - llama3.2 (3B, fast)
    - llama3.1 (8B, good balance)
    - mistral (7B)
    - qwen2.5 (various sizes)
    - deepseek-r1 (reasoning model)
    """
    
    # Local fallback chain: try smaller models if larger fail
    FALLBACK_MODELS = [
        "llama3.2",
        "llama3.1",
        "mistral",
        "qwen2.5:7b",
    ]
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_depth: int = 3,
        max_iterations: int = 10,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        """Initialize Ollama client.
        
        Args:
            model_name: Model name (default: from OLLAMA_MODEL env var, or llama3.2)
            api_key: API key (used in API mode, from OLLAMA_API_KEY env var)
            temperature: Sampling temperature
            max_depth: RLM max recursion depth
            max_iterations: RLM max iterations
            api_base: Ollama API URL (default: from OLLAMA_BASE_URL env var)
        """
        super().__init__(model_name=model_name or os.getenv("OLLAMA_MODEL", "llama3.2"), **kwargs)
        
        self.mode = os.getenv("OLLAMA_MODE", "local").lower()
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.temperature = temperature
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.api_base = api_base or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY", "")
        
        # Usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self._last_usage: ModelUsageSummary | None = None
    
    def _check_connection(self) -> None:
        """Check if Ollama is running (local mode only)."""
        if self.mode == "api":
            return  # API mode doesn't need local connection check
            
        import urllib.request
        import urllib.error
        
        try:
            req = urllib.request.Request(
                f"{self.api_base}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama returned status {response.status}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.api_base}. "
                f"Is Ollama running? Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama connection check failed: {e}")
    
    def _raw_completion(
        self,
        prompt: str | list[dict[str, Any]],
        model: Optional[str] = None,
    ) -> str:
        """Make raw completion request to Ollama API (local mode)."""
        import json
        import urllib.request
        import urllib.error
        
        use_model = model or self.model_name
        
        # Build prompt from messages or string
        if isinstance(prompt, list):
            # Convert message list to single prompt
            full_prompt = "\n".join(
                f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
                for m in prompt
            )
        else:
            full_prompt = prompt
        
        # Prepare request
        payload = {
            "model": use_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            }
        }
        
        data = json.dumps(payload).encode('utf-8')
        
        req = urllib.request.Request(
            f"{self.api_base}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            start_time = time.time()
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                response_text = result.get("response", "")
                
                # Track usage (Ollama doesn't return token counts, estimate)
                duration = time.time() - start_time
                self._track_usage(use_model, full_prompt, response_text, duration)
                
                return response_text
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise RuntimeError(f"Ollama API error: {e.code} - {error_body}")
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")
    
    def _api_completion(
        self,
        prompt: str | list[dict[str, Any]],
        model: Optional[str] = None,
    ) -> str:
        """Make completion request using Anthropic SDK via Ollama's Anthropic compatibility layer.
        
        Ollama provides Anthropic API compatibility - it translates Anthropic API calls
        to local/cloud models. The API key is required but ignored.
        
        Reference: https://docs.ollama.com/api/anthropic-compatibility
        """
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "Anthropic SDK required for Ollama API mode. "
                "Install with: pip install anthropic"
            )
        
        use_model = model or self.model_name
        
        # Build messages format
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Initialize Anthropic client pointing to Ollama's Anthropic-compatible endpoint
        # API key is required but ignored by Ollama
        client = anthropic.Anthropic(
            base_url=self.api_base,  # e.g., http://localhost:11434
            api_key=self.api_key or "ollama",  # Required but ignored
            timeout=120.0,
        )
        
        try:
            start_time = time.time()
            response = client.messages.create(
                model=use_model,
                max_tokens=4096,
                temperature=self.temperature,
                messages=messages,
            )
            
            # Extract text from response
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text += block.text
            
            # Track usage
            duration = time.time() - start_time
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            
            self.model_call_counts[use_model] += 1
            self.model_input_tokens[use_model] += input_tokens
            self.model_output_tokens[use_model] += output_tokens
            
            self._last_usage = ModelUsageSummary(
                total_calls=1,
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
            )
            
            return response_text
            
        except Exception as e:
            raise RuntimeError(f"Ollama Anthropic API call failed: {e}")
    
    def _track_usage(self, model: str, prompt: str, response: str, duration: float) -> None:
        """Track usage statistics (local mode)."""
        # Estimate tokens (rough approximation: 4 chars ≈ 1 token)
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4
        
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += input_tokens
        self.model_output_tokens[model] += output_tokens
        
        self._last_usage = ModelUsageSummary(
            model=model,
            calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration=duration,
            errors=0,
        )
    
    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Generate completion with mode-specific handling."""
        if self.mode == "api":
            # API mode: use Claude API
            try:
                return self._api_completion(prompt, model)
            except RuntimeError:
                raise RuntimeError("Claude API call failed")
        else:
            # Local mode: use Ollama local API
            try:
                return self._raw_completion(prompt, model)
            except RuntimeError:
                # Try fallback models
                for fallback_model in self.FALLBACK_MODELS:
                    if fallback_model == self.model_name:
                        continue
                    try:
                        return self._raw_completion(prompt, model=fallback_model)
                    except RuntimeError:
                        continue
                raise RuntimeError("All Ollama models failed")
    
    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None, **kwargs) -> str:
        """Async completion (delegates to sync for simplicity)."""
        return self.completion(prompt, model)
    
    def get_usage_summary(self) -> UsageSummary:
        """Get aggregated usage summary for all model calls."""
        total_calls = sum(self.model_call_counts.values())
        total_input = sum(self.model_input_tokens.values())
        total_output = sum(self.model_output_tokens.values())
        
        by_model = {}
        for model in self.model_call_counts:
            by_model[model] = ModelUsageSummary(
                model=model,
                calls=self.model_call_counts[model],
                input_tokens=self.model_input_tokens[model],
                output_tokens=self.model_output_tokens[model],
                duration=0.0,  # Ollama doesn't track duration per call
                errors=0,
            )
        
        return UsageSummary(
            total_calls=total_calls,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_duration=0.0,
            by_model=by_model,
        )
    
    def get_last_usage(self) -> ModelUsageSummary:
        """Get usage for the last model call."""
        if self._last_usage is None:
            return ModelUsageSummary(
                model=self.model_name,
                calls=0,
                input_tokens=0,
                output_tokens=0,
                duration=0.0,
                errors=0,
            )
        return self._last_usage
