"""Ollama provider for local LLM inference.

Supports local models via Ollama API (http://localhost:11434).
"""

import os
from typing import Optional

from .base import BaseClient, CompletionResult


class OllamaClient(BaseClient):
    """Client for Ollama local LLM inference.
    
    Requires Ollama running locally (default: http://localhost:11434).
    Install from: https://ollama.com
    
    Typical models:
    - llama3.2 (3B, fast)
    - llama3.1 (8B, good balance)
    - mistral (7B)
    - qwen2.5 (various sizes)
    - deepseek-r1 (reasoning model)
    """
    
    # Fallback chain: try smaller models if larger fail
    FALLBACK_MODELS = [
        "llama3.2",
        "llama3.1",
        "mistral",
        "qwen2.5:7b",
    ]
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_depth: int = 3,
        max_iterations: int = 10,
        base_url: Optional[str] = None,
    ):
        """Initialize Ollama client.
        
        Args:
            model: Model name (default: from OLLAMA_MODEL env var, or llama3.2)
            api_key: Not used (Ollama doesn't require API keys locally)
            temperature: Sampling temperature
            max_depth: RLM max recursion depth
            max_iterations: RLM max iterations
            base_url: Ollama API URL (default: from OLLAMA_BASE_URL env var, or http://localhost:11434)
        """
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.temperature = temperature
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Verify Ollama is accessible
        self._check_connection()
    
    def _check_connection(self) -> None:
        """Check if Ollama is running."""
        import urllib.request
        import urllib.error
        
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama returned status {response.status}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama connection check failed: {e}")
    
    def _raw_completion(
        self,
        query: str,
        context: str,
        model: Optional[str] = None,
    ) -> CompletionResult:
        """Make raw completion request to Ollama API."""
        import json
        import urllib.request
        import urllib.error
        
        use_model = model or self.model
        
        # Build prompt
        full_prompt = f"{context}\n\n{query}" if context else query
        
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
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                return CompletionResult(response=result.get("response", ""))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise RuntimeError(f"Ollama API error: {e.code} - {error_body}")
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")
    
    def completion(self, query: str, context: str) -> CompletionResult:
        """Generate completion with fallback model support."""
        try:
            return self._raw_completion(query, context)
        except RuntimeError:
            # Try fallback models
            for fallback_model in self.FALLBACK_MODELS:
                if fallback_model == self.model:
                    continue
                try:
                    return self._raw_completion(query, context, model=fallback_model)
                except RuntimeError:
                    continue
            raise RuntimeError("All Ollama models failed")
    
    async def acompletion(self, query: str, context: str) -> CompletionResult:
        """Async completion (delegates to sync for simplicity)."""
        return self.completion(query, context)
