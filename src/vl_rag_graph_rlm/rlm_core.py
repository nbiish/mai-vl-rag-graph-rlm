"""Core VL_RAG_GRAPH_RLM implementation combining Vision-Language, RAG, and Graph-based reasoning."""

import asyncio
import os
import time
from typing import Any, Optional

from vl_rag_graph_rlm.clients import get_client, BaseLM
from vl_rag_graph_rlm.environments.repl import REPLExecutor, REPLError
from vl_rag_graph_rlm.types import (
    CodeBlock,
    REPLResult,
    RLMIteration,
    VLRAGGraphRLMChatCompletion as RLMChatCompletion,
    ProviderType,
)
from vl_rag_graph_rlm.utils.parsing import find_code_blocks, find_final_answer, format_iteration
from vl_rag_graph_rlm.utils.prompts import build_system_prompt


def _get_default_model(provider: str) -> str:
    """Get default model for provider, checking env vars first."""
    env_var_map = {
        "openrouter": "OPENROUTER_MODEL",
        "zenmux": "ZENMUX_MODEL",
        "zai": "ZAI_MODEL",
        "openai": "OPENAI_MODEL",
        "anthropic": "ANTHROPIC_MODEL",
        "gemini": "GOOGLE_MODEL",
        "azure_openai": "AZURE_OPENAI_MODEL",
        "groq": "GROQ_MODEL",
        "mistral": "MISTRAL_MODEL",
        "fireworks": "FIREWORKS_MODEL",
        "together": "TOGETHER_MODEL",
        "deepseek": "DEEPSEEK_MODEL",
        "sambanova": "SAMBANOVA_MODEL",
        "nebius": "NEBIUS_MODEL",
        "cerebras": "CEREBRAS_MODEL",
        "modalresearch": "MODAL_RESEARCH_MODEL",
        "ollama": "OLLAMA_MODEL",
    }
    env_var = env_var_map.get(provider, f"{provider.upper()}_MODEL")
    env_model = os.getenv(env_var)
    if env_model:
        return env_model
    
    # Fall back to hardcoded defaults
    hardcoded_defaults = {
        "openrouter": "minimax/minimax-m2.1",
        "zenmux": "moonshotai/kimi-k2.5",  # ZenMux uses provider/model format
        "zai": "glm-4.7",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-20241022",
        "gemini": "gemini-1.5-flash",
        "groq": "moonshotai/kimi-k2-instruct-0905",  # Kimi K2 on Groq LPU
        "mistral": "mistral-large-latest",
        "fireworks": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "deepseek": "deepseek-chat",
        "sambanova": "DeepSeek-V3-0324",  # DeepSeek V3 on SambaNova (32K+ context, production)
        "nebius": "MiniMaxAI/MiniMax-M2.1",  # MiniMax M2.1 on Nebius Token Factory
        "cerebras": "zai-glm-4.7",  # GLM 4.7 355B on Cerebras wafer-scale (~1000 tok/s)
        "modalresearch": "zai-org/GLM-5-FP8",  # GLM-5 745B on Modal Research (experimental, free)
        "ollama": "llama3.2",
    }
    return hardcoded_defaults.get(provider, "gpt-4o-mini")


def _get_recursive_model(provider: str, primary_model: str) -> str:
    """Get cheaper recursive model, checking env vars first."""
    env_var_map = {
        "openrouter": "OPENROUTER_RECURSIVE_MODEL",
        "zenmux": "ZENMUX_RECURSIVE_MODEL",
        "zai": "ZAI_RECURSIVE_MODEL",
        "openai": "OPENAI_RECURSIVE_MODEL",
        "anthropic": "ANTHROPIC_RECURSIVE_MODEL",
        "gemini": "GOOGLE_RECURSIVE_MODEL",
    }
    env_var = env_var_map.get(provider, f"{provider.upper()}_RECURSIVE_MODEL")
    env_model = os.getenv(env_var)
    if env_model:
        return env_model
    
    # Fall back to hardcoded defaults
    hardcoded_recursive = {
        "openrouter": "solar-pro/solar-pro-3:free",
        "zenmux": "z-ai/glm-4.7-flash",  # ZenMux uses provider/model format
        "zai": "glm-4.5-air",  # Lightweight model available on z.ai Coding Plan
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-20241022",
        "gemini": "gemini-1.5-flash",
        "groq": "llama-3.3-70b-versatile",  # Fast fallback on Groq LPU
        "cerebras": "gpt-oss-120b",  # Ultra-fast on Cerebras (~3000 tok/s)
        "sambanova": "DeepSeek-V3.1",  # Latest DeepSeek on SambaNova
        "nebius": "zai-org/GLM-4.7-FP8",  # GLM 4.7 on Nebius Token Factory
        "modalresearch": "zai-org/GLM-5-FP8",  # Single model available on Modal Research
        "ollama": "llama3.2",
    }
    return hardcoded_recursive.get(provider, primary_model)


class VLRAGGraphRLMError(Exception):
    """Base error for VL_RAG_GRAPH_RLM."""
    pass


class MaxIterationsError(VLRAGGraphRLMError):
    """Max iterations exceeded."""
    pass


class MaxDepthError(VLRAGGraphRLMError):
    """Max recursion depth exceeded."""
    pass


class VLRAGGraphRLM:
    """
    Unified Recursive Language Model.

    Combines the best of alexzhang13/rlm and ysz/recursive-llm:
    - Multiple provider support (OpenRouter, ZenMux, z.ai, OpenAI, Anthropic, Gemini)
    - LiteLLM fallback for 100+ providers
    - Safe REPL execution with RestrictedPython
    - Recursive sub-processing with depth tracking
    
    Model Selection Priority:
    1. Code-specified model (highest priority)
    2. {PROVIDER}_MODEL environment variable
    3. Hardcoded default (lowest priority)
    
    Recommended Cheap SOTA Models (Jan 2026):
    - OpenRouter: kimi/kimi-k2.5, z-ai/glm-4.7, solar-pro/solar-pro-3:free
    - ZenMux: ernie-5.0-thinking-preview, dubao-seed-1.8, glm-4.7-flash
    - z.ai: glm-4.7, glm-4.7-coding
    - OpenAI: gpt-4o-mini (cheap), gpt-4o (capable)
    """

    def __init__(
        self,
        provider: ProviderType | str = "openrouter",
        model: str | None = None,
        recursive_model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_depth: int = 3,
        max_iterations: int = 10,
        temperature: float = 0.0,
        _current_depth: int = 0,
        **client_kwargs
    ):
        """
        Initialize VLRAGGraphRLM.

        Args:
            provider: API provider ('openrouter', 'zenmux', 'zai', 'openai', 'anthropic', 'gemini', 'litellm')
            model: Model name (defaults to env var {PROVIDER}_MODEL or hardcoded default)
            recursive_model: Optional cheaper model for recursive calls (defaults to env var or free/cheap tier)
            api_key: API key (falls back to environment variable)
            api_base: Custom API base URL
            max_depth: Maximum recursion depth
            max_iterations: Maximum REPL iterations per call
            temperature: Sampling temperature
            _current_depth: Internal depth tracker (don't set manually)
            **client_kwargs: Additional arguments for the client
            
        Examples:
            # Use model from env var or default (OpenRouter + Kimi K2.5)
            >>> vlrag = VLRAGGraphRLM()
            
            # Override model in code for specific section
            >>> vlrag = VLRAGGraphRLM(provider="openrouter", model="gpt-4o")
            
            # Set model via environment
            >>> # export OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
            >>> vlrag = VLRAGGraphRLM(provider="openrouter")
        """
        self.provider = provider
        
        # Model resolution priority: code > env var > hardcoded default
        self.model = model or _get_default_model(provider)
        self.recursive_model = recursive_model or _get_recursive_model(provider, self.model)
        
        self.api_key = api_key
        self.api_base = api_base
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.temperature = temperature
        self._current_depth = _current_depth
        self.client_kwargs = client_kwargs

        # Initialize client
        client_args = {
            "model_name": self.model,
            **client_kwargs
        }
        if api_key:
            client_args["api_key"] = api_key
        if api_base:
            client_args["api_base"] = api_base

        self.client = get_client(provider, **client_args)
        self.repl = REPLExecutor()

        # Statistics
        self._llm_calls = 0
        self._iterations = 0

    def completion(
        self,
        query: str = "",
        context: str = "",
        **kwargs
    ) -> RLMChatCompletion:
        """
        Execute RLM completion (synchronous).

        Args:
            query: User query
            context: Context to analyze
            **kwargs: Additional parameters

        Returns:
            RLMChatCompletion with answer and metadata
        """
        return asyncio.run(self.acompletion(query, context, **kwargs))

    async def acompletion(
        self,
        query: str = "",
        context: str = "",
        **kwargs
    ) -> RLMChatCompletion:
        """
        Execute RLM completion (asynchronous).

        Args:
            query: User query
            context: Context to analyze
            **kwargs: Additional parameters

        Returns:
            RLMChatCompletion with answer and metadata
        """
        start_time = time.time()

        # Normalize inputs
        if query and not context:
            context = query
            query = ""

        # Check depth
        if self._current_depth >= self.max_depth:
            # At max depth, just return direct completion
            prompt = f"Context:\n{context}\n\nQuery: {query}" if query else context
            response = await self.client.acompletion(prompt)
            exec_time = time.time() - start_time
            return RLMChatCompletion(
                provider=self.provider,
                model=self.model,
                prompt=prompt,
                response=response,
                usage_summary=self.client.get_last_usage(),
                execution_time=exec_time
            )

        # Build REPL environment
        env = self._build_repl_env(query, context)

        # Build initial messages
        system_prompt = build_system_prompt(len(context), self._current_depth, self.max_depth)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query or "Analyze the context and provide insights."}
        ]

        # Main loop
        for iteration in range(self.max_iterations):
            self._iterations = iteration + 1

            # Call LLM
            response = await self._call_llm(messages, **kwargs)

            # Check for final answer
            final_answer = find_final_answer(response, env)
            if final_answer is not None:
                exec_time = time.time() - start_time
                return RLMChatCompletion(
                    provider=self.provider,
                    model=self.model,
                    prompt=query,
                    response=final_answer,
                    usage_summary=self.client.get_usage_summary(),
                    execution_time=exec_time
                )

            # Execute code
            code_blocks = self._execute_code_blocks(response, env)

            # Format iteration for history
            iteration_data = RLMIteration(
                prompt=messages.copy(),
                response=response,
                code_blocks=[CodeBlock(code=cb, result=res) for cb, res in code_blocks],
                iteration_time=time.time() - start_time
            )

            # Add to conversation
            new_messages = format_iteration(iteration_data)
            messages.extend(new_messages)

        # Max iterations exceeded - fallback
        fallback_response = await self._fallback_completion(messages)
        exec_time = time.time() - start_time
        return RLMChatCompletion(
            provider=self.provider,
            model=self.model,
            prompt=query,
            response=fallback_response,
            usage_summary=self.client.get_usage_summary(),
            execution_time=exec_time
        )

    async def _call_llm(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Call the LLM."""
        self._llm_calls += 1

        # Merge kwargs
        call_kwargs = {"temperature": self.temperature, **kwargs}

        # Use recursive model for depth > 0
        if self._current_depth > 0 and self.recursive_model != self.model:
            call_kwargs["model"] = self.recursive_model

        return await self.client.acompletion(messages, **call_kwargs)

    def _execute_code_blocks(self, response: str, env: dict) -> list[tuple[str, REPLResult]]:
        """Execute code blocks from response."""
        code_blocks = find_code_blocks(response)
        results = []

        for code in code_blocks:
            try:
                exec_start = time.time()
                output = self.repl.execute(code, env)
                exec_time = time.time() - exec_start

                result = REPLResult(
                    stdout=output,
                    stderr="",
                    locals=env,
                    execution_time=exec_time
                )
                results.append((code, result))

            except REPLError as e:
                result = REPLResult(
                    stdout="",
                    stderr=str(e),
                    locals=env,
                    execution_time=0.0
                )
                results.append((code, result))

        return results

    async def _fallback_completion(self, messages: list[dict[str, str]]) -> str:
        """Generate final answer when max iterations reached."""
        fallback_messages = messages + [{
            "role": "user",
            "content": "Maximum iterations reached. Please provide your best final answer based on the analysis above."
        }]
        return await self.client.acompletion(fallback_messages)

    def _build_repl_env(self, query: str, context: str) -> dict[str, Any]:
        """Build REPL environment."""
        import re as re_module

        env: dict[str, Any] = {
            'context': context,
            'query': query,
            'recursive_llm': self._make_recursive_fn(),
            're': re_module,
        }
        return env

    def _make_recursive_fn(self) -> Any:
        """Create recursive LLM function for REPL."""
        def sync_recursive_llm(sub_query: str, sub_context: str) -> str:
            """Sync wrapper for recursive calls."""
            if self._current_depth + 1 >= self.max_depth:
                return f"Max recursion depth ({self.max_depth}) reached"

            # Create sub-RLM with increased depth
            sub_rlm = VLRAGGraphRLM(
                provider=self.provider,
                model=self.recursive_model,
                recursive_model=self.recursive_model,
                api_key=self.api_key,
                api_base=self.api_base,
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
                temperature=self.temperature,
                _current_depth=self._current_depth + 1,
                **self.client_kwargs
            )

            try:
                result = sub_rlm.completion(sub_query, sub_context)
                return result.response if hasattr(result, 'response') else str(result)
            except Exception as e:
                return f"Recursive call error: {str(e)}"

        return sync_recursive_llm

    @property
    def stats(self) -> dict[str, int]:
        """Get execution statistics."""
        return {
            'llm_calls': self._llm_calls,
            'iterations': self._iterations,
            'depth': self._current_depth,
        }


# Convenience function for simple usage
def vlraggraphrlm_complete(
    query: str,
    context: str = "",
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Simple function to get RLM completion.

    Args:
        query: User query
        context: Context to analyze
        provider: API provider
        model: Model name
        api_key: API key (or set env var)
        **kwargs: Additional RLM parameters

    Returns:
        Final answer string
    """
    rlm = VLRAGGraphRLM(provider=provider, model=model, api_key=api_key, **kwargs)
    result = rlm.completion(query, context)
    return result.response
