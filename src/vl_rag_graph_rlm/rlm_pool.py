"""RLM connection pooling and async optimization.

Provides connection reuse across recursive RLM calls to reduce
latency from TCP handshake and HTTP client initialization.
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass

logger = logging.getLogger("rlm.pool")


@dataclass
class RLMClientConfig:
    """Configuration for pooled RLM client."""
    provider: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_depth: int = 3
    max_iterations: int = 10
    temperature: float = 0.0


class RLMConnectionPool:
    """Pool and reuse RLM client connections.
    
    Reduces latency by eliminating repeated HTTP client initialization
    and TCP handshakes across recursive RLM calls.
    
    Example:
        >>> pool = RLMConnectionPool(max_size=10)
        >>> 
        >>> # Acquire client from pool
        >>> async with pool.acquire("openrouter", "gpt-4o") as rlm:
        >>>     result = await rlm.acompletion(query, context)
        >>> 
        >>> # Client returned to pool automatically
    """
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._pools: Dict[str, asyncio.Queue] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._client_counts: Dict[str, int] = {}
    
    def _get_pool_key(self, config: RLMClientConfig) -> str:
        """Generate unique key for client configuration."""
        return f"{config.provider}:{config.model}:{config.api_base or 'default'}"
    
    @asynccontextmanager
    async def acquire(self, config: RLMClientConfig):
        """Acquire an RLM client from the pool.
        
        Args:
            config: RLM client configuration
            
        Yields:
            VLRAGGraphRLM instance (from pool or newly created)
        """
        from vl_rag_graph_rlm.rlm_core import VLRAGGraphRLM
        
        key = self._get_pool_key(config)
        
        # Initialize pool for this config if needed
        if key not in self._pools:
            self._pools[key] = asyncio.Queue(maxsize=self.max_size)
            self._semaphores[key] = asyncio.Semaphore(self.max_size)
            self._client_counts[key] = 0
        
        pool = self._pools[key]
        semaphore = self._semaphores[key]
        
        async with semaphore:
            # Try to get from pool
            if not pool.empty():
                rlm = await pool.get()
                logger.debug(f"Reused RLM client from pool: {key}")
            else:
                # Create new client
                rlm = VLRAGGraphRLM(
                    provider=config.provider,
                    model=config.model,
                    api_key=config.api_key,
                    api_base=config.api_base,
                    max_depth=config.max_depth,
                    max_iterations=config.max_iterations,
                    temperature=config.temperature,
                )
                self._client_counts[key] += 1
                logger.debug(f"Created new RLM client: {key} (total: {self._client_counts[key]})")
            
            try:
                yield rlm
            finally:
                # Return to pool (reset depth for reuse)
                rlm._current_depth = 0
                try:
                    pool.put_nowait(rlm)
                except asyncio.QueueFull:
                    # Pool full, discard client
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            key: {
                "pool_size": q.qsize(),
                "max_size": self.max_size,
                "total_created": self._client_counts.get(key, 0),
            }
            for key, q in self._pools.items()
        }
    
    async def clear(self):
        """Clear all pools."""
        for pool in self._pools.values():
            while not pool.empty():
                try:
                    pool.get_nowait()
                except asyncio.QueueEmpty:
                    break
        self._pools.clear()
        self._semaphores.clear()
        self._client_counts.clear()


# Global pool instance
_global_pool: Optional[RLMConnectionPool] = None


def get_global_pool(max_size: int = 10) -> RLMConnectionPool:
    """Get or create global RLM connection pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = RLMConnectionPool(max_size=max_size)
    return _global_pool


async def pooled_rlm_completion(
    query: str,
    context: str,
    provider: str = "openrouter",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """Execute RLM completion using pooled connection.
    
    Args:
        query: User query
        context: Context to analyze
        provider: API provider
        model: Model name (optional)
        api_key: API key (optional)
        **kwargs: Additional RLM parameters
        
    Returns:
        Completion response text
    """
    from vl_rag_graph_rlm.rlm_core import _get_default_model
    
    config = RLMClientConfig(
        provider=provider,
        model=model or _get_default_model(provider),
        api_key=api_key,
        **kwargs
    )
    
    pool = get_global_pool()
    
    async with pool.acquire(config) as rlm:
        result = await rlm.acompletion(query, context)
        return result.response
