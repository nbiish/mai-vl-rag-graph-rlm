"""Vision-Language RAG Graph Recursive Language Models (VL_RAG_GRAPH_RLM).

A unified framework combining alexzhang13/rlm and ysz/recursive-llm
with support for OpenRouter, ZenMux, z.ai, and 100+ providers via LiteLLM.

Includes SOTA RAG capabilities:
- Hybrid search (dense + keyword with RRF fusion)
- Multi-factor reranking
- Provider-agnostic embeddings
- Vision/multimodal support
- Qwen3-VL multimodal embeddings (text, image, video)

Example:
    >>> from vl_rag_graph_rlm import VLRAGGraphRLM
    >>>
    >>> # OpenRouter with cheap SOTA model
    >>> vlrag = VLRAGGraphRLM(provider="openrouter")
    >>> result = vlrag.completion("Summarize this", context=long_document)
    >>> print(result.response)
    >>>
    >>> # RAG-enhanced
    >>> from vl_rag_graph_rlm.rag import RAGEnhancedVLRAGGraphRLM
    >>> rag_vlrag = RAGEnhancedVLRAGGraphRLM(
    ...     llm_provider="openrouter",
    ...     llm_model="kimi/kimi-k2.5",
    ...     embedding_provider="openrouter"
    ... )
    >>> result = rag_vlrag.query("What is the main topic?")
    >>>
    >>> # Multimodal RAG with Qwen3-VL
    >>> from vl_rag_graph_rlm.rag import MultimodalVLRAGGraphRLM
    >>> mm_rag = MultimodalVLRAGGraphRLM(
    ...     llm_provider="openrouter",
    ...     llm_model="gpt-4o",
    ...     embedding_model="Qwen/Qwen3-VL-Embedding-2B"
    ... )
    >>> mm_rag.add_pdf("document.pdf")
    >>> mm_rag.add_image("diagram.png")
    >>> result = mm_rag.query("Explain the diagram")
"""

from vl_rag_graph_rlm.core import (
    VLRAGGraphRLM,
    vlraggraphrlm_complete,
    VLRAGGraphRLMError,
    MaxIterationsError,
    MaxDepthError,
)
from vl_rag_graph_rlm.clients import (
    get_client,
    BaseLM,
    OpenAIClient,
    OpenRouterClient,
    ZenMuxClient,
    ZaiClient,
    AnthropicClient,
    GeminiClient,
    LiteLLMClient,
)
from vl_rag_graph_rlm.types import (
    VLRAGGraphRLMChatCompletion,
    REPLResult,
    UsageSummary,
    ModelUsageSummary,
    ProviderType,
)

__version__ = "0.1.0"

__all__ = [
    # Core VL_RAG_GRAPH_RLM
    "VLRAGGraphRLM",
    "vlraggraphrlm_complete",
    "VLRAGGraphRLMError",
    "MaxIterationsError",
    "MaxDepthError",
    # Client factory
    "get_client",
    "BaseLM",
    # Specific clients
    "OpenAIClient",
    "OpenRouterClient",
    "ZenMuxClient",
    "ZaiClient",
    "AnthropicClient",
    "GeminiClient",
    "LiteLLMClient",
    # Types
    "VLRAGGraphRLMChatCompletion",
    "REPLResult",
    "UsageSummary",
    "ModelUsageSummary",
    "ProviderType",
    # RAG (imported from submodule)
    "rag",
    "vision",
]
