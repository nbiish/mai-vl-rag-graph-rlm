"""Vision-Language RAG Graph Recursive Language Models (VL_RAG_GRAPH_RLM).

A unified framework combining alexzhang13/rlm and ysz/recursive-llm
with support for OpenRouter, ZenMux, z.ai, and 100+ providers via LiteLLM.

Includes SOTA RAG capabilities:
- Hybrid search (dense + keyword with RRF fusion)
- Multi-factor reranking
- Provider-agnostic embeddings
- Vision/multimodal support
- Qwen3-VL multimodal embeddings (text, image, video)

Quick Start:
    >>> from vl_rag_graph_rlm import MultimodalRAGPipeline
    >>> 
    >>> # Unified pipeline
    >>> pipeline = MultimodalRAGPipeline()
    >>> pipeline.add_pdf("document.pdf", extract_images=True)
    >>> result = pipeline.query("Explain Figure 3")
    >>> print(result.answer)
"""

from vl_rag_graph_rlm.rlm_core import (
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
    GenericOpenAIClient,
    AzureOpenAIClient,
    OpenRouterClient,
    ZenMuxClient,
    ZaiClient,
    AnthropicClient,
    AnthropicCompatibleClient,
    GeminiClient,
    LiteLLMClient,
    GroqClient,
    MistralClient,
    FireworksClient,
    TogetherClient,
    DeepSeekClient,
    CerebrasClient,
)
from vl_rag_graph_rlm.types import (
    VLRAGGraphRLMChatCompletion,
    REPLResult,
    UsageSummary,
    ModelUsageSummary,
    ProviderType,
)
from vl_rag_graph_rlm.pipeline import (
    MultimodalRAGPipeline,
    PipelineResult,
    create_pipeline,
)

__version__ = "0.1.1"

__all__ = [
    # Unified Pipeline
    "MultimodalRAGPipeline",
    "PipelineResult",
    "create_pipeline",
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
    "GenericOpenAIClient",
    "AzureOpenAIClient",
    "OpenRouterClient",
    "ZenMuxClient",
    "ZaiClient",
    "AnthropicClient",
    "AnthropicCompatibleClient",
    "GeminiClient",
    "LiteLLMClient",
    "GroqClient",
    "MistralClient",
    "FireworksClient",
    "TogetherClient",
    "DeepSeekClient",
    "CerebrasClient",
    # Types
    "VLRAGGraphRLMChatCompletion",
    "REPLResult",
    "UsageSummary",
    "ModelUsageSummary",
    "ProviderType",
    # RAG submodules
    "rag",
    "vision",
    "pipeline",
]
