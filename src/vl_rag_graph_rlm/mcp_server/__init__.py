"""VL-RAG-Graph-RLM MCP Server.

Streamlined 3-tool server with comprehensive defaults:
- analyze: Universal document analysis with 4 quality modes
- query_collection: Query persistent knowledge collections  
- collection_manage: Unified collection management

Always uses comprehensive RAG (max_depth=5, max_iterations=15, multi-query,
graph-augmented) with API provider hierarchy.

Two optional overrides via env vars:
    VRLMRAG_LOCAL=true      — Use local models instead of APIs
    VRLMRAG_COLLECTIONS=false — Disable collection tools
"""

import os

def _resolve_auto_provider() -> str:
    """Local implementation to resolve auto provider."""
    hierarchy = [
        "modalresearch", "sambanova", "nebius", "ollama", "groq",
        "cerebras", "zai", "zenmux", "openrouter", "gemini",
        "deepseek", "openai", "anthropic"
    ]
    key_map = {
        "modalresearch": "MODAL_RESEARCH_API_KEY",
        "sambanova": "SAMBANOVA_API_KEY",
        "nebius": "NEBIUS_API_KEY",
        "ollama": "OLLAMA_ENABLED",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "zai": "ZAI_API_KEY",
        "zenmux": "ZENMUX_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    for provider in hierarchy:
        env_key = key_map.get(provider, f"{provider.upper()}_API_KEY")
        val = os.getenv(env_key, "")
        if val and not val.startswith("your_"):
            return provider
    return "openrouter"

from vl_rag_graph_rlm.mcp_server.streamlined import mcp, main

__all__ = ["mcp", "main", "_resolve_auto_provider"]
