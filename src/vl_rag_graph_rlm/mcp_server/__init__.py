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

from vl_rag_graph_rlm.mcp_server.streamlined import mcp, main

__all__ = ["mcp", "main"]
