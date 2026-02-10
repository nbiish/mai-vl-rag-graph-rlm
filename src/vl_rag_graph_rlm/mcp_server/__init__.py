"""VL-RAG-Graph-RLM MCP Server.

Exposes the full VL-RAG-Graph-RLM pipeline as MCP tools for use by
LLM clients (Windsurf, Claude Desktop, etc.).

By default, the server uses the provider hierarchy system to resolve
the best available LLM provider. Users can override the provider,
model, and prompt template via a settings.json file.

Settings file location (checked in order):
    1. $VRLMRAG_MCP_SETTINGS  (env var pointing to a JSON file)
    2. <project_root>/.vrlmrag/mcp_settings.json
    3. Built-in defaults (provider=auto, hierarchy system)
"""

from vl_rag_graph_rlm.mcp_server.server import mcp, main

__all__ = ["mcp", "main"]
