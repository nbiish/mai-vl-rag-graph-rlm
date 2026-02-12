"""Allow running the MCP server via ``python -m vl_rag_graph_rlm.mcp_server``.

Uses the streamlined server (4 tools instead of 11+).
"""

from vl_rag_graph_rlm.mcp_server.streamlined import main

if __name__ == "__main__":
    main()
