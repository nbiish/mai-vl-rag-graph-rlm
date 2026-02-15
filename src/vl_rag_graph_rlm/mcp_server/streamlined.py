"""Streamlined MCP server for VL-RAG-Graph-RLM.

Consolidated tools for reduced context usage while maintaining full functionality:
- 4 core tools instead of 11+ separate tools
- Smart defaults with --comprehensive mode support
- Unified collection operations
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, List

from mcp.server.fastmcp import FastMCP, Context

# Bootstrap: resolve codebase root and load .env
def _find_project_root() -> Path:
    env_root = os.getenv("VRLMRAG_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / ".env").exists() or (p / "pyproject.toml").exists():
            return p
    file_root = Path(__file__).resolve().parent.parent.parent.parent
    if (file_root / "pyproject.toml").exists():
        return file_root
    cwd = Path.cwd()
    if (cwd / ".env").exists() or (cwd / "pyproject.toml").exists():
        return cwd
    return file_root

_PROJECT_ROOT = _find_project_root()
_SRC_DIR = _PROJECT_ROOT / "src"
_ENV_FILE = _PROJECT_ROOT / ".env"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dotenv import load_dotenv
if _ENV_FILE.exists():
    load_dotenv(dotenv_path=_ENV_FILE, override=False)
else:
    load_dotenv()

# Imports
from vl_rag_graph_rlm import VLRAGGraphRLM
from vl_rag_graph_rlm.clients.hierarchy import get_available_providers
from vl_rag_graph_rlm.collections import (
    collection_exists, create_collection, load_collection_meta,
    list_collections as _list_collections, _collection_dir,
)
from vl_rag_graph_rlm.mcp_server.settings import MCPSettings, load_settings

logger = logging.getLogger("vl_rag_graph_rlm.mcp_server")
_SETTINGS = load_settings()

logging.basicConfig(
    level=getattr(logging, _SETTINGS.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# Provider context budgets
_PROVIDER_BUDGETS: dict[str, int] = {
    "sambanova": 32000, "nebius": 100000, "openrouter": 32000,
    "openai": 32000, "anthropic": 32000, "gemini": 64000,
    "groq": 32000, "deepseek": 32000, "mistral": 32000,
    "fireworks": 32000, "together": 32000, "zenmux": 32000,
    "zai": 32000, "azure_openai": 32000, "cerebras": 32000,
    "ollama": 32000,
}

# Initialize MCP server
mcp = FastMCP(
    "VL-RAG-Graph-RLM",
    instructions=(
        "Multimodal document analysis with Vision-Language embeddings, "
        "hybrid RAG search, knowledge-graph extraction, and recursive "
        "LLM reasoning. Use --comprehensive flag for maximum quality."
    ),
)


def _resolve_auto_provider() -> str:
    """Local implementation to resolve auto provider without import dependency."""
    import os
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
    return "openrouter"  # fallback


def _effective_provider_model(
    settings: MCPSettings,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Resolve effective provider and model."""
    base_provider, base_model = settings.resolve_provider_model()
    provider = provider_override or base_provider
    model = model_override or base_model
    if provider == "auto":
        provider = _resolve_auto_provider()
    return provider, model


def _get_settings(ctx: Context) -> MCPSettings:
    try:
        return ctx.request_context.lifespan_context.settings
    except (AttributeError, TypeError):
        return load_settings()


# ═══════════════════════════════════════════════════════════════════════════
# Core Tools (4 consolidated tools instead of 11+)
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def analyze(
    ctx: Context,
    input_path: str,
    query: Optional[str] = None,
    mode: str = "balanced",
    output_path: Optional[str] = None,
) -> str:
    """Analyze documents. Default is balanced — use mode='comprehensive' for deep analysis.
    
    Args:
        input_path: Path to file or folder
        query: Question to answer (auto-generated if not provided)
        mode: balanced (default) or comprehensive
        output_path: Optional path to save report
    """
    settings = _get_settings(ctx)
    eff_provider, eff_model = _effective_provider_model(settings)

    canonical_mode = mode.strip().lower()
    if canonical_mode not in {"balanced", "comprehensive"}:
        return f"Error: Unknown mode '{mode}'. Use: balanced or comprehensive"

    profile_settings = {
        "balanced": {
            "max_depth": 3,
            "max_iterations": 8,
            "multi_query": True,
            "graph_augmented": True,
            "graph_hops": 2,
        },
        "comprehensive": {
            "max_depth": 5,
            "max_iterations": 15,
            "multi_query": True,
            "graph_augmented": True,
            "graph_hops": 2,
            "verbose": True,
        },
    }

    profile = profile_settings[canonical_mode]
    
    from vrlmrag import run_analysis
    
    try:
        results = run_analysis(
            provider=eff_provider,
            input_path=input_path,
            query=query,
            output=output_path,
            model=eff_model,
            max_depth=profile["max_depth"],
            max_iterations=profile["max_iterations"],
            use_api=settings.use_api,
            text_only=False,  # Support all content types: images, video, audio, documents
            multi_query=profile.get("multi_query", False),
            use_graph_augmented=profile.get("graph_augmented", False),
            graph_hops=profile.get("graph_hops", 2),
            output_format="markdown",
            verbose=profile.get("verbose", False),
            _quiet=False,
        )
        
        # Format concise but informative response
        lines = [
            f"# Analysis Complete [{canonical_mode} mode]",
            f"",
            f"**Provider:** {results.get('provider', 'N/A')} | "
            f"**Model:** {results.get('model', 'N/A')} | "
            f"**Time:** {results.get('execution_time', 0):.1f}s",
            f"**Documents:** {results.get('document_count', 0)} | "
            f"**Chunks:** {results.get('total_chunks', 0)}",
            f"",
        ]
        
        for qr in results.get("queries", []):
            lines.extend([f"## Q: {qr['query']}", "", qr.get("response", "No response"), ""])
        
        if output_path:
            lines.append(f"*Report saved to: {output_path}*")
            
        return "\n".join(lines)
        
    except Exception as e:
        return f"Analysis failed: {e}"


@mcp.tool()
async def query_collection(
    ctx: Context,
    collection: str,
    query: str,
    mode: str = "balanced",
) -> str:
    """Query knowledge collections. Default is balanced — use mode='comprehensive' for deep analysis.

    Args:
        collection: Collection name
        query: Question to answer
        mode: balanced (default) or comprehensive
    """
    settings = _get_settings(ctx)
    eff_provider, eff_model = _effective_provider_model(settings)
    
    # Create collection if it doesn't exist
    if not collection_exists(collection):
        create_collection(collection, description=f"Auto-created for query")
        return f"Collection '{collection}' created. Add documents with: collection_add"
    
    canonical_mode = mode.strip().lower()
    if canonical_mode not in {"balanced", "comprehensive"}:
        return f"Error: Unknown mode '{mode}'. Use: balanced or comprehensive"

    profile_settings = {
        "balanced": {"max_depth": 3, "max_iterations": 8, "multi_query": True, "graph_augmented": True},
        "comprehensive": {"max_depth": 5, "max_iterations": 15, "multi_query": True, "graph_augmented": True},
    }

    profile = profile_settings[canonical_mode]
    
    from vrlmrag import run_collection_query
    import io
    from contextlib import redirect_stdout
    
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_collection_query(
            collection_names=[collection],
            query=query,
            provider=eff_provider,
            model=eff_model,
            max_depth=profile["max_depth"],
            max_iterations=profile["max_iterations"],
            use_api=settings.use_api,
        )
    
    return buf.getvalue()


@mcp.tool()
async def collection_manage(
    ctx: Context,
    action: str,
    collection: Optional[str] = None,
    path: Optional[str] = None,
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    target_collection: Optional[str] = None,
) -> str:
    """Manage collections: add, list, info, delete, export, import, merge, tag, search.
    
    Args:
        action: Operation to perform
        collection: Collection name (for add/info/delete/export/tag)
        path: File/folder path (for add) or archive path (for import/export)
        query: Search query (for search action)
        tags: Tags to add (for tag action) or filter (for search)
        target_collection: Target for merge operation
    """
    settings = _get_settings(ctx)
    
    if action == "list":
        collections = _list_collections()
        if not collections:
            return "No collections found."
        lines = [f"{'Name':<25} {'Docs':>6} {'Chunks':>8} {'Updated':>20}", "-" * 65]
        for meta in collections:
            name = meta.get('display_name', meta['name'])
            docs = meta.get('document_count', 0)
            chunks = meta.get('chunk_count', 0)
            updated = meta.get('updated', '?')[:19]
            lines.append(f"{name:<25} {docs:>6} {chunks:>8}  {updated}")
        return "\n".join(lines)
    
    if action == "info" and collection:
        if not collection_exists(collection):
            return f"Collection '{collection}' not found."
        meta = load_collection_meta(collection)
        return json.dumps(meta, indent=2, default=str)
    
    if action == "add" and collection and path:
        from vrlmrag import run_collection_add
        eff_provider, eff_model = _effective_provider_model(settings)
        run_collection_add(
            collection_names=[collection],
            input_path=path,
            provider=eff_provider,
            model=eff_model,
            max_depth=settings.max_depth,
            max_iterations=settings.max_iterations,
        )
        meta = load_collection_meta(collection)
        return f"Added to '{collection}': {meta.get('document_count', 0)} docs, {meta.get('chunk_count', 0)} chunks"
    
    if action == "delete" and collection:
        from vl_rag_graph_rlm.collections import delete_collection
        if delete_collection(collection):
            return f"Deleted collection: {collection}"
        return f"Collection not found: {collection}"
    
    if action == "export" and collection and path:
        from vl_rag_graph_rlm.collections import export_collection
        try:
            archive_path = export_collection(collection, path)
            return f"Exported to: {archive_path}"
        except Exception as e:
            return f"Export failed: {e}"
    
    if action == "import" and path:
        from vl_rag_graph_rlm.collections import import_collection
        try:
            meta = import_collection(path)
            return f"Imported as '{meta['name']}'"
        except Exception as e:
            return f"Import failed: {e}"
    
    if action == "merge" and collection and target_collection:
        from vl_rag_graph_rlm.collections import merge_collections
        try:
            meta = merge_collections(collection, target_collection)
            return f"Merged '{collection}' into '{target_collection}'"
        except Exception as e:
            return f"Merge failed: {e}"
    
    if action == "tag" and collection and tags:
        from vl_rag_graph_rlm.collections import add_tags
        add_tags(collection, tags)
        return f"Added tags to '{collection}': {', '.join(tags)}"
    
    if action == "search":
        from vl_rag_graph_rlm.collections import search_collections
        results = search_collections(query=query, tags=tags)
        if not results:
            return "No collections found."
        lines = [f"Found {len(results)} collection(s):", ""]
        for c in results:
            name = c.get('display_name', c['name'])
            lines.append(f"- {name}: {c.get('document_count', 0)} docs")
        return "\n".join(lines)
    
    return f"Unknown action: {action}. Use: add, list, info, delete, export, import, merge, tag, search"


# Entry point
def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
