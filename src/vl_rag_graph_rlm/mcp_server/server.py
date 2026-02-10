"""VL-RAG-Graph-RLM MCP Server — tool definitions and entry point.

Exposes the full CLI surface as MCP tools.  By default every tool uses
the **provider hierarchy** system (``provider="auto"``) so the calling
LLM does not need to know which backend is available — the hierarchy
resolves the first provider with a valid API key automatically.

Users can pin a specific provider/model/template for *all* queries by
creating a settings file at ``.vrlmrag/mcp_settings.json`` (see
``settings.py`` for the schema).  Per-call overrides are also accepted
via the optional ``provider`` / ``model`` parameters on each tool.
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from mcp.server.fastmcp import FastMCP, Context

# ---------------------------------------------------------------------------
# Bootstrap: resolve codebase root and load .env
#
# Priority for finding the codebase root:
#   1. VRLMRAG_ROOT env var  (set in MCP client config — always works)
#   2. __file__-based walk   (works for editable / local installs)
#   3. CWD fallback          (last resort)
#
# This ensures users set their .env in the cloned repo once and the
# MCP server picks it up everywhere — no need to duplicate secrets
# into mcp_settings.json or MCP client env blocks.
# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    """Locate the VL-RAG-Graph-RLM project root directory."""
    # 1. Explicit env var (recommended for uvx / remote installs)
    env_root = os.getenv("VRLMRAG_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / ".env").exists() or (p / "pyproject.toml").exists():
            return p

    # 2. Walk up from this file (works for editable installs)
    #    server.py → mcp_server/ → vl_rag_graph_rlm/ → src/ → <root>
    file_root = Path(__file__).resolve().parent.parent.parent.parent
    if (file_root / "pyproject.toml").exists():
        return file_root

    # 3. CWD fallback
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

# Always load .env from the codebase root so API keys are available
if _ENV_FILE.exists():
    load_dotenv(dotenv_path=_ENV_FILE, override=False)
else:
    load_dotenv()

# ---------------------------------------------------------------------------
# Internal imports (after path + env setup)
# ---------------------------------------------------------------------------
from vl_rag_graph_rlm import VLRAGGraphRLM
from vl_rag_graph_rlm.clients.hierarchy import (
    get_available_providers,
    resolve_auto_provider,
)
from vl_rag_graph_rlm.collections import (
    collection_exists,
    create_collection,
    load_collection_meta,
    list_collections as _list_collections_meta,
    delete_collection,
    record_source,
    load_kg as collection_load_kg,
    save_kg as collection_save_kg,
    merge_kg as collection_merge_kg,
    _embeddings_path as collection_embeddings_path,
    _kg_path as collection_kg_path,
    _collection_dir,
)

from vl_rag_graph_rlm.mcp_server.settings import (
    MCPSettings,
    load_settings,
)

logger = logging.getLogger("vl_rag_graph_rlm.mcp_server")

# ---------------------------------------------------------------------------
# Eager settings load — must happen at module level so tool registration
# (which also happens at module level) can check collections_enabled.
# ---------------------------------------------------------------------------
_SETTINGS = load_settings()

logging.basicConfig(
    level=getattr(logging, _SETTINGS.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger.info(
    "MCP server init — provider=%s, model=%s, template=%s, collections=%s",
    _SETTINGS.provider,
    _SETTINGS.model,
    _SETTINGS.template,
    "enabled" if _SETTINGS.collections_enabled else "disabled",
)

# ---------------------------------------------------------------------------
# Supported providers (duplicated from vrlmrag.py for context budgets)
# ---------------------------------------------------------------------------
_PROVIDER_BUDGETS: dict[str, int] = {
    "sambanova": 8000,
    "nebius": 100000,
    "openrouter": 32000,
    "openai": 32000,
    "anthropic": 32000,
    "gemini": 64000,
    "groq": 32000,
    "deepseek": 32000,
    "mistral": 32000,
    "fireworks": 32000,
    "together": 32000,
    "zenmux": 32000,
    "zai": 32000,
    "azure_openai": 32000,
    "cerebras": 32000,
}

# KG extraction prompt (shared with CLI)
_KG_EXTRACTION_PROMPT = (
    "You are a knowledge-graph extraction engine. Analyse the document below "
    "and produce a structured knowledge graph in Markdown.\n\n"
    "For every entity you find, output a bullet under **Entities** with its "
    "name, type (Person, Organisation, Concept, Technology, Location, Event, "
    "Metric, etc.), and a one-line description.\n\n"
    "For every relationship between entities, output a bullet under "
    "**Relationships** in the form:\n"
    "  - EntityA → relationship → EntityB (brief context)\n\n"
    "Group entities by type. Be exhaustive — capture every meaningful concept, "
    "term, and connection. Prefer precision over brevity."
)

_DOCUMENT_INSTRUCTION = "Represent this document for retrieval."
_QUERY_INSTRUCTION = (
    "Find passages that are relevant to and answer the following query."
)


# ---------------------------------------------------------------------------
# Lifespan: load settings once at server startup
# ---------------------------------------------------------------------------
@dataclass
class AppContext:
    """Shared state available to every tool via ``ctx``."""

    settings: MCPSettings


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Yield shared context (settings loaded eagerly at module level)."""
    logger.info("MCP server starting")
    yield AppContext(settings=_SETTINGS)


# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "VL-RAG-Graph-RLM",
    instructions=(
        "Multimodal document analysis with Vision-Language embeddings, "
        "hybrid RAG search, knowledge-graph extraction, and recursive "
        "LLM reasoning.  Uses the provider hierarchy system by default."
    ),
    lifespan=app_lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════

def _effective_provider_model(
    settings: MCPSettings,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Resolve the effective provider and model.

    Per-call overrides beat settings.json which beats hierarchy defaults.
    """
    base_provider, base_model = settings.resolve_provider_model()

    provider = provider_override or base_provider
    model = model_override or base_model

    # Resolve "auto" via hierarchy
    if provider == "auto":
        provider = resolve_auto_provider()

    return provider, model


def _build_rlm(
    provider: str,
    model: Optional[str],
    settings: MCPSettings,
) -> VLRAGGraphRLM:
    """Construct a VLRAGGraphRLM instance with resolved settings."""
    kwargs: dict[str, Any] = {
        "provider": provider,
        "temperature": settings.temperature,
        "max_depth": settings.max_depth,
        "max_iterations": settings.max_iterations,
    }
    if model:
        kwargs["model"] = model
    return VLRAGGraphRLM(**kwargs)


def _get_context_budget(provider: str) -> int:
    return _PROVIDER_BUDGETS.get(provider, 32000)


def _get_settings(ctx: Context) -> MCPSettings:
    """Extract MCPSettings from the lifespan context."""
    try:
        return ctx.request_context.lifespan_context.settings
    except (AttributeError, TypeError):
        # Fallback: reload settings if lifespan context unavailable
        return load_settings()


# ═══════════════════════════════════════════════════════════════════════════
# MCP Tools
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def query_document(
    ctx: Context,
    input_path: str,
    query: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_depth: Optional[int] = None,
    max_iterations: Optional[int] = None,
) -> str:
    """Query a document or folder using the full VL-RAG-Graph-RLM pipeline.

    Processes the document(s) at input_path, builds embeddings and a
    knowledge graph, then answers the query using hybrid retrieval
    (dense + keyword search with RRF fusion), Qwen3-VL reranking,
    and recursive LLM reasoning.

    By default uses the provider hierarchy system to pick the best
    available LLM provider automatically.

    Args:
        input_path: Path to a file (.pptx, .pdf, .txt, .md) or folder.
        query: The question to answer about the document(s).
        provider: Override LLM provider (default: auto/hierarchy).
        model: Override LLM model name.
        max_depth: Override max RLM recursion depth.
        max_iterations: Override max RLM iterations per call.

    Returns:
        The LLM's answer with source attribution.
    """
    settings = _get_settings(ctx)
    eff_provider, eff_model = _effective_provider_model(settings, provider, model)
    depth = max_depth or settings.max_depth
    iterations = max_iterations or settings.max_iterations

    # Import heavy modules lazily
    from vrlmrag import DocumentProcessor, _run_vl_rag_query, _load_knowledge_graph, _save_knowledge_graph, _merge_knowledge_graphs

    from vl_rag_graph_rlm.rag import CompositeReranker, ReciprocalRankFusion

    try:
        from vl_rag_graph_rlm.rag.qwen3vl import (
            create_qwen3vl_embedder,
            create_qwen3vl_reranker,
        )
        from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore

        has_qwen3vl = True
    except ImportError:
        has_qwen3vl = False

    context_budget = _get_context_budget(eff_provider)

    # Process documents
    processor = DocumentProcessor()
    documents = processor.process_path(input_path)
    if isinstance(documents, dict):
        documents = [documents]

    all_chunks: list[dict] = []
    for doc in documents:
        all_chunks.extend(doc.get("chunks", []))

    # Persistence directory
    p = Path(input_path)
    store_dir = (p.parent if p.is_file() else p) / ".vrlmrag_store"
    store_dir.mkdir(parents=True, exist_ok=True)

    # Embeddings + reranker
    store = None
    reranker_vl = None
    if has_qwen3vl:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedder = create_qwen3vl_embedder(device=device)
        store = MultimodalVectorStore(
            embedding_provider=embedder,
            storage_path=str(store_dir / "embeddings.json"),
        )
        for chunk in all_chunks:
            content = chunk.get("content", "")
            if content.strip() and not store.content_exists(content):
                metadata = {"type": chunk.get("type", "text")}
                if "slide" in chunk:
                    metadata["slide"] = chunk["slide"]
                store.add_text(content=content, metadata=metadata, instruction=_DOCUMENT_INSTRUCTION)

        # Embed images
        for doc in documents:
            for img_info in doc.get("image_data", []):
                try:
                    temp_path = f"/tmp/vrlmrag_{img_info['filename']}"
                    with open(temp_path, "wb") as f:
                        f.write(img_info["blob"])
                    store.add_image(
                        image_path=temp_path,
                        description=f"Image from slide {img_info['slide']}",
                        metadata={"type": "image", "slide": img_info["slide"]},
                        instruction=_DOCUMENT_INSTRUCTION,
                    )
                except Exception:
                    pass

        reranker_vl = create_qwen3vl_reranker(device=device)

    # RLM
    rlm = _build_rlm(eff_provider, eff_model, settings)
    fallback_hierarchy = get_available_providers() if (provider or settings.provider) == "auto" else None

    # Knowledge graph
    kg_file = store_dir / "knowledge_graph.md"
    knowledge_graph = _load_knowledge_graph(kg_file)
    if not knowledge_graph and all_chunks:
        kg_limit = min(context_budget, 25000)
        kg_context = "\n\n".join(d.get("content", "")[:2000] for d in documents)
        try:
            kg_result = rlm.completion(_KG_EXTRACTION_PROMPT, kg_context[:kg_limit])
            knowledge_graph = kg_result.response
            _save_knowledge_graph(kg_file, knowledge_graph)
        except Exception:
            knowledge_graph = ""

    # Run query
    rrf = ReciprocalRankFusion(k=60)
    fallback_reranker = CompositeReranker()

    result = _run_vl_rag_query(
        query,
        store=store,
        reranker_vl=reranker_vl,
        rrf=rrf,
        fallback_reranker=fallback_reranker,
        all_chunks=all_chunks,
        knowledge_graph=knowledge_graph,
        context_budget=context_budget,
        rlm=rlm,
        fallback_hierarchy=fallback_hierarchy,
        provider=eff_provider,
        resolved_model=eff_model,
        max_depth=depth,
        max_iterations=iterations,
        verbose=False,
    )

    # Format response
    parts = [result["response"]]
    if result.get("time"):
        parts.append(f"\n---\n*Completed in {result['time']:.2f}s via {eff_provider}*")
    if result.get("sources"):
        parts.append(f"*{len(result['sources'])} sources retrieved*")
    return "\n".join(parts)


@mcp.tool()
async def analyze_document(
    ctx: Context,
    input_path: str,
    query: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    output_path: Optional[str] = None,
    max_depth: Optional[int] = None,
    max_iterations: Optional[int] = None,
) -> str:
    """Run a full VL-RAG-Graph-RLM analysis on a document or folder.

    This is the comprehensive 6-pillar pipeline:
      1. Document intake (PPTX, PDF, TXT, MD)
      2. Qwen3-VL multimodal embedding
      3. Hybrid search (dense + keyword + RRF fusion)
      4. Qwen3-VL cross-attention reranking
      5. Knowledge graph extraction
      6. Recursive LLM reasoning → markdown report

    By default uses the provider hierarchy system.

    Args:
        input_path: Path to file or folder to analyze.
        query: Custom query (default: auto-generated summary queries).
        provider: Override LLM provider (default: auto/hierarchy).
        model: Override LLM model name.
        output_path: Save markdown report to this path.
        max_depth: Override max RLM recursion depth.
        max_iterations: Override max RLM iterations per call.

    Returns:
        Markdown analysis report.
    """
    settings = _get_settings(ctx)
    eff_provider, eff_model = _effective_provider_model(settings, provider, model)
    depth = max_depth or settings.max_depth
    iterations = max_iterations or settings.max_iterations

    from vrlmrag import run_analysis

    results = run_analysis(
        provider=eff_provider,
        input_path=input_path,
        query=query,
        output=output_path,
        model=eff_model,
        max_depth=depth,
        max_iterations=iterations,
    )

    # Build a concise summary
    parts = [
        f"# Analysis Complete",
        f"",
        f"- **Provider:** {results.get('provider', 'N/A')}",
        f"- **Model:** {results.get('model', 'N/A')}",
        f"- **Documents:** {results.get('document_count', 0)}",
        f"- **Chunks:** {results.get('total_chunks', 0)}",
        f"- **Embedded:** {results.get('embedded_count', 0)}",
        f"- **Time:** {results.get('execution_time', 0):.2f}s",
        "",
    ]

    for qr in results.get("queries", []):
        parts.append(f"## Query: {qr['query']}")
        parts.append("")
        parts.append(qr["response"])
        parts.append("")

    if output_path:
        parts.append(f"*Report saved to: {output_path}*")

    return "\n".join(parts)


async def collection_add(
    ctx: Context,
    collection_name: str,
    input_path: str,
    description: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Add documents to a named persistent collection.

    Creates the collection if it does not exist.  Embeds content with
    Qwen3-VL, builds/merges the knowledge graph, and persists everything.

    Collections are reusable knowledge stores that can be queried later
    without re-processing the source documents.

    Args:
        collection_name: Name for the collection (e.g. "research", "code-docs").
        input_path: Path to file or folder to ingest.
        description: Optional description for a new collection.
        provider: Override LLM provider (default: auto/hierarchy).
        model: Override LLM model name.

    Returns:
        Summary of documents added.
    """
    settings = _get_settings(ctx)
    eff_provider, eff_model = _effective_provider_model(settings, provider, model)

    from vrlmrag import run_collection_add

    run_collection_add(
        collection_names=[collection_name],
        input_path=input_path,
        provider=eff_provider,
        model=eff_model,
        max_depth=settings.max_depth,
        max_iterations=settings.max_iterations,
        description=description,
    )

    meta = load_collection_meta(collection_name)
    return (
        f"Collection '{meta.get('display_name', collection_name)}' updated.\n"
        f"- Documents: {meta.get('document_count', 0)}\n"
        f"- Chunks: {meta.get('chunk_count', 0)}\n"
        f"- Sources: {len(meta.get('sources', []))}"
    )


async def collection_query(
    ctx: Context,
    query: str,
    collection_names: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Query one or more named collections.

    Blends the embeddings and knowledge graphs from all specified
    collections, then runs the full VL-RAG retrieval pipeline to
    answer the query.  Multiple collections can be blended for
    cross-domain reasoning.

    By default uses the provider hierarchy system.

    Args:
        query: The question to answer.
        collection_names: List of collection names to query (e.g. ["research", "code"]).
        provider: Override LLM provider (default: auto/hierarchy).
        model: Override LLM model name.
        output_path: Optional path to save the response as markdown.

    Returns:
        The LLM's answer with source attribution.
    """
    settings = _get_settings(ctx)
    eff_provider, eff_model = _effective_provider_model(settings, provider, model)

    from vrlmrag import run_collection_query as _run_cq

    # Capture output — run_collection_query prints to stdout
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _run_cq(
            collection_names=collection_names,
            query=query,
            provider=eff_provider,
            model=eff_model,
            max_depth=settings.max_depth,
            max_iterations=settings.max_iterations,
            output=output_path,
        )

    return buf.getvalue()


async def collection_list(ctx: Context) -> str:
    """List all available named collections.

    Returns a formatted table of collections with document counts,
    chunk counts, source counts, and last-updated timestamps.
    """
    collections = _list_collections_meta()
    if not collections:
        return "No collections found. Create one with the collection_add tool."

    lines = [
        f"{'Name':<25} {'Docs':>6} {'Chunks':>8} {'Sources':>8}  Updated",
        "-" * 78,
    ]
    for meta in collections:
        name = meta.get("display_name", meta["name"])
        docs = meta.get("document_count", 0)
        chunks = meta.get("chunk_count", 0)
        sources = len(meta.get("sources", []))
        updated = meta.get("updated", "?")[:19]
        lines.append(f"{name:<25} {docs:>6} {chunks:>8} {sources:>8}  {updated}")
    return "\n".join(lines)


async def collection_info(ctx: Context, collection_name: str) -> str:
    """Get detailed information about a specific collection.

    Args:
        collection_name: Name of the collection to inspect.

    Returns:
        Detailed metadata including sources, embedding count, and KG size.
    """
    if not collection_exists(collection_name):
        return f"Error: Collection '{collection_name}' does not exist."

    meta = load_collection_meta(collection_name)
    slug = meta["name"]

    lines = [
        f"# Collection: {meta.get('display_name', slug)}",
        f"",
        f"- **Slug:** {slug}",
        f"- **Description:** {meta.get('description') or '(none)'}",
        f"- **Created:** {meta.get('created', '?')[:19]}",
        f"- **Updated:** {meta.get('updated', '?')[:19]}",
        f"- **Documents:** {meta.get('document_count', 0)}",
        f"- **Chunks:** {meta.get('chunk_count', 0)}",
        f"- **Directory:** {_collection_dir(slug)}",
    ]

    # Embedding count
    emb_file = Path(collection_embeddings_path(slug))
    if emb_file.exists():
        try:
            data = json.loads(emb_file.read_text(encoding="utf-8"))
            lines.append(f"- **Embeddings:** {len(data.get('documents', {}))}")
        except Exception:
            lines.append("- **Embeddings:** (file exists, could not parse)")
    else:
        lines.append("- **Embeddings:** 0")

    # KG size
    kg = collection_load_kg(slug)
    lines.append(f"- **KG size:** {len(kg):,} chars" if kg else "- **KG size:** 0 chars")

    # Sources
    sources = meta.get("sources", [])
    if sources:
        lines.append(f"\n## Sources ({len(sources)})")
        for src in sources:
            lines.append(
                f"- `{src['path']}` — {src['documents']} docs, "
                f"{src['chunks']} chunks ({src['added'][:19]})"
            )

    return "\n".join(lines)


async def collection_delete_tool(ctx: Context, collection_name: str) -> str:
    """Delete a named collection and all its data.

    This permanently removes the collection's embeddings, knowledge graph,
    and metadata.  This action cannot be undone.

    Args:
        collection_name: Name of the collection to delete.

    Returns:
        Confirmation message.
    """
    if delete_collection(collection_name):
        return f"Deleted collection: {collection_name}"
    return f"Collection not found: {collection_name}"


# ---------------------------------------------------------------------------
# Conditionally register collection tools at module level.
# This MUST happen here (not in lifespan) so the tool list is correct
# before the MCP protocol advertises tools to the client.
# ---------------------------------------------------------------------------
if _SETTINGS.collections_enabled:
    mcp.tool()(collection_add)
    mcp.tool()(collection_query)
    mcp.tool()(collection_list)
    mcp.tool()(collection_info)
    mcp.tool()(collection_delete_tool)
    logger.info("Collection tools registered (5 tools)")
else:
    logger.info("Collection tools disabled — set VRLMRAG_COLLECTIONS=true to enable")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
