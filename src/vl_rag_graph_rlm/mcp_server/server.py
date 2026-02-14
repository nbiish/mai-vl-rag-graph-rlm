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

def _resolve_auto_provider() -> str:
    """Local implementation to resolve auto provider without import dependency."""
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


# ---------------------------------------------------------------------------
# Internal imports (after path + env setup)
# ---------------------------------------------------------------------------
from vl_rag_graph_rlm import VLRAGGraphRLM
from vl_rag_graph_rlm.clients.hierarchy import get_available_providers
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
from vl_rag_graph_rlm.local_model_lock import (
    local_model_lock,
    lock_status,
    is_local_provider,
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
    "sambanova": 32000,
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
        provider = _resolve_auto_provider()

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


# ---------------------------------------------------------------------------
# Store persistence helpers — manifest tracks indexed files for smart reuse
# ---------------------------------------------------------------------------
_SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pptx", ".pdf", ".docx",
    ".mp4", ".wav", ".mp3", ".flac", ".m4a",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg",
}


def _resolve_input_path(input_path: Optional[str]) -> Path:
    """Resolve input_path, defaulting to CWD if None or '.'."""
    if not input_path or input_path.strip() in ("", "."):
        return Path.cwd()
    return Path(input_path).resolve()


def _store_dir_for(input_path: Path) -> Path:
    """Compute the .vrlmrag_store directory for a given input path."""
    if input_path.is_file():
        return input_path.parent / ".vrlmrag_store"
    return input_path / ".vrlmrag_store"


def _load_manifest(store_dir: Path) -> dict:
    """Load the file manifest (tracks indexed files + mtimes)."""
    manifest_file = store_dir / "manifest.json"
    if manifest_file.exists():
        try:
            return json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_manifest(store_dir: Path, manifest: dict) -> None:
    """Persist the file manifest."""
    store_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = store_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _scan_files(input_path: Path) -> dict[str, float]:
    """Scan input_path for supported files, returning {path: mtime}."""
    files: dict[str, float] = {}
    if input_path.is_file():
        if input_path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            files[str(input_path)] = input_path.stat().st_mtime
    elif input_path.is_dir():
        for f in sorted(input_path.rglob("*")):
            if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS:
                # Skip hidden dirs and .vrlmrag_store
                parts = f.relative_to(input_path).parts
                if any(p.startswith(".") for p in parts):
                    continue
                files[str(f)] = f.stat().st_mtime
    return files


def _detect_changes(current_files: dict[str, float], manifest: dict) -> tuple[list[str], list[str]]:
    """Compare current files against manifest.

    Returns:
        (new_or_modified, deleted) — lists of file paths
    """
    indexed = manifest.get("files", {})
    new_or_modified = []
    for fpath, mtime in current_files.items():
        prev_mtime = indexed.get(fpath)
        if prev_mtime is None or mtime > prev_mtime:
            new_or_modified.append(fpath)
    deleted = [f for f in indexed if f not in current_files]
    return new_or_modified, deleted


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

    **Persistent vector store:** Embeddings and knowledge graphs are
    saved to a `.vrlmrag_store/` directory next to the input.  On
    subsequent queries the existing store is reloaded automatically —
    only new or modified files are re-processed (tracked via a file
    manifest).  This means you can iterate on the same directory
    without re-embedding everything each time.

    Args:
        input_path: Path to a file (.pptx, .pdf, .txt, .md, .mp4, .wav, .mp3) or folder.
                    Use "." or leave empty to query the current working directory.
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
        from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder
        from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore

        has_qwen3vl = True
    except ImportError:
        has_qwen3vl = False

    try:
        from vl_rag_graph_rlm.rag.api_embedding import create_api_embedder
        has_api_embedding = True
    except ImportError:
        has_api_embedding = False

    try:
        from vl_rag_graph_rlm.rag.flashrank_reranker import create_flashrank_reranker
        has_flashrank = True
    except ImportError:
        has_flashrank = False

    use_api = settings.use_api
    context_budget = _get_context_budget(eff_provider)

    # ── Resolve input path (defaults to CWD) ──────────────────────────
    resolved_path = _resolve_input_path(input_path)
    if not resolved_path.exists():
        return f"Error: Path not found: {resolved_path}"

    # ── Persistence directory ─────────────────────────────────────────
    store_dir = _store_dir_for(resolved_path)
    store_dir.mkdir(parents=True, exist_ok=True)

    # ── Manifest-based change detection ───────────────────────────────
    current_files = _scan_files(resolved_path)
    manifest = _load_manifest(store_dir)
    new_or_modified, deleted = _detect_changes(current_files, manifest)

    embeddings_file = store_dir / "embeddings.json"
    store_exists = embeddings_file.exists()
    needs_processing = bool(new_or_modified) or not store_exists

    logger.info(
        "Store check: exists=%s, new/modified=%d, deleted=%d, needs_processing=%s",
        store_exists, len(new_or_modified), len(deleted), needs_processing,
    )

    # ── Process only new/modified documents ───────────────────────────
    all_chunks: list[dict] = []
    documents: list[dict] = []

    if needs_processing:
        _transcriber = None
        if not use_api:
            try:
                from vl_rag_graph_rlm.rag.parakeet import create_parakeet_transcriber
                _transcriber = create_parakeet_transcriber(
                    cache_dir=str(Path.home() / ".vrlmrag" / "parakeet_cache"),
                )
            except ImportError:
                pass
        processor = DocumentProcessor(
            transcription_provider=_transcriber,
            use_api=use_api,
        )

        if new_or_modified and store_exists:
            # Incremental: only process changed files
            for fpath in new_or_modified:
                result = processor.process_path(fpath)
                if isinstance(result, dict):
                    documents.append(result)
                elif isinstance(result, list):
                    documents.extend(result)
        else:
            # Full processing (first run or no store)
            result = processor.process_path(str(resolved_path))
            if isinstance(result, dict):
                documents = [result]
            elif isinstance(result, list):
                documents = result

        for doc in documents:
            all_chunks.extend(doc.get("chunks", []))

    # ── Embeddings — API mode or local Qwen3-VL ──────────────────────
    store = None
    reranker_vl = None
    _lock_ctx = None

    if use_api and has_api_embedding:
        embedder = create_api_embedder()
        store = MultimodalVectorStore(
            embedding_provider=embedder,
            storage_path=str(embeddings_file),
        )
    elif has_qwen3vl:
        _emb_model = os.getenv("VRLMRAG_LOCAL_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
        _lock_ctx = local_model_lock(
            _emb_model,
            description="MCP query_document",
        )
        _lock_ctx.__enter__()

        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedder = create_qwen3vl_embedder(
            model_name=_emb_model,
            device=device,
        )
        store = MultimodalVectorStore(
            embedding_provider=embedder,
            storage_path=str(embeddings_file),
        )

    try:
        if store is not None and needs_processing:
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

            # Embed video frames from media documents
            for doc in documents:
                for i, frame_path in enumerate(doc.get("frame_paths", [])):
                    try:
                        store.add_image(
                            image_path=frame_path,
                            description=f"Video frame {i+1} from {Path(doc['path']).name}",
                            metadata={"type": "video_frame", "frame_index": i, "source": doc["path"]},
                            instruction=_DOCUMENT_INSTRUCTION,
                        )
                    except Exception:
                        pass

            # Update manifest after successful indexing
            manifest["files"] = {str(k): v for k, v in current_files.items()}
            _save_manifest(store_dir, manifest)

        # Load lightweight FlashRank reranker (~34 MB — coexists with embedder)
        if has_flashrank:
            reranker_vl = create_flashrank_reranker(
                model_name=os.getenv("VRLMRAG_RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2"),
            )
    finally:
        if _lock_ctx is not None:
            _lock_ctx.__exit__(None, None, None)

    # ── RLM ───────────────────────────────────────────────────────────
    rlm = _build_rlm(eff_provider, eff_model, settings)
    fallback_hierarchy = get_available_providers() if (provider or settings.provider) == "auto" else None

    # ── Knowledge graph (load existing or build from new content) ─────
    kg_file = store_dir / "knowledge_graph.md"
    knowledge_graph = _load_knowledge_graph(kg_file)
    if needs_processing and all_chunks:
        kg_limit = min(context_budget, 25000)
        kg_context = "\n\n".join(d.get("content", "")[:2000] for d in documents)
        try:
            kg_result = rlm.completion(_KG_EXTRACTION_PROMPT, kg_context[:kg_limit])
            new_kg = kg_result.response
            if knowledge_graph:
                knowledge_graph = _merge_knowledge_graphs(knowledge_graph, new_kg)
            else:
                knowledge_graph = new_kg
            _save_knowledge_graph(kg_file, knowledge_graph)
        except Exception:
            if not knowledge_graph:
                knowledge_graph = ""

    # ── Collect all chunks for fallback reranking ─────────────────────
    # If we skipped processing (store reuse), we still need chunks for
    # the fallback reranker path.  Reconstruct from the store.
    if not all_chunks and store is not None:
        all_chunks = [
            {"content": doc.content, "type": doc.metadata.get("type", "text")}
            for doc in store.documents.values()
        ]

    # ── Run query ─────────────────────────────────────────────────────
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
    store_info = f"{len(store.documents)} embeddings" if store else "no store"
    reused = "reused" if not needs_processing else "updated"
    parts = [result["response"]]
    if result.get("time"):
        parts.append(f"\n---\n*Completed in {result['time']:.2f}s via {eff_provider} ({store_info}, store {reused})*")
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


@mcp.tool()
async def query_text_document(
    ctx: Context,
    input_path: str,
    query: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_depth: Optional[int] = None,
    max_iterations: Optional[int] = None,
) -> str:
    """Query a document using text-only RAG (no multimodal models).

    Lightweight alternative to query_document that uses the text-only
    Qwen3-Embedding model (~1.2 GB) instead of Qwen3-VL (~4.6 GB).
    Skips image/video embedding — ideal for plain text, markdown, and
    text-heavy PDFs where multimodal features are not needed.

    Uses local embedding (zero API calls) with FlashRank reranking,
    hybrid retrieval, knowledge-graph extraction, and recursive LLM
    reasoning — the same 6-pillar pipeline, just without vision.

    **Persistent vector store:** Reuses existing `.vrlmrag_store/`
    embeddings automatically.  Only new or modified files are re-processed.

    Args:
        input_path: Path to a file (.txt, .md, .pdf) or folder.
                    Use "." or leave empty to query the current working directory.
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
        from vl_rag_graph_rlm.rag.text_embedding import create_text_embedder
        from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore

        has_text_embedding = True
    except ImportError:
        has_text_embedding = False

    try:
        from vl_rag_graph_rlm.rag.flashrank_reranker import create_flashrank_reranker
        has_flashrank = True
    except ImportError:
        has_flashrank = False

    if not has_text_embedding:
        return "Error: text-only embedding not available. Install transformers."

    context_budget = _get_context_budget(eff_provider)

    # ── Resolve input path (defaults to CWD) ──────────────────────────
    resolved_path = _resolve_input_path(input_path)
    if not resolved_path.exists():
        return f"Error: Path not found: {resolved_path}"

    # ── Persistence directory ─────────────────────────────────────────
    store_dir = _store_dir_for(resolved_path)
    store_dir.mkdir(parents=True, exist_ok=True)

    # ── Manifest-based change detection ───────────────────────────────
    current_files = _scan_files(resolved_path)
    manifest = _load_manifest(store_dir)
    new_or_modified, deleted = _detect_changes(current_files, manifest)

    embeddings_file = store_dir / "embeddings_text.json"
    store_exists = embeddings_file.exists()
    needs_processing = bool(new_or_modified) or not store_exists

    logger.info(
        "Text store check: exists=%s, new/modified=%d, deleted=%d, needs_processing=%s",
        store_exists, len(new_or_modified), len(deleted), needs_processing,
    )

    # ── Process only new/modified documents ───────────────────────────
    all_chunks: list[dict] = []
    documents: list[dict] = []

    if needs_processing:
        processor = DocumentProcessor()

        if new_or_modified and store_exists:
            for fpath in new_or_modified:
                result = processor.process_path(fpath)
                if isinstance(result, dict):
                    documents.append(result)
                elif isinstance(result, list):
                    documents.extend(result)
        else:
            result = processor.process_path(str(resolved_path))
            if isinstance(result, dict):
                documents = [result]
            elif isinstance(result, list):
                documents = result

        for doc in documents:
            all_chunks.extend(doc.get("chunks", []))

    if not all_chunks and not store_exists:
        return f"No text content found in: {resolved_path}"

    # ── Text-only embedding — acquire cross-process lock ──────────────
    text_model = os.getenv("VRLMRAG_TEXT_ONLY_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    reranker_vl = None

    with local_model_lock(text_model, description="MCP query_text_document"):
        embedder = create_text_embedder(model_name=text_model)
        store = MultimodalVectorStore(
            embedding_provider=embedder,
            storage_path=str(embeddings_file),
        )

        if needs_processing:
            for chunk in all_chunks:
                content = chunk.get("content", "")
                if content.strip() and not store.content_exists(content):
                    metadata = {"type": chunk.get("type", "text")}
                    store.add_text(
                        content,
                        metadata=metadata,
                        instruction=_DOCUMENT_INSTRUCTION,
                    )

            # Update manifest after successful indexing
            manifest["files"] = {str(k): v for k, v in current_files.items()}
            _save_manifest(store_dir, manifest)

        # FlashRank reranker (~34 MB — coexists with embedder)
        if has_flashrank:
            reranker_vl = create_flashrank_reranker(
                model_name=os.getenv("VRLMRAG_RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2"),
            )
    # Lock released — RLM calls below are API-based

    # ── Reconstruct chunks from store if we skipped processing ────────
    if not all_chunks and store is not None:
        all_chunks = [
            {"content": doc.content, "type": doc.metadata.get("type", "text")}
            for doc in store.documents.values()
        ]

    # ── RLM ───────────────────────────────────────────────────────────
    rlm = _build_rlm(eff_provider, eff_model, settings)
    fallback_hierarchy = get_available_providers() if (provider or settings.provider) == "auto" else None

    # ── Knowledge graph ───────────────────────────────────────────────
    kg_file = store_dir / "knowledge_graph.md"
    knowledge_graph = _load_knowledge_graph(kg_file)
    if needs_processing and all_chunks and not knowledge_graph:
        kg_limit = min(context_budget, 25000)
        kg_text = "\n\n".join(c.get("content", "") for c in all_chunks)[:kg_limit]
        try:
            kg_result = rlm.completion(_KG_EXTRACTION_PROMPT, kg_text[:kg_limit])
            knowledge_graph = kg_result.response
            _save_knowledge_graph(kg_file, knowledge_graph)
        except Exception:
            knowledge_graph = ""

    # ── Run query ─────────────────────────────────────────────────────
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

    store_info = f"{len(store.documents)} embeddings" if store else "no store"
    reused = "reused" if not needs_processing else "updated"
    parts = [result["response"]]
    if result.get("time"):
        parts.append(f"\n---\n*Completed in {result['time']:.2f}s via {eff_provider} ({store_info}, store {reused})*")
    if result.get("sources"):
        parts.append(f"*{len(result['sources'])} sources retrieved*")
    return "\n".join(parts)


@mcp.tool()
async def run_text_only_cli(
    ctx: Context,
    input_path: str,
    query: Optional[str] = None,
    output_path: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_depth: Optional[int] = None,
    max_iterations: Optional[int] = None,
) -> str:
    """Run the vrlmrag CLI in text-only mode via subprocess.

    This executes the exact same CLI command you would run locally:
        vrlmrag --text-only <input_path> -q "<query>"

    Captures stdout/stderr and returns the full CLI output.  Useful when
    you want the authentic CLI experience (progress bars, timing info,
    etc.) rather than the internal API.

    Uses lightweight text-only embeddings (~1.2 GB RAM, fully offline).
    Skips image/video processing — best for .txt, .md, text-heavy PDFs.

    Args:
        input_path: Path to file or folder to analyze.
        query: Optional query string (default: auto-generated summary).
        output_path: Optional path to save markdown report.
        provider: LLM provider (default: auto/hierarchy).
        model: Override LLM model name.
        max_depth: Max RLM recursion depth (default: 3).
        max_iterations: Max RLM iterations (default: 10).

    Returns:
        Full CLI stdout/stderr output as a string.
    """
    import asyncio
    import shlex
    from pathlib import Path

    settings = _get_settings(ctx)
    eff_provider, eff_model = _effective_provider_model(settings, provider, model)

    # Resolve the vrlmrag CLI entry point
    codebase_root = Path(__file__).parent.parent.parent.parent.parent.resolve()
    vrlmrag_py = codebase_root / "src" / "vrlmrag.py"

    if not vrlmrag_py.exists():
        return f"Error: vrlmrag.py not found at expected path: {vrlmrag_py}"

    # Build command line
    cmd_parts = [
        sys.executable,
        str(vrlmrag_py),
        "--text-only",
        shlex.quote(input_path),
        "--provider", shlex.quote(eff_provider),
    ]

    if query:
        cmd_parts.extend(["--query", shlex.quote(query)])
    if output_path:
        cmd_parts.extend(["--output", shlex.quote(output_path)])
    if eff_model:
        cmd_parts.extend(["--model", shlex.quote(eff_model)])
    if max_depth is not None:
        cmd_parts.extend(["--max-depth", str(max_depth)])
    if max_iterations is not None:
        cmd_parts.extend(["--max-iterations", str(max_iterations)])

    cmd_str = " ".join(cmd_parts)
    logger.info("Running CLI: %s", cmd_str)

    # Run subprocess and capture output
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(codebase_root),
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            return f"CLI exited with code {proc.returncode}:\n{output}"
        return output
    except Exception as exc:
        return f"Error running CLI: {exc}"


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


async def collection_reindex(
    ctx: Context,
    collection_name: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Re-index a collection — re-embed all documents with current embedding model.

    Clears existing embeddings and re-processes all source documents.
    Useful when upgrading to a new embedding model or when the existing
    index may be corrupted.

    Args:
        collection_name: Name of the collection to reindex.
        provider: Override LLM provider for KG extraction (default: auto/hierarchy).
        model: Override LLM model name.

    Returns:
        Summary of reindexing operation.
    """
    if not collection_exists(collection_name):
        return f"Error: Collection '{collection_name}' does not exist."

    meta = load_collection_meta(collection_name)
    slug = meta["name"]
    coll_dir = _collection_dir(slug)
    sources = meta.get("sources", [])

    if not sources:
        return f"Collection '{collection_name}' has no recorded sources — nothing to reindex."

    # Clear embeddings
    _emb_file = coll_dir / "embeddings.json"
    _text_emb_file = coll_dir / "embeddings_text.json"
    cleared = []
    if _emb_file.exists():
        _emb_file.unlink()
        cleared.append("embeddings")
    if _text_emb_file.exists():
        _text_emb_file.unlink()
        cleared.append("text embeddings")

    settings = _get_settings(ctx)
    eff_provider, eff_model = _effective_provider_model(settings, provider, model)

    from vrlmrag import run_collection_add

    # Re-add all sources
    readded = 0
    for src in sources:
        src_path = src["path"]
        if Path(src_path).exists():
            run_collection_add(
                collection_names=[collection_name],
                input_path=src_path,
                provider=eff_provider,
                model=eff_model,
                max_depth=settings.max_depth,
                max_iterations=settings.max_iterations,
                description=meta.get("description", ""),
            )
            readded += 1

    meta = load_collection_meta(collection_name)
    return (
        f"Collection '{collection_name}' reindexed.\n"
        f"- Cleared: {', '.join(cleared) if cleared else 'none'}\n"
        f"- Re-added sources: {readded}/{len(sources)}\n"
        f"- Current documents: {meta.get('document_count', 0)}\n"
        f"- Current chunks: {meta.get('chunk_count', 0)}"
    )


async def collection_rebuild_kg(
    ctx: Context,
    collection_name: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Rebuild knowledge graph for a collection — regenerate KG with current RLM.

    Clears the existing knowledge graph and re-extracts entities/relationships
    from all documents using the current RLM model. Useful when the KG model
    is upgraded or when the existing KG may be incomplete.

    Args:
        collection_name: Name of the collection to rebuild KG for.
        provider: Override LLM provider (default: auto/hierarchy).
        model: Override LLM model name.

    Returns:
        Summary of KG rebuild operation.
    """
    if not collection_exists(collection_name):
        return f"Error: Collection '{collection_name}' does not exist."

    meta = load_collection_meta(collection_name)
    slug = meta["name"]
    coll_dir = _collection_dir(slug)

    # Clear existing KG
    _kg_file = coll_dir / "knowledge_graph.md"
    had_kg = _kg_file.exists()
    if had_kg:
        _kg_file.unlink()

    # Rebuild KG by re-adding all sources (which triggers KG extraction)
    settings = _get_settings(ctx)
    eff_provider, eff_model = _effective_provider_model(settings, provider, model)

    from vrlmrag import run_collection_add

    sources = meta.get("sources", [])
    readded = 0
    for src in sources:
        src_path = src["path"]
        if Path(src_path).exists():
            run_collection_add(
                collection_names=[collection_name],
                input_path=src_path,
                provider=eff_provider,
                model=eff_model,
                max_depth=settings.max_depth,
                max_iterations=settings.max_iterations,
                description=meta.get("description", ""),
            )
            readded += 1

    # Check new KG size
    kg = collection_load_kg(slug)
    return (
        f"Collection '{collection_name}' KG rebuilt.\n"
        f"- Previous KG existed: {had_kg}\n"
        f"- Re-processed sources: {readded}/{len(sources)}\n"
        f"- New KG size: {len(kg):,} chars"
    )


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
    mcp.tool()(collection_reindex)
    mcp.tool()(collection_rebuild_kg)
    logger.info("Collection tools registered (7 tools)")
else:
    logger.info("Collection tools disabled — set VRLMRAG_COLLECTIONS=true to enable")


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic tools
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def local_model_lock_status(ctx: Context) -> str:
    """Check the status of the cross-process local model lock.

    Shows whether a local model is currently loaded in RAM, which
    process holds the lock, and whether that process is still alive.
    Useful for diagnosing why a query might be waiting.

    Returns:
        Lock status including holder PID, model, and process info.
    """
    status = lock_status()
    if not status["locked"]:
        return (
            "Local model lock: **FREE**\n"
            "No local model is currently loaded in RAM.\n"
            f"This process holds lock: {status['this_process_holds']}"
        )

    return (
        f"Local model lock: **HELD**\n"
        f"- **Holder PID:** {status['holder_pid']}\n"
        f"- **Holder alive:** {status['holder_alive']}\n"
        f"- **Model:** {status['model_id']}\n"
        f"- **Since:** {status['acquired_at']}\n"
        f"- **Process:** {status['process_name']}\n"
        f"- **Description:** {status['description']}\n"
        f"- **This process holds:** {status['this_process_holds']}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
