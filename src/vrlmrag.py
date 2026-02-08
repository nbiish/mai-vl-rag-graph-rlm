#!/usr/bin/env python3
"""
vrlmrag — VL-RAG-Graph-RLM CLI

Full multimodal document analysis pipeline:
  1. VL:       Qwen3-VL multimodal embeddings (text + images)
  2. RAG:      Hybrid search (dense + keyword) with RRF fusion
  3. Reranker:  Qwen3-VL cross-attention reranking
  4. Graph:    Knowledge graph extraction via RLM
  5. RLM:      Recursive Language Model with REPL sandbox
  6. Report:   Markdown report generation

Usage:
    vrlmrag document.pptx                          # auto: uses provider hierarchy
    vrlmrag --provider sambanova document.pptx     # explicit provider
    vrlmrag --provider nebius doc.pdf -o report.md  # with output file
    vrlmrag ./folder -q "Summarize key concepts"   # auto + custom query
    vrlmrag --show-hierarchy                       # see fallback order
    vrlmrag --list-providers                       # see all providers

Provider Hierarchy:
    When --provider is omitted (or set to 'auto'), providers are tried in order:
    sambanova → nebius → groq → cerebras → zai → zenmux → openrouter → ...
    Only providers with valid API keys are attempted. Configurable via
    PROVIDER_HIERARCHY env var in .env.
"""

__version__ = "0.1.0"

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# Load environment variables from .env file in project root
from dotenv import load_dotenv

# Find project root (where .env is located)
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    # Try loading from current directory as fallback
    load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from vl_rag_graph_rlm import VLRAGGraphRLM
from vl_rag_graph_rlm.clients.hierarchy import (
    get_hierarchy,
    get_available_providers,
    resolve_auto_provider,
    PROVIDER_KEY_MAP,
)
from vl_rag_graph_rlm.rag import (
    CompositeReranker,
    SearchResult,
    ReciprocalRankFusion,
    MultiFactorReranker,
)

# Qwen3-VL imports (requires torch, transformers>=5.1.0)
try:
    from vl_rag_graph_rlm.rag.qwen3vl import (
        Qwen3VLEmbeddingProvider,
        Qwen3VLRerankerProvider,
        create_qwen3vl_embedder,
        create_qwen3vl_reranker,
        MultimodalDocument,
    )
    from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore

    HAS_QWEN3VL = True
except ImportError:
    HAS_QWEN3VL = False

# Optional PPTX support
try:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False


class DocumentProcessor:
    """Process various document types for VL-RAG-Graph-RLM."""

    def __init__(self):
        self.supported_extensions = {".txt", ".md", ".pdf", ".pptx", ".docx"}

    def process_path(self, path: str) -> List[dict]:
        """Process a file or folder path."""
        path_obj = Path(path)

        if path_obj.is_file():
            return [self.process_file(path_obj)]
        elif path_obj.is_dir():
            return self.process_folder(path_obj)
        else:
            raise ValueError(f"Path not found: {path}")

    def process_file(self, file_path: Path) -> dict:
        """Process a single file."""
        ext = file_path.suffix.lower()

        if ext == ".pptx" and HAS_PPTX:
            return self._process_pptx(file_path)
        elif ext in {".txt", ".md"}:
            return self._process_text(file_path)
        else:
            return {
                "type": "unsupported",
                "path": str(file_path),
                "content": f"File type {ext} not yet supported",
                "chunks": [],
            }

    def process_folder(self, folder_path: Path) -> List[dict]:
        """Process all supported files in a folder."""
        results = []
        for file_path in folder_path.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_extensions
            ):
                try:
                    results.append(self.process_file(file_path))
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
        return results

    def _process_pptx(self, file_path: Path) -> dict:
        """Extract content from PowerPoint."""
        prs = Presentation(file_path)
        slides = []
        all_text = []
        image_count = 0
        image_data: List[Dict[str, Any]] = []

        for slide_num, slide in enumerate(prs.slides, 1):
            slide_texts = []
            slide_images = []

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())

                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        image = shape.image
                        img_filename = (
                            f"slide_{slide_num}_img_{len(slide_images)}.{image.ext}"
                        )
                        slide_images.append(
                            {
                                "slide": slide_num,
                                "filename": img_filename,
                                "size": (shape.width, shape.height),
                            }
                        )
                        # Save image blob for Qwen3-VL embedding
                        image_data.append(
                            {
                                "slide": slide_num,
                                "blob": image.blob,
                                "ext": image.ext,
                                "filename": img_filename,
                            }
                        )
                        image_count += 1
                    except Exception:
                        pass

            text_content = "\n".join(slide_texts)
            if text_content:
                all_text.append(f"SLIDE {slide_num}:\n{text_content}")

            slides.append(
                {"slide_num": slide_num, "text": text_content, "images": slide_images}
            )

        return {
            "type": "pptx",
            "path": str(file_path),
            "slides": slides,
            "content": "\n\n".join(all_text),
            "image_count": image_count,
            "image_data": image_data,
            "chunks": [
                {"content": slide["text"], "type": "text", "slide": slide["slide_num"]}
                for slide in slides
                if slide["text"]
            ],
        }

    def _process_text(self, file_path: Path) -> dict:
        """Process text/markdown files."""
        content = file_path.read_text(encoding="utf-8")

        # Simple chunking by headers
        chunks = []
        lines = content.split("\n")
        current_chunk = []

        for line in lines:
            if line.strip().startswith("#") and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return {
            "type": "text",
            "path": str(file_path),
            "content": content,
            "chunks": [
                {"content": chunk, "type": "text"}
                for chunk in chunks
                if chunk.strip()
            ],
        }


class MarkdownReportGenerator:
    """Generate markdown reports from VL-RAG-Graph-RLM results."""

    def generate(self, results: dict, output_path: Optional[str] = None) -> str:
        """Generate a markdown report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "# VL-RAG-Graph-RLM Analysis Report",
            "",
            f"**Generated:** {timestamp}",
            f"**Provider:** {results.get('provider', 'N/A')}",
            f"**Model:** {results.get('model', 'N/A')}",
            f"**Embeddings:** {results.get('embedding_model', 'N/A')}",
            f"**Reranker:** {results.get('reranker_model', 'N/A')}",
            "",
            "## Summary",
            "",
            f"- **Documents Processed:** {results.get('document_count', 0)}",
            f"- **Total Chunks:** {results.get('total_chunks', 0)}",
            f"- **Embedded Documents:** {results.get('embedded_count', 0)}",
            f"- **Processing Time:** {results.get('execution_time', 0):.2f}s",
            "",
            "## Knowledge Graph",
            "",
            results.get("knowledge_graph", "No knowledge graph generated"),
            "",
            "## Query Responses",
            "",
        ]

        for query_response in results.get("queries", []):
            lines.extend(
                [
                    f"### Query: {query_response['query']}",
                    "",
                    query_response["response"],
                    "",
                    f"*Time: {query_response.get('time', 0):.2f}s*",
                    "",
                ]
            )

            # Add source info if available
            if query_response.get("sources"):
                lines.append("**Retrieved Sources:**")
                for src in query_response["sources"][:5]:
                    score = src.get("score", 0)
                    content_preview = src.get("content", "")[:100]
                    lines.append(f"- [Score: {score:.2f}] {content_preview}...")
                lines.append("")

        lines.extend(["## Source Documents", ""])

        for doc in results.get("documents", []):
            lines.extend(
                [
                    f"### {Path(doc['path']).name}",
                    "",
                    f"- **Type:** {doc['type']}",
                    f"- **Path:** `{doc['path']}`",
                ]
            )
            if "slide_count" in doc:
                lines.append(f"- **Slides:** {doc['slide_count']}")
            if "image_count" in doc:
                lines.append(f"- **Images:** {doc['image_count']}")
            lines.append("")

        markdown = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(markdown, encoding="utf-8")
            print(f"\nReport saved to: {output_path}")

        return markdown


SUPPORTED_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "sambanova": {
        "env_key": "SAMBANOVA_API_KEY",
        "url": "https://cloud.sambanova.ai",
        "description": "SambaNova Cloud — DeepSeek-V3.2 (200+ tok/s, 128K context, 200K TPD free)",
        "context_budget": 8000,
    },
    "nebius": {
        "env_key": "NEBIUS_API_KEY",
        "url": "https://tokenfactory.nebius.com",
        "description": "Nebius Token Factory — MiniMax-M2.1 (128K context, no daily limits)",
        "context_budget": 100000,
    },
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai",
        "description": "OpenRouter — 200+ models, pay-per-token routing",
        "context_budget": 32000,
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "url": "https://platform.openai.com",
        "description": "OpenAI — GPT-4o-mini (128K context)",
        "context_budget": 32000,
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "url": "https://console.anthropic.com",
        "description": "Anthropic — Claude 3.5 Haiku (200K context)",
        "context_budget": 32000,
    },
    "gemini": {
        "env_key": "GOOGLE_API_KEY",
        "url": "https://makersuite.google.com/app/apikey",
        "description": "Google Gemini — gemini-1.5-flash (1M context)",
        "context_budget": 64000,
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "url": "https://console.groq.com",
        "description": "Groq — Ultra-fast LPU inference (Kimi K2, GPT-OSS-120B, Llama 4)",
        "context_budget": 32000,
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "url": "https://platform.deepseek.com",
        "description": "DeepSeek — DeepSeek-V3 / R1 reasoning",
        "context_budget": 32000,
    },
    "mistral": {
        "env_key": "MISTRAL_API_KEY",
        "url": "https://console.mistral.ai",
        "description": "Mistral AI — mistral-large-latest (128K context)",
        "context_budget": 32000,
    },
    "fireworks": {
        "env_key": "FIREWORKS_API_KEY",
        "url": "https://fireworks.ai",
        "description": "Fireworks AI — Serverless open-source models",
        "context_budget": 32000,
    },
    "together": {
        "env_key": "TOGETHER_API_KEY",
        "url": "https://api.together.ai",
        "description": "Together AI — Open-source model hosting",
        "context_budget": 32000,
    },
    "zenmux": {
        "env_key": "ZENMUX_API_KEY",
        "url": "https://zenmux.ai",
        "description": "ZenMux — Unified API gateway (59+ models, OpenAI/Anthropic protocols)",
        "context_budget": 32000,
    },
    "zai": {
        "env_key": "ZAI_API_KEY",
        "url": "https://open.bigmodel.cn",
        "description": "z.ai (Zhipu AI) — GLM-4.7 (Coding Plan first, then normal API)",
        "context_budget": 32000,
    },
    "azure_openai": {
        "env_key": "AZURE_OPENAI_API_KEY",
        "url": "https://portal.azure.com",
        "description": "Azure OpenAI — Enterprise GPT deployments",
        "context_budget": 32000,
    },
    "cerebras": {
        "env_key": "CEREBRAS_API_KEY",
        "url": "https://cloud.cerebras.ai",
        "description": "Cerebras — Ultra-fast wafer-scale (GLM-4.7, GPT-OSS-120B, Qwen3-235B)",
        "context_budget": 32000,
    },
}


def list_providers() -> None:
    """Print all supported providers with their status."""
    print(f"vrlmrag v{__version__} — Supported Providers\n")
    print(f"{'Provider':<16} {'API Key':<12} {'Description'}")
    print("-" * 80)
    for name, info in SUPPORTED_PROVIDERS.items():
        key_set = "✓ SET" if os.getenv(info["env_key"]) else "✗ unset"
        print(f"{name:<16} {key_set:<12} {info['description']}")
    print()
    print("Set API keys in .env or via environment variables.")
    print("See .env.example for all configuration options.")


def show_hierarchy() -> None:
    """Print the provider hierarchy with availability status."""
    hierarchy = get_hierarchy()
    available = get_available_providers(hierarchy)

    print(f"vrlmrag v{__version__} — Provider Hierarchy\n")
    print("Providers are tried in order. First available provider is used.")
    print("Edit PROVIDER_HIERARCHY in .env to customize the order.\n")
    print(f"{'#':<4} {'Provider':<16} {'Status':<12} {'API Key Env Var'}")
    print("-" * 60)
    for i, provider in enumerate(hierarchy, 1):
        env_key = PROVIDER_KEY_MAP.get(provider, f"{provider.upper()}_API_KEY")
        if provider in available:
            status = "✓ READY"
        else:
            status = "✗ no key"
        print(f"{i:<4} {provider:<16} {status:<12} {env_key}")

    print(f"\nAvailable: {len(available)}/{len(hierarchy)} providers ready")
    if available:
        print(f"Auto mode will use: {available[0]}")
    print(f"\nCustomize: PROVIDER_HIERARCHY={','.join(hierarchy)}")


def run_analysis(
    provider: str,
    input_path: str,
    query: Optional[str] = None,
    output: Optional[str] = None,
    model: Optional[str] = None,
    max_depth: int = 3,
    max_iterations: int = 10,
) -> dict:
    """Run full VL-RAG-Graph-RLM analysis with any supported provider.

    Pipeline (6 pillars):
      1. Document intake (PPTX, TXT, MD)
      2. Qwen3-VL embedding (text + images → unified vector space)
      3. Hybrid search (dense + keyword + RRF fusion)
      4. Qwen3-VL reranking (cross-attention relevance scoring)
      5. Knowledge graph extraction via RLM
      6. Query answering via RLM with retrieved context → markdown report
    """
    # Handle 'auto' provider — resolve from hierarchy
    is_auto = provider == "auto"
    if is_auto:
        try:
            provider = resolve_auto_provider()
            print(f"[auto] Resolved provider from hierarchy: {provider}")
        except RuntimeError as e:
            print(f"Error: {e}")
            print("Run 'vrlmrag --show-hierarchy' to see provider status.")
            sys.exit(1)

    if provider not in SUPPORTED_PROVIDERS:
        print(f"Error: Unknown provider '{provider}'")
        print(f"Supported: {', '.join(SUPPORTED_PROVIDERS.keys())}")
        sys.exit(1)

    prov_info = SUPPORTED_PROVIDERS[provider]
    context_budget = prov_info["context_budget"]

    # Check API key
    api_key = os.getenv(prov_info["env_key"])
    if not api_key:
        print(f"Error: {prov_info['env_key']} not set")
        print(f"Get your API key from: {prov_info['url']}")
        sys.exit(1)

    # Resolve model (env var → explicit arg → provider default)
    resolved_model = model  # explicit --model flag takes priority
    if not resolved_model:
        resolved_model = None  # let VLRAGGraphRLM resolve from env/defaults

    print("=" * 70)
    print(f"vrlmrag v{__version__} — Full VL-RAG-Graph-RLM Pipeline")
    print("=" * 70)
    print(f"Provider:       {provider}")
    print(f"Model:          {resolved_model or '(auto-detect from env/defaults)'}")
    print(f"Context budget: {context_budget:,} chars")
    print(f"Input:          {input_path}")
    print("=" * 70)

    overall_start = time.time()

    # ================================================================
    # Step 1: Document Processing
    # ================================================================
    processor = DocumentProcessor()
    print(f"\n[1/6] Processing documents: {input_path}")

    documents = processor.process_path(input_path)
    if isinstance(documents, dict):
        documents = [documents]

    all_chunks: List[dict] = []
    for doc in documents:
        all_chunks.extend(doc.get("chunks", []))

    print(f"  Processed {len(documents)} document(s), {len(all_chunks)} chunks")

    # ================================================================
    # Step 2: Qwen3-VL Embedding + Vector Store
    # ================================================================
    embedding_model_name = "Qwen/Qwen3-VL-Embedding-2B"
    reranker_model_name = "Qwen/Qwen3-VL-Reranker-2B"
    embedded_count = 0

    # Persistence directory — used for embeddings + knowledge graph in all modes
    _store_path = Path(input_path)
    if _store_path.is_file():
        store_dir = _store_path.parent / ".vrlmrag_store"
    else:
        store_dir = _store_path / ".vrlmrag_store"
    store_dir.mkdir(parents=True, exist_ok=True)

    if HAS_QWEN3VL:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\n[2/6] Loading Qwen3-VL Embedding model ({device})...")
        embedder = create_qwen3vl_embedder(
            model_name=embedding_model_name, device=device
        )
        print(f"  Embedding dim: {embedder.embedding_dim}")

        storage_file = str(store_dir / "embeddings.json")

        store = MultimodalVectorStore(
            embedding_provider=embedder,
            storage_path=storage_file,
        )
        existing_count = len(store.documents)
        if existing_count:
            print(f"  Loaded {existing_count} existing embeddings from store")

        # Embed text chunks (dedup: skips already-embedded content)
        skipped_count = 0
        print(f"\n[3/6] Embedding {len(all_chunks)} chunks with Qwen3-VL...")
        for i, chunk in enumerate(all_chunks):
            content = chunk.get("content", "")
            if not content.strip():
                continue
            if store.content_exists(content):
                skipped_count += 1
                continue
            metadata = {"type": chunk.get("type", "text")}
            if "slide" in chunk:
                metadata["slide"] = chunk["slide"]
            store.add_text(content=content, metadata=metadata)
            embedded_count += 1
            if (embedded_count) % 5 == 0:
                print(f"  Embedded {embedded_count} new chunks so far...")

        # Embed any extracted images (dedup handled inside store)
        for doc in documents:
            for img_info in doc.get("image_data", []):
                try:
                    temp_path = f"/tmp/vrlmrag_{img_info['filename']}"
                    with open(temp_path, "wb") as f:
                        f.write(img_info["blob"])
                    prev_count = len(store.documents)
                    store.add_image(
                        image_path=temp_path,
                        description=f"Image from slide {img_info['slide']}",
                        metadata={
                            "type": "image",
                            "slide": img_info["slide"],
                            "filename": img_info["filename"],
                        },
                    )
                    if len(store.documents) > prev_count:
                        embedded_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"  Warning: Could not embed image {img_info['filename']}: {e}")

        print(f"  New embeddings: {embedded_count} | Skipped (already in store): {skipped_count}")
        print(f"  Total in store: {len(store.documents)} documents")
        print(f"  Store persisted to: {storage_file}")

        # Load reranker
        print("\n[4/6] Loading Qwen3-VL Reranker...")
        reranker_vl = create_qwen3vl_reranker(
            model_name=reranker_model_name, device=device
        )
        print("  Reranker loaded successfully")
    else:
        print("\n[2/6] Qwen3-VL not available — using fallback text reranking")
        print("  Install with: pip install torch transformers qwen-vl-utils torchvision")
        store = None
        reranker_vl = None
        embedding_model_name = "N/A (fallback)"
        reranker_model_name = "N/A (fallback)"

    # ================================================================
    # Step 3: Initialize RLM
    # ================================================================
    rlm_kwargs: Dict[str, Any] = {
        "provider": provider,
        "temperature": 0.0,
        "max_depth": max_depth,
        "max_iterations": max_iterations,
    }
    if resolved_model:
        rlm_kwargs["model"] = resolved_model

    print(f"\n[5/6] Initializing RLM ({provider})...")
    rlm = VLRAGGraphRLM(**rlm_kwargs)
    print(f"  Model: {rlm.model}")

    # If auto mode, set up fallback hierarchy for query errors
    fallback_hierarchy = get_available_providers() if is_auto else None

    # ================================================================
    # Step 4: Build / Extend Knowledge Graph (persistent)
    # ================================================================
    kg_file = store_dir / "knowledge_graph.md"
    knowledge_graph = _load_knowledge_graph(kg_file)
    if knowledge_graph:
        print(f"\n  Loaded existing knowledge graph ({len(knowledge_graph):,} chars)")

    print("\n  Building knowledge graph for new content...")
    kg_char_limit = min(context_budget, 25000)
    kg_doc_limit = max(2000, kg_char_limit // max(len(documents), 1))
    kg_context = "\n\n".join([d.get("content", "")[:kg_doc_limit] for d in documents])
    try:
        kg_result = rlm.completion(
            "Extract key concepts, entities, and relationships from this document.",
            kg_context[:kg_char_limit],
        )
        knowledge_graph = _merge_knowledge_graphs(knowledge_graph, kg_result.response)
        _save_knowledge_graph(kg_file, knowledge_graph)
        print(f"  Knowledge graph persisted ({len(knowledge_graph):,} chars)")
    except Exception as e:
        if not knowledge_graph:
            knowledge_graph = f"Could not build knowledge graph: {e}"
        print(f"  Warning: KG extraction failed: {e}")

    # ================================================================
    # Step 5: Run Queries with Qwen3-VL retrieval
    # ================================================================
    queries_to_run: List[str] = []
    if query:
        queries_to_run = [query]
    else:
        queries_to_run = [
            "What are the main topics covered?",
            "Summarize the key concepts presented.",
        ]

    print(f"\n[6/6] Running {len(queries_to_run)} queries with RAG retrieval...")
    query_results: List[Dict[str, Any]] = []
    rrf = ReciprocalRankFusion(k=60)
    fallback_reranker = CompositeReranker()

    for q in queries_to_run:
        print(f"\n  Query: {q}")
        sources_info: List[Dict[str, Any]] = []

        if store is not None and HAS_QWEN3VL:
            # --- Full Qwen3-VL RAG pipeline ---
            dense_results = store.search(q, top_k=20)
            print(f"    Dense search: {len(dense_results)} results")

            keyword_results = _keyword_search(store, q, top_k=20)
            print(f"    Keyword search: {len(keyword_results)} results")

            if keyword_results:
                fused_results = rrf.fuse(
                    [dense_results, keyword_results], weights=[4.0, 1.0]
                )
            else:
                fused_results = dense_results
            print(f"    After RRF fusion: {len(fused_results)} results")

            if reranker_vl and fused_results:
                docs_for_rerank: List[Dict[str, Any]] = []
                for r in fused_results[:15]:
                    doc = store.get(r.id)
                    if doc:
                        doc_dict: Dict[str, Any] = {"text": doc.content}
                        if doc.image_path:
                            doc_dict["image"] = doc.image_path
                        docs_for_rerank.append(doc_dict)

                reranked_indices = reranker_vl.rerank(
                    query={"text": q}, documents=docs_for_rerank
                )
                reordered = []
                for idx, score in reranked_indices:
                    if idx < len(fused_results):
                        result = fused_results[idx]
                        result.composite_score = score * 100
                        reordered.append(result)
                final_results = reordered[:5]
                print(f"    After Qwen3-VL reranking: {len(final_results)} results")
            else:
                final_results = fused_results[:5]

            context_parts = []
            for i, result in enumerate(final_results):
                doc = store.get(result.id)
                if doc:
                    context_parts.append(
                        f"[Source {i + 1}] (score: {result.composite_score:.1f})\n{doc.content}"
                    )
                    sources_info.append(
                        {
                            "content": doc.content[:150],
                            "score": result.composite_score,
                            "metadata": doc.metadata,
                        }
                    )
            context = "\n\n---\n\n".join(context_parts)
        else:
            # --- Fallback: CompositeReranker without embeddings ---
            search_results = [
                SearchResult(
                    id=i,
                    content=chunk.get("content", ""),
                    metadata={"type": chunk.get("type", "text")},
                    semantic_score=1.0,
                )
                for i, chunk in enumerate(all_chunks)
            ]
            reranked, _ = fallback_reranker.process(
                q, [r.__dict__ for r in search_results]
            )
            reranked.sort(
                key=lambda x: x.get("composite_score", 0), reverse=True
            )
            top_chunks = reranked[:3] if len(reranked) >= 3 else reranked
            context = "\n\n---\n\n".join(
                [c.get("content", "") for c in top_chunks]
            )

        # Prepend knowledge graph context for richer answers
        if knowledge_graph and not knowledge_graph.startswith("Could not"):
            kg_budget = min(5000, context_budget // 4)
            context = f"[Knowledge Graph]\n{knowledge_graph[:kg_budget]}\n\n---\n\n{context}"

        try:
            q_start = time.time()
            result = rlm.completion(q, context[:context_budget])
            elapsed = time.time() - q_start

            query_results.append(
                {
                    "query": q,
                    "response": result.response,
                    "time": elapsed,
                    "sources": sources_info,
                }
            )
            print(f"    RLM answer in {elapsed:.2f}s")
        except Exception as e:
            # If auto mode, try fallback providers for this query
            if fallback_hierarchy and len(fallback_hierarchy) > 1:
                fallback_result = _try_fallback_query(
                    q, context[:context_budget], fallback_hierarchy,
                    provider, resolved_model, max_depth, max_iterations,
                )
                if fallback_result:
                    query_results.append(fallback_result)
                    print(f"    Fallback answer via {fallback_result.get('fallback_provider', '?')}")
                    continue

            query_results.append(
                {"query": q, "response": f"Error: {e}", "time": 0, "sources": []}
            )
            print(f"    Error: {e}")

    # ================================================================
    # Compile Results & Generate Report
    # ================================================================
    total_elapsed = time.time() - overall_start
    results = {
        "provider": provider,
        "model": rlm.model,
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "document_count": len(documents),
        "total_chunks": len(all_chunks),
        "embedded_count": embedded_count,
        "execution_time": total_elapsed,
        "documents": documents,
        "knowledge_graph": knowledge_graph,
        "queries": query_results,
    }

    print("\n" + "=" * 70)
    print("Generating markdown report...")
    print("=" * 70)

    generator = MarkdownReportGenerator()
    report = generator.generate(results, output)

    if not output:
        print("\n" + report)

    return results


# Backward-compatible aliases
def run_sambanova_analysis(
    input_path: str, query: Optional[str] = None, output: Optional[str] = None
) -> dict:
    """Backward-compatible alias for run_analysis(provider='sambanova', ...)."""
    return run_analysis("sambanova", input_path, query=query, output=output)


def run_nebius_analysis(
    input_path: str, query: Optional[str] = None, output: Optional[str] = None
) -> dict:
    """Backward-compatible alias for run_analysis(provider='nebius', ...)."""
    return run_analysis("nebius", input_path, query=query, output=output)


def _try_fallback_query(
    query: str,
    context: str,
    hierarchy: List[str],
    failed_provider: str,
    model: Optional[str],
    max_depth: int,
    max_iterations: int,
) -> Optional[Dict[str, Any]]:
    """Try fallback providers for a failed query.

    Note: model is NOT forwarded to fallback providers because model names
    are provider-specific (e.g., 'DeepSeek-V3.2' on SambaNova vs 'glm-4.7'
    on z.ai). Each fallback provider uses its own default model from env or
    hardcoded defaults.
    """
    for fb_provider in hierarchy:
        if fb_provider == failed_provider:
            continue
        try:
            fb_kwargs: Dict[str, Any] = {
                "provider": fb_provider,
                "temperature": 0.0,
                "max_depth": max_depth,
                "max_iterations": max_iterations,
            }

            fb_rlm = VLRAGGraphRLM(**fb_kwargs)
            q_start = time.time()
            result = fb_rlm.completion(query, context)
            elapsed = time.time() - q_start

            return {
                "query": query,
                "response": result.response,
                "time": elapsed,
                "sources": [],
                "fallback_provider": fb_provider,
            }
        except Exception:
            continue
    return None


def _keyword_search(
    store: "MultimodalVectorStore", query: str, top_k: int = 20
) -> List[SearchResult]:
    """Simple keyword search across the vector store documents."""
    import re

    query_terms = set(re.findall(r"\w+", query.lower()))
    results = []

    for doc in store.documents.values():
        content_lower = doc.content.lower()
        matches = sum(1 for term in query_terms if term in content_lower)

        if matches > 0:
            score = matches / len(query_terms) if query_terms else 0
            results.append(
                SearchResult(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    keyword_score=score,
                    composite_score=score,
                )
            )

    results.sort(key=lambda x: x.keyword_score, reverse=True)
    return results[:top_k]


def _resolve_provider(provider: str) -> tuple[str, bool]:
    """Resolve provider name, handling 'auto' mode.

    Returns:
        (resolved_provider_name, is_auto)
    """
    is_auto = provider == "auto"
    if is_auto:
        try:
            provider = resolve_auto_provider()
        except RuntimeError as e:
            print(f"Error: {e}")
            print("Run 'vrlmrag --show-hierarchy' to see provider status.")
            sys.exit(1)
    if provider not in SUPPORTED_PROVIDERS:
        print(f"Error: Unknown provider '{provider}'")
        print(f"Supported: {', '.join(SUPPORTED_PROVIDERS.keys())}")
        sys.exit(1)
    return provider, is_auto


def _load_knowledge_graph(kg_path: Path) -> str:
    """Load persisted knowledge graph from disk."""
    if kg_path.exists():
        return kg_path.read_text(encoding="utf-8")
    return ""


def _save_knowledge_graph(kg_path: Path, kg_text: str) -> None:
    """Persist knowledge graph to disk."""
    kg_path.parent.mkdir(parents=True, exist_ok=True)
    kg_path.write_text(kg_text, encoding="utf-8")


def _merge_knowledge_graphs(existing: str, new_fragment: str) -> str:
    """Merge a new KG fragment into the existing knowledge graph."""
    if not existing:
        return new_fragment
    if not new_fragment:
        return existing
    return f"{existing}\n\n---\n\n{new_fragment}"


def run_interactive_session(
    provider: str,
    input_path: Optional[str] = None,
    model: Optional[str] = None,
    max_depth: int = 3,
    max_iterations: int = 10,
    store_dir: Optional[str] = None,
) -> None:
    """Run an interactive VL-RAG-Graph-RLM session.

    Loads VL models once at startup and keeps them resident in memory.
    Supports incremental document addition and continuous querying.
    The knowledge graph grows across queries and persists to disk.

    Commands (type at the ``vrlmrag>`` prompt):
        <any text>            — run as a query against loaded documents
        /add <path>           — add more documents (file or folder)
        /kg                   — show the current knowledge graph
        /stats                — show session statistics
        /save [path]          — save report to file
        /help                 — show available commands
        /quit or /exit        — end session
    """
    import json as _json
    import readline  # noqa: F401 — enables arrow-key history in input()

    provider, is_auto = _resolve_provider(provider)
    prov_info = SUPPORTED_PROVIDERS[provider]
    context_budget = prov_info["context_budget"]

    api_key = os.getenv(prov_info["env_key"])
    if not api_key:
        print(f"Error: {prov_info['env_key']} not set")
        print(f"Get your API key from: {prov_info['url']}")
        sys.exit(1)

    resolved_model = model  # explicit --model flag takes priority

    # ----------------------------------------------------------------
    # Determine persistence directory
    # ----------------------------------------------------------------
    if store_dir:
        session_dir = Path(store_dir)
    elif input_path:
        p = Path(input_path)
        session_dir = (p.parent if p.is_file() else p) / ".vrlmrag_store"
    else:
        session_dir = Path.cwd() / ".vrlmrag_store"
    session_dir.mkdir(parents=True, exist_ok=True)
    storage_file = str(session_dir / "embeddings.json")
    kg_file = session_dir / "knowledge_graph.md"

    # ----------------------------------------------------------------
    # Banner
    # ----------------------------------------------------------------
    print("=" * 70)
    print(f"vrlmrag v{__version__} — Interactive Session")
    print("=" * 70)
    print(f"Provider:       {provider}" + (" (auto)" if is_auto else ""))
    print(f"Model:          {resolved_model or '(auto-detect from env/defaults)'}")
    print(f"Context budget: {context_budget:,} chars")
    print(f"Store:          {session_dir}")
    if input_path:
        print(f"Input:          {input_path}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Step 1: Load VL models (once)
    # ----------------------------------------------------------------
    embedding_model_name = "Qwen/Qwen3-VL-Embedding-2B"
    reranker_model_name = "Qwen/Qwen3-VL-Reranker-2B"

    embedder = None
    reranker_vl = None
    store: Optional["MultimodalVectorStore"] = None

    if HAS_QWEN3VL:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\n[init] Loading Qwen3-VL Embedding model ({device})...")
        embedder = create_qwen3vl_embedder(
            model_name=embedding_model_name, device=device
        )
        print(f"  Embedding dim: {embedder.embedding_dim}")

        store = MultimodalVectorStore(
            embedding_provider=embedder,
            storage_path=storage_file,
        )
        existing_docs = len(store.documents)
        if existing_docs:
            print(f"  Loaded {existing_docs} existing documents from store")

        print("[init] Loading Qwen3-VL Reranker...")
        reranker_vl = create_qwen3vl_reranker(
            model_name=reranker_model_name, device=device
        )
        print("  Reranker loaded successfully")
    else:
        print("\n[init] Qwen3-VL not available — using fallback text reranking")
        print("  Install with: pip install torch transformers qwen-vl-utils torchvision")

    # ----------------------------------------------------------------
    # Step 2: Initialize RLM
    # ----------------------------------------------------------------
    rlm_kwargs: Dict[str, Any] = {
        "provider": provider,
        "temperature": 0.0,
        "max_depth": max_depth,
        "max_iterations": max_iterations,
    }
    if resolved_model:
        rlm_kwargs["model"] = resolved_model

    print(f"[init] Initializing RLM ({provider})...")
    rlm = VLRAGGraphRLM(**rlm_kwargs)
    print(f"  Model: {rlm.model}")

    fallback_hierarchy = get_available_providers() if is_auto else None

    # ----------------------------------------------------------------
    # Step 3: Load persisted knowledge graph
    # ----------------------------------------------------------------
    knowledge_graph = _load_knowledge_graph(kg_file)
    if knowledge_graph:
        print(f"[init] Loaded existing knowledge graph ({len(knowledge_graph):,} chars)")

    # ----------------------------------------------------------------
    # Step 4: Process initial documents (if provided)
    # ----------------------------------------------------------------
    processor = DocumentProcessor()
    all_chunks: List[dict] = []
    all_documents: List[dict] = []
    rrf = ReciprocalRankFusion(k=60)
    fallback_reranker = CompositeReranker()
    query_count = 0
    total_query_time = 0.0

    def _ingest_path(path_str: str) -> int:
        """Ingest documents from a path, returning count of new embeddings."""
        nonlocal knowledge_graph
        docs = processor.process_path(path_str)
        if isinstance(docs, dict):
            docs = [docs]

        new_chunks: List[dict] = []
        for doc in docs:
            new_chunks.extend(doc.get("chunks", []))
        all_documents.extend(docs)
        all_chunks.extend(new_chunks)

        embedded = 0
        if store is not None and HAS_QWEN3VL:
            for i, chunk in enumerate(new_chunks):
                content = chunk.get("content", "")
                if not content.strip():
                    continue
                metadata = {"type": chunk.get("type", "text")}
                if "slide" in chunk:
                    metadata["slide"] = chunk["slide"]
                store.add_text(content=content, metadata=metadata)
                embedded += 1

            # Embed images
            for doc in docs:
                for img_info in doc.get("image_data", []):
                    try:
                        temp_path = f"/tmp/vrlmrag_{img_info['filename']}"
                        with open(temp_path, "wb") as f:
                            f.write(img_info["blob"])
                        store.add_image(
                            image_path=temp_path,
                            description=f"Image from slide {img_info['slide']}",
                            metadata={
                                "type": "image",
                                "slide": img_info["slide"],
                                "filename": img_info["filename"],
                            },
                        )
                        embedded += 1
                    except Exception as e:
                        print(f"  Warning: Could not embed image {img_info['filename']}: {e}")

        # Build/extend knowledge graph for new documents
        if new_chunks:
            kg_char_limit = min(context_budget, 25000)
            kg_doc_limit = max(2000, kg_char_limit // max(len(docs), 1))
            kg_context = "\n\n".join(
                [d.get("content", "")[:kg_doc_limit] for d in docs]
            )
            try:
                kg_result = rlm.completion(
                    "Extract key concepts, entities, and relationships from this document.",
                    kg_context[:kg_char_limit],
                )
                knowledge_graph = _merge_knowledge_graphs(
                    knowledge_graph, kg_result.response
                )
                _save_knowledge_graph(kg_file, knowledge_graph)
            except Exception as e:
                print(f"  Warning: Could not extend knowledge graph: {e}")

        print(f"  Processed {len(docs)} doc(s), {len(new_chunks)} chunks, {embedded} embedded")
        return embedded

    if input_path:
        print(f"\n[ingest] Processing: {input_path}")
        _ingest_path(input_path)

    # ----------------------------------------------------------------
    # Step 5: Query helper
    # ----------------------------------------------------------------
    def _run_query(q: str) -> Dict[str, Any]:
        """Run a single query against the loaded store + KG."""
        nonlocal query_count, total_query_time
        sources_info: List[Dict[str, Any]] = []

        if store is not None and HAS_QWEN3VL:
            dense_results = store.search(q, top_k=20)
            keyword_results = _keyword_search(store, q, top_k=20)

            if keyword_results:
                fused_results = rrf.fuse(
                    [dense_results, keyword_results], weights=[4.0, 1.0]
                )
            else:
                fused_results = dense_results

            if reranker_vl and fused_results:
                docs_for_rerank: List[Dict[str, Any]] = []
                for r in fused_results[:15]:
                    doc = store.get(r.id)
                    if doc:
                        doc_dict: Dict[str, Any] = {"text": doc.content}
                        if doc.image_path:
                            doc_dict["image"] = doc.image_path
                        docs_for_rerank.append(doc_dict)

                reranked_indices = reranker_vl.rerank(
                    query={"text": q}, documents=docs_for_rerank
                )
                reordered = []
                for idx, score in reranked_indices:
                    if idx < len(fused_results):
                        result = fused_results[idx]
                        result.composite_score = score * 100
                        reordered.append(result)
                final_results = reordered[:5]
            else:
                final_results = fused_results[:5]

            context_parts = []
            for i, result in enumerate(final_results):
                doc = store.get(result.id)
                if doc:
                    context_parts.append(
                        f"[Source {i + 1}] (score: {result.composite_score:.1f})\n{doc.content}"
                    )
                    sources_info.append(
                        {
                            "content": doc.content[:150],
                            "score": result.composite_score,
                            "metadata": doc.metadata,
                        }
                    )
            context = "\n\n---\n\n".join(context_parts)
        else:
            search_results = [
                SearchResult(
                    id=i,
                    content=chunk.get("content", ""),
                    metadata={"type": chunk.get("type", "text")},
                    semantic_score=1.0,
                )
                for i, chunk in enumerate(all_chunks)
            ]
            reranked, _ = fallback_reranker.process(
                q, [r.__dict__ for r in search_results]
            )
            reranked.sort(
                key=lambda x: x.get("composite_score", 0), reverse=True
            )
            top_chunks = reranked[:3] if len(reranked) >= 3 else reranked
            context = "\n\n---\n\n".join(
                [c.get("content", "") for c in top_chunks]
            )

        # Prepend KG context if available
        if knowledge_graph:
            context = (
                f"[Knowledge Graph]\n{knowledge_graph[:5000]}\n\n---\n\n{context}"
            )

        try:
            q_start = time.time()
            result = rlm.completion(q, context[:context_budget])
            elapsed = time.time() - q_start
            query_count += 1
            total_query_time += elapsed
            return {
                "query": q,
                "response": result.response,
                "time": elapsed,
                "sources": sources_info,
            }
        except Exception as e:
            if fallback_hierarchy and len(fallback_hierarchy) > 1:
                fallback_result = _try_fallback_query(
                    q, context[:context_budget], fallback_hierarchy,
                    provider, resolved_model, max_depth, max_iterations,
                )
                if fallback_result:
                    query_count += 1
                    total_query_time += fallback_result.get("time", 0)
                    return fallback_result
            return {"query": q, "response": f"Error: {e}", "time": 0, "sources": []}

    # ----------------------------------------------------------------
    # Step 6: Interactive REPL
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Interactive session ready. Type queries or commands.")
    print("Commands: /add <path>  /kg  /stats  /save [path]  /help  /quit")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nvrlmrag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[session] Goodbye.")
            break

        if not user_input:
            continue

        # ---- Commands ----
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("[session] Goodbye.")
            break

        if user_input.lower() == "/help":
            print(
                "Commands:\n"
                "  <text>            Query loaded documents\n"
                "  /add <path>       Add more documents (file or folder)\n"
                "  /kg               Show the current knowledge graph\n"
                "  /stats            Show session statistics\n"
                "  /save [path]      Save current KG + session report\n"
                "  /help             Show this help\n"
                "  /quit             End session"
            )
            continue

        if user_input.lower() == "/kg":
            if knowledge_graph:
                print(f"\n--- Knowledge Graph ({len(knowledge_graph):,} chars) ---")
                print(knowledge_graph[:8000])
                if len(knowledge_graph) > 8000:
                    print(f"\n... ({len(knowledge_graph) - 8000:,} more chars)")
            else:
                print("No knowledge graph built yet. Add documents first.")
            continue

        if user_input.lower() == "/stats":
            doc_count = len(store.documents) if store else len(all_chunks)
            print(
                f"\n--- Session Statistics ---\n"
                f"  Provider:           {provider} (model: {rlm.model})\n"
                f"  Documents in store: {doc_count}\n"
                f"  Total chunks:       {len(all_chunks)}\n"
                f"  Knowledge graph:    {len(knowledge_graph):,} chars\n"
                f"  Queries run:        {query_count}\n"
                f"  Total query time:   {total_query_time:.2f}s\n"
                f"  Avg query time:     {(total_query_time / query_count):.2f}s"
                if query_count else
                f"\n--- Session Statistics ---\n"
                f"  Provider:           {provider} (model: {rlm.model})\n"
                f"  Documents in store: {doc_count}\n"
                f"  Total chunks:       {len(all_chunks)}\n"
                f"  Knowledge graph:    {len(knowledge_graph):,} chars\n"
                f"  Queries run:        0\n"
                f"  Total query time:   0.00s"
            )
            continue

        if user_input.lower().startswith("/save"):
            parts = user_input.split(maxsplit=1)
            save_path = parts[1] if len(parts) > 1 else str(session_dir / "session_report.md")
            generator = MarkdownReportGenerator()
            results = {
                "provider": provider,
                "model": rlm.model,
                "embedding_model": embedding_model_name if HAS_QWEN3VL else "N/A",
                "reranker_model": reranker_model_name if HAS_QWEN3VL else "N/A",
                "document_count": len(all_documents),
                "total_chunks": len(all_chunks),
                "embedded_count": len(store.documents) if store else 0,
                "execution_time": total_query_time,
                "documents": all_documents,
                "knowledge_graph": knowledge_graph,
                "queries": [],
            }
            report = generator.generate(results, save_path)
            print(f"  Report saved to: {save_path}")
            continue

        if user_input.lower().startswith("/add"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /add <path>")
                continue
            add_path = parts[1].strip()
            if not Path(add_path).exists():
                print(f"Error: Path not found: {add_path}")
                continue
            print(f"[ingest] Processing: {add_path}")
            _ingest_path(add_path)
            continue

        # ---- Default: treat as query ----
        print(f"  Querying...")
        result = _run_query(user_input)
        print(f"\n{result['response']}")
        if result.get("time"):
            print(f"\n  [{result['time']:.2f}s]")
        if result.get("sources"):
            print(f"  [{len(result['sources'])} sources retrieved]")


def main():
    provider_names = ", ".join(SUPPORTED_PROVIDERS.keys())

    parser = argparse.ArgumentParser(
        prog="vrlmrag",
        description=(
            "vrlmrag — Full VL-RAG-Graph-RLM document analysis pipeline.\n\n"
            "Process documents (PPTX, PDF, TXT, MD) through the complete 6-pillar\n"
            "multimodal pipeline: VL embeddings → RAG → reranking → knowledge graph\n"
            "→ recursive LLM reasoning → markdown report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  vrlmrag presentation.pptx                           # auto mode\n"
            "  vrlmrag --provider sambanova presentation.pptx      # explicit provider\n"
            "  vrlmrag --provider nebius document.pdf -o report.md  # with output\n"
            "  vrlmrag ./docs -q 'Summarize key findings'          # auto + query\n"
            "  vrlmrag --interactive presentation.pptx             # interactive session\n"
            "  vrlmrag -i ./codebase                               # load & query continuously\n"
            "  vrlmrag --show-hierarchy                            # see fallback order\n"
            "  vrlmrag --list-providers                            # see all providers\n"
            "\n"
            "interactive mode:\n"
            "  Loads VL models once, then lets you query continuously and add more\n"
            "  documents without reloading. Knowledge graph persists across queries.\n"
            "  Commands: /add <path>  /kg  /stats  /save  /help  /quit\n"
            "\n"
            "provider hierarchy (auto mode):\n"
            "  When --provider is omitted, providers are tried in PROVIDER_HIERARCHY order.\n"
            "  Default: sambanova → nebius → groq → cerebras → zai → zenmux → ...\n"
            "  Customize in .env: PROVIDER_HIERARCHY=groq,cerebras,openrouter,...\n"
            "\n"
            "backward-compatible aliases:\n"
            "  vrlmrag --samba-nova presentation.pptx\n"
            "  vrlmrag --nebius document.pdf\n"
            f"\nsupported providers: auto, {provider_names}\n"
            "\nSet API keys in .env or via environment variables. See .env.example."
        ),
    )

    parser.add_argument(
        "--version", "-V", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "--list-providers", action="store_true",
        help="List all supported providers and their API key status",
    )

    parser.add_argument(
        "--show-hierarchy", action="store_true",
        help="Show the provider fallback hierarchy and availability",
    )

    parser.add_argument(
        "--provider", "-p",
        metavar="NAME",
        default="auto",
        help=f"LLM provider to use (default: auto — uses hierarchy). Options: auto, {provider_names}",
    )

    parser.add_argument(
        "input", nargs="?", metavar="PATH",
        help="File or folder to process (PPTX, PDF, TXT, MD)",
    )

    parser.add_argument(
        "--query", "-q",
        help="Custom query (default: auto-generated summary queries)",
    )

    parser.add_argument(
        "--output", "-o",
        help="Output markdown report path (default: print to stdout)",
    )

    parser.add_argument(
        "--model", "-m",
        help="Override the default model for the chosen provider",
    )

    parser.add_argument(
        "--max-depth", type=int, default=3,
        help="Maximum RLM recursion depth (default: 3)",
    )

    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Maximum RLM iterations per call (default: 10)",
    )

    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Start an interactive session (load VL models once, query continuously)",
    )

    parser.add_argument(
        "--store-dir",
        help="Directory for persisting embeddings and knowledge graph (default: .vrlmrag_store next to input)",
    )

    # Backward-compatible aliases (hidden from main help)
    parser.add_argument("--samba-nova", metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument("--nebius", metavar="PATH", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Handle --list-providers / --show-hierarchy
    if args.list_providers:
        list_providers()
        return

    if args.show_hierarchy:
        show_hierarchy()
        return

    # Resolve provider and input from args (support old-style flags)
    provider = args.provider
    input_path = args.input

    if args.samba_nova:
        provider = "sambanova"
        input_path = args.samba_nova
    elif args.nebius and not provider:
        provider = "nebius"
        input_path = args.nebius

    # Interactive mode
    if args.interactive:
        run_interactive_session(
            provider=provider,
            input_path=input_path,
            model=args.model,
            max_depth=args.max_depth,
            max_iterations=args.max_iterations,
            store_dir=args.store_dir,
        )
        return

    if not input_path:
        parser.print_help()
        print(f"\nError: input PATH is required.")
        sys.exit(1)

    run_analysis(
        provider=provider,
        input_path=input_path,
        query=args.query,
        output=args.output,
        model=args.model,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
