#!/usr/bin/env python3
"""
VRLMRAG - VL-RAG-Graph-RLM CLI Tool

Process documents using the full VL-RAG-Graph-RLM architecture:
- VL: Vision-Language embeddings (Qwen3-VL-Embedding-2B)
- RAG: Retrieval-Augmented Generation with hybrid search + RRF
- Graph: Knowledge graph construction
- RLM: Recursive Language Model (DeepSeek-V3.1 via SambaNova)

Usage:
    vrlmrag --samba-nova <file_or_folder>
    vrlmrag --samba-nova document.pdf --output report.md
    vrlmrag --samba-nova ./folder --query "Summarize key concepts"
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from vl_rag_graph_rlm import VLRAGGraphRLM
from vl_rag_graph_rlm.rag import (
    CompositeReranker,
    SearchResult,
    ReciprocalRankFusion,
    MultiFactorReranker,
)

# Qwen3-VL imports (requires torch, transformers)
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


def run_sambanova_analysis(
    input_path: str, query: Optional[str] = None, output: Optional[str] = None
):
    """Run full VL-RAG-Graph-RLM analysis with SambaNova + Qwen3-VL.

    Pipeline:
    1. Document intake (PPTX, TXT, MD)
    2. Qwen3-VL embedding (text + images in unified vector space)
    3. Hybrid search (dense cosine + keyword + RRF fusion)
    4. Qwen3-VL reranking (cross-attention relevance scoring)
    5. Knowledge graph extraction via RLM
    6. Query answering via RLM with retrieved context
    7. Markdown report generation
    """

    # Check API key
    api_key = os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        print("Error: SAMBANOVA_API_KEY not set")
        print("Get your API key from: https://cloud.sambanova.ai")
        sys.exit(1)

    print("=" * 70)
    print("VL-RAG-Graph-RLM: SambaNova + Qwen3-VL Full Pipeline")
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

    all_chunks = []
    for doc in documents:
        all_chunks.extend(doc.get("chunks", []))

    print(f"  Processed {len(documents)} document(s), {len(all_chunks)} chunks")

    # ================================================================
    # Step 2: Qwen3-VL Embedding + Vector Store
    # ================================================================
    embedding_model_name = "Qwen/Qwen3-VL-Embedding-2B"
    reranker_model_name = "Qwen/Qwen3-VL-Reranker-2B"
    embedded_count = 0

    if HAS_QWEN3VL:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\n[2/6] Loading Qwen3-VL Embedding model ({device})...")
        embedder = create_qwen3vl_embedder(
            model_name=embedding_model_name, device=device
        )
        print(f"  Embedding dim: {embedder.embedding_dim}")

        # Create vector store with persistence
        store_path = Path(input_path)
        if store_path.is_file():
            store_dir = store_path.parent / ".vrlmrag_store"
        else:
            store_dir = store_path / ".vrlmrag_store"
        store_dir.mkdir(parents=True, exist_ok=True)
        storage_file = str(store_dir / "embeddings.json")

        store = MultimodalVectorStore(
            embedding_provider=embedder,
            storage_path=storage_file,
        )

        # Embed all text chunks
        print(f"\n[3/6] Embedding {len(all_chunks)} chunks with Qwen3-VL...")
        for i, chunk in enumerate(all_chunks):
            content = chunk.get("content", "")
            if not content.strip():
                continue
            metadata = {"type": chunk.get("type", "text")}
            if "slide" in chunk:
                metadata["slide"] = chunk["slide"]
            store.add_text(content=content, metadata=metadata)
            embedded_count += 1
            if (i + 1) % 5 == 0 or (i + 1) == len(all_chunks):
                print(f"  Embedded {i + 1}/{len(all_chunks)} chunks")

        # Embed any extracted images
        for doc in documents:
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
                    embedded_count += 1
                except Exception as e:
                    print(f"  Warning: Could not embed image {img_info['filename']}: {e}")

        print(f"  Total embedded: {embedded_count} documents")
        print(f"  Store persisted to: {storage_file}")

        # Load reranker
        print(f"\n[4/6] Loading Qwen3-VL Reranker...")
        reranker_vl = create_qwen3vl_reranker(
            model_name=reranker_model_name, device=device
        )
        print(f"  Reranker loaded successfully")
    else:
        print("\n[2/6] Qwen3-VL not available - using fallback text reranking")
        print("  Install with: pip install torch transformers qwen-vl-utils torchvision")
        store = None
        reranker_vl = None
        embedding_model_name = "N/A (fallback)"
        reranker_model_name = "N/A (fallback)"

    # ================================================================
    # Step 3: Initialize RLM
    # ================================================================
    print(f"\n[5/6] Initializing RLM (DeepSeek-V3.1 on SambaNova)...")
    rlm = VLRAGGraphRLM(
        provider="sambanova",
        model="DeepSeek-V3.1",
        temperature=0.0,
        max_depth=3,
        max_iterations=10,
    )

    # ================================================================
    # Step 4: Build Knowledge Graph
    # ================================================================
    print("\n  Building knowledge graph...")
    # SambaNova DeepSeek-V3.1 supports 128K context, but free tier has 200K TPD.
    # Budget ~8K chars (~2K tokens) per call to conserve daily token budget.
    kg_context = "\n\n".join([d.get("content", "")[:2000] for d in documents])
    try:
        kg_result = rlm.completion(
            "Extract key concepts, entities, and relationships from this document.",
            kg_context[:8000],
        )
        knowledge_graph = kg_result.response
    except Exception as e:
        knowledge_graph = f"Could not build knowledge graph: {e}"

    # ================================================================
    # Step 5: Run Queries with Qwen3-VL retrieval
    # ================================================================
    queries_to_run = []
    if query:
        queries_to_run = [query]
    else:
        queries_to_run = [
            "What are the main topics covered?",
            "Summarize the key concepts presented.",
        ]

    print(f"\n[6/6] Running {len(queries_to_run)} queries with RAG retrieval...")
    query_results = []
    rrf = ReciprocalRankFusion(k=60)
    fallback_reranker = CompositeReranker()

    for q in queries_to_run:
        print(f"\n  Query: {q}")
        sources_info: List[Dict[str, Any]] = []

        if store is not None and HAS_QWEN3VL:
            # --- Full Qwen3-VL RAG pipeline ---

            # Dense vector search via Qwen3-VL embeddings
            dense_results = store.search(q, top_k=20)
            print(f"    Dense search: {len(dense_results)} results")

            # Keyword search for hybrid
            keyword_results = _keyword_search(store, q, top_k=20)
            print(f"    Keyword search: {len(keyword_results)} results")

            # RRF Fusion
            if keyword_results:
                fused_results = rrf.fuse(
                    [dense_results, keyword_results], weights=[4.0, 1.0]
                )
            else:
                fused_results = dense_results

            print(f"    After RRF fusion: {len(fused_results)} results")

            # Qwen3-VL Reranking
            if reranker_vl and fused_results:
                docs_for_rerank = []
                for r in fused_results[:15]:  # Rerank top 15
                    doc = store.get(r.id)
                    if doc:
                        doc_dict: Dict[str, Any] = {"text": doc.content}
                        if doc.image_path:
                            doc_dict["image"] = doc.image_path
                        docs_for_rerank.append(doc_dict)

                reranked_indices = reranker_vl.rerank(
                    query={"text": q}, documents=docs_for_rerank
                )
                # Reorder by reranker scores
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

            # Build context from top results
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

        # Query RLM with retrieved context
        try:
            q_start = time.time()
            result = rlm.completion(q, context[:8000])
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
            query_results.append(
                {"query": q, "response": f"Error: {e}", "time": 0, "sources": []}
            )
            print(f"    Error: {e}")

    # ================================================================
    # Compile Results & Generate Report
    # ================================================================
    total_elapsed = time.time() - overall_start
    results = {
        "provider": "sambanova",
        "model": "DeepSeek-V3.1",
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


def main():
    parser = argparse.ArgumentParser(
        description="VL-RAG-Graph-RLM: Process documents with SambaNova DeepSeek-V3.1 + Qwen3-VL"
    )

    parser.add_argument(
        "--samba-nova",
        metavar="PATH",
        required=True,
        help="File or folder to process with SambaNova DeepSeek-V3.1 + Qwen3-VL embeddings",
    )

    parser.add_argument(
        "--query", "-q", help="Custom query to run (default: auto-generated)"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output markdown file path (default: print to stdout)",
    )

    args = parser.parse_args()

    run_sambanova_analysis(
        input_path=args.samba_nova, query=args.query, output=args.output
    )


if __name__ == "__main__":
    main()
