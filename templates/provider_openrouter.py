#!/usr/bin/env python3
"""
OpenRouter Template — Full VL-RAG-Graph-RLM Pipeline

Demonstrates all six pillars:
  1. VL: Qwen3-VL multimodal embeddings (text + images)
  2. RAG: Hybrid search (dense + keyword) with RRF fusion
  3. Reranker: Qwen3-VL cross-attention reranking
  4. Graph: Knowledge graph extraction via RLM
  5. RLM: Recursive Language Model with REPL
  6. Pipeline: Markdown report generation

Recommended Models:
    - minimax/minimax-m2.1: Minimax 2.1, excellent reasoning
    - kimi/kimi-k2.5: Excellent reasoning, very cheap
    - z-ai/glm-4.7: Great for coding
    - solar-pro/solar-pro-3:free: Free tier
    - google/gemini-3-flash-preview: Fast with 1M context

Environment:
    export OPENROUTER_API_KEY=your_key_here
    # Optional: export OPENROUTER_MODEL=minimax/minimax-m2.1
    # Optional: export OPENROUTER_RECURSIVE_MODEL=solar-pro/solar-pro-3:free

Get API Key: https://openrouter.ai
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from vl_rag_graph_rlm import VLRAGGraphRLM, MultimodalRAGPipeline, create_pipeline


def example_simple():
    """Example 1: Simple RLM query (pillar 5 only)."""
    rlm = VLRAGGraphRLM(provider="openrouter", temperature=0.0)

    print(f"[Simple] Model: {rlm.model}")
    result = rlm.completion("Compare the efficiency of different sorting algorithms.")
    print(f"Response: {result.response[:200]}...")
    print(f"Time: {result.execution_time:.2f}s")


def example_full_pipeline(input_path: str, query: str = "What are the main topics covered?"):
    """Example 2: Full 6-pillar pipeline with document processing."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set")
        print("Get your API key from: https://openrouter.ai")
        return

    # --- Pillar 6: Unified Pipeline ---
    pipeline = create_pipeline(
        llm_provider="openrouter",
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",
        use_reranker=True,
    )

    # --- Pillar 1: VL — Ingest document with images ---
    path = Path(input_path)
    if path.suffix.lower() == ".pptx":
        pipeline.add_pptx(str(path), extract_images=True)
    elif path.suffix.lower() == ".pdf":
        pipeline.add_pdf(str(path), extract_images=True)
    else:
        pipeline.add_text(path.read_text())

    # --- Pillars 2-5: RAG + Reranker + Graph + RLM ---
    result = pipeline.query(query)

    print(f"\nAnswer: {result.answer[:500]}...")
    print(f"Sources: {len(result.sources)}")
    print(f"Time: {result.execution_time:.2f}s")
    print(f"LLM calls: {result.llm_calls}, Iterations: {result.iterations}")


def example_manual_pipeline(input_path: str):
    """Example 3: Manual pipeline showing each pillar explicitly."""
    from vl_rag_graph_rlm.rag import SearchResult, ReciprocalRankFusion, HybridSearcher

    # Qwen3-VL imports (optional)
    try:
        from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder, create_qwen3vl_reranker
        from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore
        import torch
        has_vl = True
    except ImportError:
        has_vl = False

    # --- Pillar 1: VL Embeddings ---
    if has_vl:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedder = create_qwen3vl_embedder(device=device)
        store = MultimodalVectorStore(embedding_provider=embedder)

        # Add sample text
        store.add_text("OpenRouter provides access to 200+ models via a single API.", metadata={"source": "docs"})
        store.add_text("RRF fusion combines dense and keyword search results.", metadata={"source": "rag"})
        store.add_text("Qwen3-VL embeds text and images in a unified vector space.", metadata={"source": "vl"})

        # --- Pillar 2: RAG — Hybrid Search ---
        query = "How does multimodal search work?"
        dense_results = store.search(query, top_k=10)
        print(f"Dense results: {len(dense_results)}")

        # --- Pillar 3: Reranker ---
        reranker = create_qwen3vl_reranker(device=device)
        docs = [{"text": store.get(r.id).content} for r in dense_results if store.get(r.id)]
        reranked = reranker.rerank(query={"text": query}, documents=docs)
        print(f"Reranked: {len(reranked)} results")

        context = "\n".join([docs[idx]["text"] for idx, _ in reranked[:3]])
    else:
        context = "OpenRouter provides access to 200+ models. RRF combines search results."
        query = "How does multimodal search work?"

    # --- Pillar 4: Graph — Knowledge Graph Extraction ---
    rlm = VLRAGGraphRLM(provider="openrouter", temperature=0.0)
    kg = rlm.completion("Extract key entities and relationships.", context)
    print(f"Knowledge graph: {kg.response[:200]}...")

    # --- Pillar 5: RLM — Query with Retrieved Context ---
    result = rlm.completion(query, context)
    print(f"Answer: {result.response[:300]}...")
    print(f"Time: {result.execution_time:.2f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OpenRouter — Full VL-RAG-Graph-RLM Pipeline")
    parser.add_argument("--input", "-i", help="Document to process (PPTX, PDF, TXT, MD)")
    parser.add_argument("--query", "-q", default="What are the main topics covered?")
    parser.add_argument("--simple", action="store_true", help="Run simple RLM-only example")
    parser.add_argument("--manual", action="store_true", help="Run manual pipeline example")
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set")
        print("Get your API key from: https://openrouter.ai")
        return

    if args.simple:
        example_simple()
    elif args.manual and args.input:
        example_manual_pipeline(args.input)
    elif args.input:
        example_full_pipeline(args.input, args.query)
    else:
        print("Usage:")
        print("  python provider_openrouter.py --simple")
        print("  python provider_openrouter.py --input document.pptx")
        print("  python provider_openrouter.py --input document.pptx --manual")
        print("  python provider_openrouter.py --input document.pptx --query 'Summarize key concepts'")


if __name__ == "__main__":
    main()
