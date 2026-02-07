#!/usr/bin/env python3
"""
ZenMux Template — Full VL-RAG-Graph-RLM Pipeline

Demonstrates all six pillars:
  1. VL: Qwen3-VL multimodal embeddings (text + images)
  2. RAG: Hybrid search (dense + keyword) with RRF fusion
  3. Reranker: Qwen3-VL cross-attention reranking
  4. Graph: Knowledge graph extraction via RLM
  5. RLM: Recursive Language Model with REPL
  6. Pipeline: Markdown report generation

Recommended Models:
    - ernie-5.0-thinking-preview: Best for reasoning
    - dubao-seed-1.8: Best for coding
    - glm-4.7-flash: Fast, cheap responses

Environment:
    export ZENMUX_API_KEY=your_key_here
    # Optional: export ZENMUX_MODEL=ernie-5.0-thinking-preview
    # Optional: export ZENMUX_RECURSIVE_MODEL=glm-4.7-flash

Get API Key: https://zenmux.ai
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from vl_rag_graph_rlm import VLRAGGraphRLM, create_pipeline


def example_full_pipeline(input_path: str, query: str = "What are the main topics covered?"):
    """Full 6-pillar pipeline: VL embeddings → RAG → reranker → graph → RLM → report."""
    pipeline = create_pipeline(
        llm_provider="zenmux",
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",
        use_reranker=True,
    )

    path = Path(input_path)
    if path.suffix.lower() == ".pptx":
        pipeline.add_pptx(str(path), extract_images=True)
    elif path.suffix.lower() == ".pdf":
        pipeline.add_pdf(str(path), extract_images=True)
    else:
        pipeline.add_text(path.read_text())

    result = pipeline.query(query)
    print(f"Answer: {result.answer[:500]}...")
    print(f"Sources: {len(result.sources)}, Time: {result.execution_time:.2f}s")


def example_manual_pipeline():
    """Manual pipeline showing each pillar explicitly."""
    try:
        from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder, create_qwen3vl_reranker
        from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore
        import torch
        has_vl = True
    except ImportError:
        has_vl = False

    texts = [
        "Quantum computing uses qubits that can exist in superposition states.",
        "量子计算利用量子比特的叠加态进行并行计算。",
        "Quantum entanglement enables instantaneous correlation between particles.",
    ]

    if has_vl:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedder = create_qwen3vl_embedder(device=device)
        store = MultimodalVectorStore(embedding_provider=embedder)
        for t in texts:
            store.add_text(t, metadata={"type": "text"})

        query = "Explain quantum computing in Chinese and English."
        dense_results = store.search(query, top_k=10)

        reranker = create_qwen3vl_reranker(device=device)
        docs = [{"text": store.get(r.id).content} for r in dense_results if store.get(r.id)]
        reranked = reranker.rerank(query={"text": query}, documents=docs)
        context = "\n".join([docs[idx]["text"] for idx, _ in reranked[:3]])
    else:
        context = "\n".join(texts)
        query = "Explain quantum computing in Chinese and English."

    rlm = VLRAGGraphRLM(provider="zenmux", temperature=0.0)
    kg = rlm.completion("Extract key entities and relationships.", context)
    print(f"Knowledge graph: {kg.response[:200]}...")

    result = rlm.completion(query, context)
    print(f"Answer: {result.response[:300]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ZenMux — Full VL-RAG-Graph-RLM Pipeline")
    parser.add_argument("--input", "-i", help="Document to process (PPTX, PDF, TXT, MD)")
    parser.add_argument("--query", "-q", default="What are the main topics covered?")
    parser.add_argument("--manual", action="store_true", help="Run manual pipeline example")
    args = parser.parse_args()

    if not os.getenv("ZENMUX_API_KEY"):
        print("Error: ZENMUX_API_KEY not set")
        print("Get your API key from: https://zenmux.ai")
        return

    if args.manual:
        example_manual_pipeline()
    elif args.input:
        example_full_pipeline(args.input, args.query)
    else:
        print("Usage:")
        print("  python provider_zenmux.py --input document.pptx")
        print("  python provider_zenmux.py --manual")


if __name__ == "__main__":
    main()
