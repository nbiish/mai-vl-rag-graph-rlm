#!/usr/bin/env python3
"""
Together AI Template — Full VL-RAG-Graph-RLM Pipeline

Demonstrates all six pillars:
  1. VL: Qwen3-VL multimodal embeddings (text + images)
  2. RAG: Hybrid search (dense + keyword) with RRF fusion
  3. Reranker: Qwen3-VL cross-attention reranking
  4. Graph: Knowledge graph extraction via RLM
  5. RLM: Recursive Language Model with REPL
  6. Pipeline: Markdown report generation

Recommended Models:
    - meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo: Fast Llama 3.1
    - mistralai/Mixtral-8x22B-Instruct-v0.1: Mixtral 8x22B

Environment:
    export TOGETHER_API_KEY=your_key_here
    # Optional: export TOGETHER_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo

Get API Key: https://api.together.ai
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
        llm_provider="together",
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
        "A linked list node contains data and a pointer to the next node.",
        "Reversing a linked list requires updating each node's next pointer.",
        "Iterative reversal uses three pointers: prev, current, and next.",
    ]

    if has_vl:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedder = create_qwen3vl_embedder(device=device)
        store = MultimodalVectorStore(embedding_provider=embedder)
        for t in texts:
            store.add_text(t, metadata={"type": "text"})

        query = "Write a function to reverse a linked list."
        dense_results = store.search(query, top_k=10)

        reranker = create_qwen3vl_reranker(device=device)
        docs = [{"text": store.get(r.id).content} for r in dense_results if store.get(r.id)]
        reranked = reranker.rerank(query={"text": query}, documents=docs)
        context = "\n".join([docs[idx]["text"] for idx, _ in reranked[:3]])
    else:
        context = "\n".join(texts)
        query = "Write a function to reverse a linked list."

    rlm = VLRAGGraphRLM(provider="together", temperature=0.0)
    kg = rlm.completion("Extract key entities and relationships.", context)
    print(f"Knowledge graph: {kg.response[:200]}...")

    result = rlm.completion(query, context)
    print(f"Answer: {result.response[:300]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Together — Full VL-RAG-Graph-RLM Pipeline")
    parser.add_argument("--input", "-i", help="Document to process (PPTX, PDF, TXT, MD)")
    parser.add_argument("--query", "-q", default="What are the main topics covered?")
    parser.add_argument("--manual", action="store_true", help="Run manual pipeline example")
    args = parser.parse_args()

    if not os.getenv("TOGETHER_API_KEY"):
        print("Error: TOGETHER_API_KEY not set")
        print("Get your API key from: https://api.together.ai")
        return

    if args.manual:
        example_manual_pipeline()
    elif args.input:
        example_full_pipeline(args.input, args.query)
    else:
        print("Usage:")
        print("  python provider_together.py --input document.pptx")
        print("  python provider_together.py --manual")


if __name__ == "__main__":
    main()
