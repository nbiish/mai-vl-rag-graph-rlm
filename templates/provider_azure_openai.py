#!/usr/bin/env python3
"""
Azure OpenAI Template — Full VL-RAG-Graph-RLM Pipeline

Demonstrates all six pillars:
  1. VL: Qwen3-VL multimodal embeddings (text + images)
  2. RAG: Hybrid search (dense + keyword) with RRF fusion
  3. Reranker: Qwen3-VL cross-attention reranking
  4. Graph: Knowledge graph extraction via RLM
  5. RLM: Recursive Language Model with REPL
  6. Pipeline: Markdown report generation

Required Environment Variables:
    export AZURE_OPENAI_API_KEY=your_azure_key_here
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_API_VERSION=2024-02-01  # Optional
    # Optional: export AZURE_OPENAI_MODEL=gpt-4o

Get API Key: Azure Portal > OpenAI Service > Keys and Endpoint
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
        llm_provider="azure_openai",
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
        "Azure OpenAI provides enterprise-grade access to GPT models.",
        "Cloud computing offers scalability, reliability, and cost efficiency.",
        "Azure regions provide low-latency access worldwide with data residency compliance.",
    ]

    if has_vl:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedder = create_qwen3vl_embedder(device=device)
        store = MultimodalVectorStore(embedding_provider=embedder)
        for t in texts:
            store.add_text(t, metadata={"type": "text"})

        query = "Summarize the benefits of cloud computing."
        dense_results = store.search(query, top_k=10)

        reranker = create_qwen3vl_reranker(device=device)
        docs = [{"text": store.get(r.id).content} for r in dense_results if store.get(r.id)]
        reranked = reranker.rerank(query={"text": query}, documents=docs)
        context = "\n".join([docs[idx]["text"] for idx, _ in reranked[:3]])
    else:
        context = "\n".join(texts)
        query = "Summarize the benefits of cloud computing."

    rlm = VLRAGGraphRLM(provider="azure_openai", temperature=0.0)
    kg = rlm.completion("Extract key entities and relationships.", context)
    print(f"Knowledge graph: {kg.response[:200]}...")

    result = rlm.completion(query, context)
    print(f"Answer: {result.response[:300]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Azure OpenAI — Full VL-RAG-Graph-RLM Pipeline")
    parser.add_argument("--input", "-i", help="Document to process (PPTX, PDF, TXT, MD)")
    parser.add_argument("--query", "-q", default="What are the main topics covered?")
    parser.add_argument("--manual", action="store_true", help="Run manual pipeline example")
    args = parser.parse_args()

    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Error: AZURE_OPENAI_API_KEY not set")
        print("Get your key from Azure Portal > OpenAI Service > Keys and Endpoint")
        return
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Error: AZURE_OPENAI_ENDPOINT not set")
        print("Format: https://your-resource.openai.azure.com/")
        return

    if args.manual:
        example_manual_pipeline()
    elif args.input:
        example_full_pipeline(args.input, args.query)
    else:
        print("Usage:")
        print("  python provider_azure_openai.py --input document.pptx")
        print("  python provider_azure_openai.py --manual")


if __name__ == "__main__":
    main()
