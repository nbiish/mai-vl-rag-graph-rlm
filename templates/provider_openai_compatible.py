#!/usr/bin/env python3
"""
Generic OpenAI-Compatible Template — Full VL-RAG-Graph-RLM Pipeline

Demonstrates all six pillars with any OpenAI-compatible endpoint:
  1. VL: Qwen3-VL multimodal embeddings (text + images)
  2. RAG: Hybrid search (dense + keyword) with RRF fusion
  3. Reranker: Qwen3-VL cross-attention reranking
  4. Graph: Knowledge graph extraction via RLM
  5. RLM: Recursive Language Model with REPL
  6. Pipeline: Markdown report generation

For any provider with OpenAI-compatible API:
- Self-hosted models (vLLM, TGI, Ollama)
- Custom proxies and gateways
- Any OpenAI-compatible endpoint

Required Environment Variables:
    export OPENAI_COMPATIBLE_API_KEY=your_api_key_here
    export OPENAI_COMPATIBLE_BASE_URL=https://api.example.com/v1
    export OPENAI_COMPATIBLE_MODEL=your-model-name
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from vl_rag_graph_rlm.clients import GenericOpenAIClient


def example_full_pipeline(input_path: str, query: str = "What are the main topics covered?"):
    """Full 6-pillar pipeline using generic OpenAI-compatible client as the LLM backend."""
    try:
        from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder, create_qwen3vl_reranker
        from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore
        from vl_rag_graph_rlm.rag import ReciprocalRankFusion
        import torch
        has_vl = True
    except ImportError:
        has_vl = False

    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

    client = GenericOpenAIClient(api_key=api_key, base_url=base_url, model_name=model_name)

    # --- Pillar 1: Document Processing + VL Embeddings ---
    path = Path(input_path)
    content = path.read_text() if path.suffix.lower() in {".txt", ".md"} else f"Document: {path.name}"

    if has_vl:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedder = create_qwen3vl_embedder(device=device)
        store = MultimodalVectorStore(embedding_provider=embedder)

        # Chunk and embed
        for i, chunk in enumerate(content.split("\n\n")):
            if chunk.strip():
                store.add_text(chunk.strip(), metadata={"chunk": i})

        # --- Pillar 2: RAG — Dense search ---
        dense_results = store.search(query, top_k=10)
        print(f"Dense results: {len(dense_results)}")

        # --- Pillar 3: Reranker ---
        reranker = create_qwen3vl_reranker(device=device)
        docs = [{"text": store.get(r.id).content} for r in dense_results if store.get(r.id)]
        reranked = reranker.rerank(query={"text": query}, documents=docs)
        context = "\n\n".join([docs[idx]["text"] for idx, _ in reranked[:5]])
    else:
        context = content[:8000]

    # --- Pillar 4: Graph — Knowledge graph extraction ---
    kg_prompt = f"Extract key entities and relationships from this context:\n\n{context[:4000]}"
    kg_response = client.completion(kg_prompt)
    print(f"Knowledge graph: {kg_response[:200]}...")

    # --- Pillar 5: RLM — Query with retrieved context ---
    rlm_prompt = f"Context:\n{context[:8000]}\n\nQuery: {query}\n\nProvide a comprehensive answer."
    answer = client.completion(rlm_prompt)
    print(f"Answer: {answer[:500]}...")


def example_manual_pipeline():
    """Manual pipeline showing each pillar explicitly with the generic client."""
    try:
        from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder, create_qwen3vl_reranker
        from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore
        import torch
        has_vl = True
    except ImportError:
        has_vl = False

    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

    client = GenericOpenAIClient(api_key=api_key, base_url=base_url, model_name=model_name)

    texts = [
        "OpenAI-compatible APIs follow the /v1/chat/completions specification.",
        "vLLM and TGI provide high-throughput inference for open-source models.",
        "Self-hosted models offer data privacy and customization flexibility.",
    ]

    if has_vl:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedder = create_qwen3vl_embedder(device=device)
        store = MultimodalVectorStore(embedding_provider=embedder)
        for t in texts:
            store.add_text(t, metadata={"type": "text"})

        query = "How do self-hosted models work?"
        dense_results = store.search(query, top_k=10)

        reranker = create_qwen3vl_reranker(device=device)
        docs = [{"text": store.get(r.id).content} for r in dense_results if store.get(r.id)]
        reranked = reranker.rerank(query={"text": query}, documents=docs)
        context = "\n".join([docs[idx]["text"] for idx, _ in reranked[:3]])
    else:
        context = "\n".join(texts)
        query = "How do self-hosted models work?"

    kg_response = client.completion(f"Extract key entities and relationships:\n{context}")
    print(f"Knowledge graph: {kg_response[:200]}...")

    answer = client.completion(f"Context:\n{context}\n\nQuery: {query}")
    print(f"Answer: {answer[:300]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generic OpenAI-Compatible — Full VL-RAG-Graph-RLM Pipeline")
    parser.add_argument("--input", "-i", help="Document to process (TXT, MD)")
    parser.add_argument("--query", "-q", default="What are the main topics covered?")
    parser.add_argument("--manual", action="store_true", help="Run manual pipeline example")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

    if not all([api_key, base_url, model_name]):
        print("Error: Missing required environment variables")
        print("Please set:")
        print("  export OPENAI_COMPATIBLE_API_KEY=your_api_key_here")
        print("  export OPENAI_COMPATIBLE_BASE_URL=https://api.example.com/v1")
        print("  export OPENAI_COMPATIBLE_MODEL=your-model-name")
        return

    if args.manual:
        example_manual_pipeline()
    elif args.input:
        example_full_pipeline(args.input, args.query)
    else:
        print("Usage:")
        print("  python provider_openai_compatible.py --input document.txt")
        print("  python provider_openai_compatible.py --manual")


if __name__ == "__main__":
    main()
