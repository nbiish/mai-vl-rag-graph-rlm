#!/usr/bin/env python3
"""
Modal Research Template — Full VL-RAG-Graph-RLM Pipeline (GLM-5 Frontier Inference)

Demonstrates all six pillars:
  1. VL: Qwen3-VL multimodal embeddings (text + images)
  2. RAG: Hybrid search (dense + keyword) with RRF fusion
  3. Reranker: Qwen3-VL cross-attention reranking
  4. Graph: Knowledge graph extraction via RLM
  5. RLM: Recursive Language Model with REPL
  6. Pipeline: Markdown report generation

Modal Research provides free OpenAI-compatible inference for GLM-5,
Zhipu AI's 745B MoE frontier model (44B active params, MIT license).
Runs on 8×B200 GPUs via SGLang with 30-75 tok/s per user.

Available Models:
    - zai-org/GLM-5-FP8: GLM-5 745B in FP8 (frontier-class, recommended)

⚠️  Experimental provider — free tier has rate limits:
    - 1 concurrent request per credential
    - No direct token limits (request-based throttling)
    - May have intermittent availability

Environment:
    export MODAL_RESEARCH_API_KEY=your_key_here
    # Optional fallback key (different account):
    # export MODAL_RESEARCH_API_KEY_FALLBACK=your_second_key_here

Get API Key: https://modal.com/glm-5-endpoint
Blog: https://modal.com/blog/try-glm-5
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
        llm_provider="modalresearch",
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
        "GLM-5 is a 745B parameter mixture-of-experts model with 44B active parameters.",
        "Modal Research deploys GLM-5 on 8×B200 GPUs using SGLang for inference.",
        "The model uses DeepSeek Sparse Attention for efficient long-context processing.",
        "GLM-5 matches proprietary frontier models and is available under MIT license.",
    ]

    if has_vl:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedder = create_qwen3vl_embedder(device=device)
        store = MultimodalVectorStore(embedding_provider=embedder)
        for t in texts:
            store.add_text(t, metadata={"type": "text"})

        query = "Explain the architecture and deployment of GLM-5."
        dense_results = store.search(query, top_k=10)

        reranker = create_qwen3vl_reranker(device=device)
        docs = [{"text": store.get(r.id).content} for r in dense_results if store.get(r.id)]
        reranked = reranker.rerank(query={"text": query}, documents=docs)
        context = "\n".join([docs[idx]["text"] for idx, _ in reranked[:3]])
    else:
        context = "\n".join(texts)
        query = "Explain the architecture and deployment of GLM-5."

    rlm = VLRAGGraphRLM(provider="modalresearch", temperature=0.0)
    kg = rlm.completion("Extract key entities and relationships.", context)
    print(f"Knowledge graph: {kg.response[:200]}...")

    result = rlm.completion(query, context)
    print(f"Answer: {result.response[:300]}...")
    print(f"Time: {result.execution_time:.2f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Modal Research — Full VL-RAG-Graph-RLM Pipeline (GLM-5)")
    parser.add_argument("--input", "-i", help="Document to process (PPTX, PDF, TXT, MD)")
    parser.add_argument("--query", "-q", default="What are the main topics covered?")
    parser.add_argument("--manual", action="store_true", help="Run manual pipeline example")
    args = parser.parse_args()

    if not os.getenv("MODAL_RESEARCH_API_KEY"):
        print("Error: MODAL_RESEARCH_API_KEY not set")
        print("Get your API key from: https://modal.com/glm-5-endpoint")
        return

    if args.manual:
        example_manual_pipeline()
    elif args.input:
        example_full_pipeline(args.input, args.query)
    else:
        print("Usage:")
        print("  python provider_modalresearch.py --input document.pptx")
        print("  python provider_modalresearch.py --manual")


if __name__ == "__main__":
    main()
