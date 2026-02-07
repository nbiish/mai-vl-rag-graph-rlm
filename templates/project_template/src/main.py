"""VL-RAG-Graph-RLM Project - Main Entry Point

This demonstrates the beyond-expert capabilities:
- Unlimited context processing
- Vision RAG (PDFs with images)
- Knowledge graph construction
- Hybrid search with reranking
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vl_rag_graph_rlm import VLRAGGraphRLM
from vl_rag_graph_rlm.rag import (
    CompositeReranker,
    SearchResult,
    ReciprocalRankFusion,
)


def example_1_unlimited_context():
    """Process documents of ANY size via recursive chunking."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Unlimited Context Processing")
    print("=" * 70)
    
    # Initialize RLM with recursive depth
    rlm = VLRAGGraphRLM(
        provider="sambanova",
        model="DeepSeek-V3.1",
        max_depth=5,  # Handle 5x the base context
        max_iterations=20,
    )
    
    # Simulate a large document (would be much larger in reality)
    large_doc = "Chapter 1: Introduction... [1000+ pages of content]"
    
    query = "Summarize key findings across all sections"
    result = rlm.completion(query, large_doc)
    
    print(f"Response: {result.response[:200]}...")
    print(f"Time: {result.execution_time:.2f}s")


def example_2_vision_rag():
    """Query across text AND images simultaneously."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Vision RAG (Text + Images)")
    print("=" * 70)
    
    print("With Qwen3-VL installed:")
    print("  from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder")
    print("  embedder = create_qwen3vl_embedder('Qwen/Qwen3-VL-Embedding-2B')")
    print()
    print("  # Process PDF with images")
    print("  pipeline.add_pdf('doc.pdf', extract_images=True)")
    print("  result = pipeline.query('Explain Figure 3')")


def example_3_knowledge_graph():
    """Extract entities and relationships from documents."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Knowledge Graph Construction")
    print("=" * 70)
    
    rlm = VLRAGGraphRLM(provider="sambanova", model="DeepSeek-V3.1")
    
    doc = "Kubernetes: kube-apiserver manages API. etcd stores state."
    result = rlm.completion(
        "Extract entities and relationships (Format: Entity -> Rel -> Entity)",
        doc
    )
    
    print(f"Knowledge Graph:\n{result.response}")


def example_4_hybrid_search():
    """Combine dense + keyword + RRF + reranking."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Hybrid Search")
    print("=" * 70)
    
    rrf = ReciprocalRankFusion(k=60)
    reranker = CompositeReranker()
    
    # Simulated results
    dense = [SearchResult(content="Neural networks...", score=0.95)]
    keyword = [SearchResult(content="Deep learning...", score=0.92)]
    
    fused = rrf.fuse([dense, keyword])
    reranked, _ = reranker.process("query", [r.__dict__ for r in fused])
    
    print(f"Results: {len(reranked)} chunks after fusion + reranking")


def main():
    """Run all examples."""
    if not os.getenv("SAMBANOVA_API_KEY"):
        print("Error: SAMBANOVA_API_KEY not set")
        print("Get key: https://cloud.sambanova.ai")
        return
    
    print("=" * 70)
    print("VL-RAG-Graph-RLM: Beyond Expert Capabilities")
    print("=" * 70)
    
    example_1_unlimited_context()
    example_2_vision_rag()
    example_3_knowledge_graph()
    example_4_hybrid_search()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
