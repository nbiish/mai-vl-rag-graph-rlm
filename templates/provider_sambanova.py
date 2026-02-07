#!/usr/bin/env python3
"""
SambaNova Cloud Template - Advanced VL-RAG-Graph-RLM

This template demonstrates the BEYOND EXPERT capabilities of vl-rag-graph-rlm:
- Unlimited context length via recursive processing
- Vision RAG: Process documents with images (PDFs, screenshots, diagrams)
- Knowledge Graph construction from unstructured text
- Multimodal embeddings (Qwen3-VL for text + image + video)
- Hybrid search: Dense + Keyword + RRF fusion
- SOTA Reranking with composite scoring

Use Cases:
    - Analyze 1000+ page technical manuals with figures
    - Process entire codebases with architecture diagrams
    - Build knowledge graphs from research papers
    - Query across text AND visual content simultaneously

Requirements:
    pip install vl-rag-graph-rlm[qwen3vl,paddle]

Environment:
    export SAMBANOVA_API_KEY=your_key_here
    # Optional: export SAMBANOVA_MODEL=DeepSeek-V3.1

Get API Key: https://cloud.sambanova.ai
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Add src to path if running from repo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM, vlraggraphrlm_complete
from vl_rag_graph_rlm.core import REPLExecutor, extract_final
from vl_rag_graph_rlm.rag import (
    ERNIEClient,
    MilvusVectorStore,
    CompositeReranker,
    SearchResult,
    ReciprocalRankFusion,
)

# Optional: Qwen3-VL for multimodal embeddings (requires torch, transformers)
try:
    from vl_rag_graph_rlm.rag.qwen3vl import (
        Qwen3VLEmbeddingProvider,
        Qwen3VLRerankerProvider,
        create_qwen3vl_embedder,
        create_qwen3vl_reranker,
        MultimodalDocument,
    )
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False
    print("Warning: Qwen3-VL not available. Install with: pip install vl-rag-graph-rlm[qwen3vl]")


def example_1_unlimited_context():
    """
    Example 1: Process content beyond model context limits.
    
    Uses recursive chunking + RLM to analyze documents of ANY size.
    The RLM recursively processes chunks and builds up understanding.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Unlimited Context Processing")
    print("=" * 70)
    print("Process documents of ANY size by recursively analyzing chunks")
    print()
    
    # Simulate a large document (would be 100k+ tokens in reality)
    large_document = """
    [Chapter 1: Introduction to Neural Networks]
    Neural networks are computational models inspired by biological neural systems...
    [50 pages of content...]
    
    [Chapter 5: Attention Mechanisms]
    The attention mechanism allows models to focus on relevant parts of input...
    [80 pages of content...]
    
    [Chapter 12: Vision Transformers]
    ViTs apply transformer architectures to image patches...
    [60 pages of content with figures...]
    """
    
    rlm = VLRAGGraphRLM(
        provider="sambanova",
        model="DeepSeek-V3.1",
        temperature=0.0,
        max_depth=5,  # Recursive depth for unlimited context
        max_iterations=20,
    )
    
    query = "Summarize the key architectural innovations across all chapters, focusing on how attention mechanisms evolved from text to vision applications."
    
    print(f"Document size: ~200 pages (simulated)")
    print(f"Query: {query[:80]}...")
    print(f"Max recursion depth: {rlm.max_depth}")
    print("-" * 70)
    
    result = rlm.completion(query, large_document)
    print(f"\nResponse:\n{result.response[:500]}...")
    print(f"\nExecution time: {result.execution_time:.2f}s")


def example_2_multimodal_rag():
    """
    Example 2: Vision RAG - Query across text AND images.
    
    Uses Qwen3-VL embeddings to create unified embeddings for:
    - Text documents
    - Screenshots and diagrams  
    - PDF pages with figures
    
    All content is searchable in a single vector space.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Vision RAG (Multimodal Document Processing)")
    print("=" * 70)
    
    if not HAS_MULTIMODAL:
        print("SKIPPED: Qwen3-VL not installed")
        print("Install with: pip install vl-rag-graph_rlm[qwen3vl]")
        return
    
    print("Process PDFs with figures, screenshots, and diagrams")
    print("All content embedded in unified multimodal vector space")
    print()
    
    # Initialize multimodal embedding model
    embedder = create_qwen3vl_embedder("Qwen/Qwen3-VL-Embedding-2B")
    reranker = create_qwen3vl_reranker("Qwen/Qwen3-VL-Reranker-2B")
    
    print("Models loaded:")
    print(f"  - Embedder: Qwen3-VL-Embedding-2B")
    print(f"  - Reranker: Qwen3-VL-Reranker-2B")
    print()
    
    # Example: Process a multimodal document
    print("Simulating document ingestion:")
    print("  1. PDF 'research_paper.pdf' with 20 pages + 8 figures")
    print("  2. Screenshot 'architecture_diagram.png'")
    print("  3. Code file 'model.py' with docstrings")
    print()
    
    # In real usage:
    # from vl_rag_graph_rlm.pipeline import MultimodalRAGPipeline
    # pipeline = MultimodalRAGPipeline(embedder=embedder, reranker=reranker)
    # pipeline.add_pdf("research_paper.pdf", extract_images=True)
    # pipeline.add_image("architecture_diagram.png")
    # result = pipeline.query("Explain the attention mechanism shown in Figure 3")
    
    query = "Explain the architecture diagram showing the transformer blocks"
    print(f"Query: {query}")
    print("\nNote: This would search across text AND image embeddings")
    print("      to find relevant figures and their captions.")


def example_3_knowledge_graph():
    """
    Example 3: Build Knowledge Graph from documents.
    Extracts entities and relationships for complex reasoning.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Knowledge Graph Construction")
    print("=" * 70)
    print("Extract entities and relationships from documents")
    print()
    
    # Simplified example to fit within token limits
    tech_doc = """Kubernetes: kube-apiserver manages API requests.
    etcd stores cluster state. kube-scheduler assigns pods to nodes.
    Worker nodes run kubelet and kube-proxy."""
    
    rlm = VLRAGGraphRLM(
        provider="sambanova",
        model="DeepSeek-V3.1",
        temperature=0.0,
    )
    
    print("Document: Kubernetes components")
    print("Extracting entities and relationships...")
    print("-" * 70)
    
    # Simpler prompt to avoid token limit
    result = rlm.completion(
        "Extract entities and relationships from this text. Format: Entity -> Relationship -> Entity",
        tech_doc
    )
    print(f"\nKnowledge Graph Extraction:\n{result.response}")
    
    # Query the extracted knowledge
    query = "How do kube-apiserver and etcd interact?"
    print(f"\nQuery: {query}")
    print("\nThe RLM would trace relationships through the extracted graph...")


def example_4_hybrid_search():
    """
    Example 4: Hybrid Search with RRF Fusion + Composite Reranker.
    
    Combines:
    - Dense vector search (semantic similarity)
    - Keyword/BM25 search (exact matches)
    - Reciprocal Rank Fusion for combining results
    - Composite reranking (fuzzy + keyword + semantic + position)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Hybrid Search (Dense + Keyword + RRF + Rerank)")
    print("=" * 70)
    print("Milvus vector store with hybrid dense + keyword search")
    print("RRF fusion + composite reranking for best results")
    print()
    
    # Initialize components
    print("Components:")
    print("  - MilvusVectorStore: Hybrid dense + keyword search")
    print("  - ReciprocalRankFusion: Combine dense and keyword results")
    print("  - CompositeReranker: Fuzzy + keyword + semantic scoring")
    print()
    
    # Simulate search results
    dense_results = [
        SearchResult(content="Neural networks use attention mechanisms...", score=0.95),
        SearchResult(content="Transformer architecture overview...", score=0.88),
    ]
    
    keyword_results = [
        SearchResult(content="Attention is all you need paper...", score=0.92),
        SearchResult(content="Self-attention in transformers...", score=0.85),
    ]
    
    rrf = ReciprocalRankFusion(k=60)
    reranker = CompositeReranker()
    
    query = "attention mechanism in transformers"
    
    print(f"Query: {query}")
    print(f"Dense results: {len(dense_results)} chunks")
    print(f"Keyword results: {len(keyword_results)} chunks")
    print()
    
    # Fuse results
    fused = rrf.fuse([dense_results, keyword_results])
    print(f"After RRF fusion: {len(fused)} chunks")
    
    # Rerank
    reranked, status = reranker.process(query, [r.__dict__ for r in fused])
    print(f"After composite reranking: {len(reranked)} chunks")
    print(f"Top result score: {reranked[0].get('composite_score', 'N/A')}")


def example_5_codebase_analysis():
    """
    Example 5: Analyze entire codebase with vision support.
    
    Process:
    - All source files (Python, JS, etc.)
    - Architecture diagrams (PNG, SVG)
    - README documentation
    - Build a unified searchable index
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Complete Codebase Analysis (Code + Diagrams)")
    print("=" * 70)
    print("Process entire repositories including:")
    print("  - Source code files")
    print("  - Architecture diagrams and screenshots")
    print("  - Documentation (markdown, rst)")
    print("  - Configuration files")
    print()
    
    rlm = VLRAGGraphRLM(
        provider="sambanova",
        model="DeepSeek-V3.1",
        temperature=0.1,
        max_depth=3,
    )
    
    print("Simulated repository: microservices-platform/")
    print("  - 150 Python files (25,000 lines)")
    print("  - 12 architecture diagrams")
    print("  - 8 markdown documentation files")
    print("  - Database schemas and API specs")
    print()
    
    queries = [
        "Find all authentication-related code and explain the flow",
        "Show me the architecture diagram for the payment service",
        "What database models are used for user management?",
        "Explain the caching strategy across all services",
    ]
    
    print("Example queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    
    print(f"\nSelected query: {queries[0]}")
    print("\nThe RLM would:")
    print("  1. Search code + docs + diagrams for 'authentication'")
    print("  2. Extract relevant code snippets")
    print("  3. Reference architecture diagrams")
    print("  4. Build comprehensive explanation")


def main():
    """Run all advanced examples."""
    if not os.getenv("SAMBANOVA_API_KEY"):
        print("Error: SAMBANOVA_API_KEY not set")
        print("Get your API key from: https://cloud.sambanova.ai")
        print("\nThen run:")
        print("  export SAMBANOVA_API_KEY=your_key_here")
        return
    
    print("=" * 70)
    print("VL-RAG-Graph-RLM: Advanced Capabilities Demo")
    print("=" * 70)
    print()
    print("This template showcases BEYOND EXPERT features:")
    print("  ✓ Unlimited context via recursive processing")
    print("  ✓ Vision RAG (text + image + video)")
    print("  ✓ Knowledge graph construction")
    print("  ✓ Hybrid search with RRF fusion")
    print("  ✓ Complete codebase analysis")
    print()
    print("Provider: SambaNova Cloud (DeepSeek-V3.1)")
    print("=" * 70)
    
    try:
        example_1_unlimited_context()
        example_2_multimodal_rag()
        example_3_knowledge_graph()
        example_4_hybrid_search()
        example_5_codebase_analysis()
        
        print("\n" + "=" * 70)
        print("All advanced examples completed!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Install with full features: pip install vl-rag-graph-rlm[all]")
        print("  2. Try with your own documents: pipeline.add_pdf('your_doc.pdf')")
        print("  3. Scale to larger datasets with Milvus vector store")
        print()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
