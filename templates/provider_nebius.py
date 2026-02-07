#!/usr/bin/env python3
"""
Nebius Token Factory Template - Advanced VL-RAG-Graph-RLM

This template demonstrates the BEYOND EXPERT capabilities of vl-rag-graph-rlm:
- Large context window processing (128K+ tokens supported)
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
    export NEBIUS_API_KEY=your_key_here
    # Optional: export NEBIUS_MODEL=MiniMaxAI/MiniMax-M2.1
    # Optional: export NEBIUS_CONTEXT_WINDOW=128000  # Large context window

Recommended Models (Nebius Token Factory):
    - MiniMaxAI/MiniMax-M2.1: MiniMax M2.1 (default)
    - z-ai/GLM-4.7: Z.AI's flagship, 128K context, excellent reasoning
    - deepseek-ai/DeepSeek-R1-0528: Reasoning model, large context
    - meta-llama/Meta-Llama-3.1-70B-Instruct: Llama 3.1 70B, 128K context

Get API Key: https://tokenfactory.nebius.com
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


def get_context_window() -> int:
    """Get context window size from environment or default to large value."""
    env_window = os.getenv("NEBIUS_CONTEXT_WINDOW")
    if env_window:
        try:
            return int(env_window)
        except ValueError:
            pass
    return 128000  # GLM-4.7 default large context


def get_model() -> str:
    """Get model from environment or use default."""
    return os.getenv("NEBIUS_MODEL", "MiniMaxAI/MiniMax-M2.1")


def example_1_large_context_processing():
    """
    Example 1: Process content with large context window.
    
    Unlike SambaNova's 200K TPD limit, Nebius Token Factory allows
    using the full 128K context window of models like GLM-4.7.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Large Context Processing (No Token Limits!)")
    print("=" * 70)
    
    context_window = get_context_window()
    model = get_model()
    
    print(f"Model: {model}")
    print(f"Context window: {context_window:,} tokens")
    print("Nebius Token Factory has no daily token limits like SambaNova!")
    print()
    
    # Simulate a large document - with Nebius we can use much more content
    large_document = """
    [Chapter 1: Introduction to Neural Networks]
    Neural networks are computational models inspired by biological neural systems.
    They consist of interconnected nodes (neurons) that process information.
    [Extensive content spanning 50 pages...]
    
    [Chapter 5: Attention Mechanisms]
    The attention mechanism allows models to focus on relevant parts of input.
    Self-attention computes relationships between all positions simultaneously.
    [Extensive content spanning 80 pages with diagrams...]
    
    [Chapter 12: Vision Transformers]
    ViTs apply transformer architectures to image patches.
    They divide images into fixed-size patches and process them as sequences.
    [Extensive content spanning 60 pages with figures...]
    
    [Chapter 20: Multimodal Learning]
    Multimodal models combine vision and language understanding.
    CLIP, DALL-E, and GPT-4V represent major advances in this field.
    [Extensive content spanning 100 pages...]
    """
    
    # With Nebius, we can use deeper recursion and larger context
    rlm = VLRAGGraphRLM(
        provider="nebius",
        model=model,
        temperature=0.0,
        max_depth=5,
        max_iterations=20,
    )
    
    query = "Analyze the evolution from basic neural networks to multimodal learning. " \
            "Compare attention mechanisms in NLP vs Vision Transformers."
    
    print(f"Document size: ~290 pages (simulated)")
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
    print("  1. PDF 'research_paper.pdf' with 50 pages + 20 figures")
    print("  2. Screenshot 'architecture_diagram.png'")
    print("  3. Code file 'model.py' with docstrings")
    print("  4. Presentation 'results.pptx' with charts")
    print()
    
    query = "Explain the architecture diagram showing the transformer blocks and compare with Figure 3"
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
    
    model = get_model()
    
    # With Nebius's large context, we can process much more content
    tech_doc = """Kubernetes Architecture:
    
    kube-apiserver: The central management entity that exposes the Kubernetes API.
    It handles all REST requests and is the frontend of the control plane.
    
    etcd: A consistent and highly-available key-value store used as Kubernetes' 
    backing store for all cluster data. It stores the entire cluster state.
    
    kube-scheduler: Watches for newly created Pods with no assigned node, and 
    selects a node for them to run on based on resource requirements and constraints.
    
    kube-controller-manager: Runs controller processes that regulate the state
    of the cluster. Controllers include Node, Replication, Endpoints, and Service Account.
    
    Worker Nodes:
    - kubelet: An agent that runs on each node in the cluster. It ensures containers
      are running in a Pod and communicates with the control plane.
    - kube-proxy: Maintains network rules on nodes, allowing network communication
      to Pods from inside or outside the cluster.
    - Container runtime: Software responsible for running containers (containerd, CRI-O).
    
    Add-ons:
    - CoreDNS: Provides cluster DNS service for service discovery.
    - Ingress Controller: Manages external access to services in the cluster.
    - Metrics Server: Collects resource usage data for autoscaling.
    """
    
    rlm = VLRAGGraphRLM(
        provider="nebius",
        model=model,
        temperature=0.0,
    )
    
    print("Document: Kubernetes architecture with 10+ components")
    print(f"Model: {model}")
    print("Extracting entities and relationships...")
    print("-" * 70)
    
    result = rlm.completion(
        "Extract all entities (components, services) and their relationships. "
        "Format as: Entity -> Relationship -> Entity. Include component responsibilities.",
        tech_doc
    )
    print(f"\nKnowledge Graph Extraction:\n{result.response}")
    
    # Query the extracted knowledge
    query = "How do kube-apiserver, etcd, and kube-scheduler interact in pod creation?"
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
        SearchResult(content="Neural networks use attention mechanisms to weigh input importance...", score=0.95),
        SearchResult(content="Transformer architecture introduced multi-head attention...", score=0.88),
        SearchResult(content="Vision Transformers apply attention to image patches...", score=0.85),
    ]
    
    keyword_results = [
        SearchResult(content="Attention is all you need paper explains the mechanism...", score=0.92),
        SearchResult(content="Self-attention in transformers uses query, key, value...", score=0.89),
        SearchResult(content="Cross-attention enables encoder-decoder communication...", score=0.84),
    ]
    
    rrf = ReciprocalRankFusion(k=60)
    reranker = CompositeReranker()
    
    query = "attention mechanism implementation in transformers"
    
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
    
    model = get_model()
    context_window = get_context_window()
    
    rlm = VLRAGGraphRLM(
        provider="nebius",
        model=model,
        temperature=0.1,
        max_depth=3,
    )
    
    print(f"Model: {model}")
    print(f"Context window: {context_window:,} tokens")
    print("Simulated repository: microservices-platform/")
    print("  - 500 Python files (50,000 lines)")
    print("  - 25 architecture diagrams")
    print("  - 15 markdown documentation files")
    print("  - Database schemas and API specs")
    print()
    
    queries = [
        "Find all authentication-related code and explain the security flow",
        "Show me the architecture diagram for the payment service",
        "What database models are used for user management?",
        "Explain the caching strategy across all microservices",
        "Identify all API endpoints and their rate limiting",
    ]
    
    print("Example queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    
    print(f"\nSelected query: {queries[0]}")
    print("\nThe RLM would:")
    print("  1. Search code + docs + diagrams for 'authentication'")
    print("  2. Extract relevant code snippets (can use more context with Nebius)")
    print("  3. Reference architecture diagrams")
    print("  4. Build comprehensive explanation")


def example_6_recursive_analysis():
    """
    Example 6: Deep recursive analysis with large context.
    
    Demonstrates how Nebius's lack of token limits enables
    more aggressive recursive processing.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Deep Recursive Analysis")
    print("=" * 70)
    print("Nebius advantage: No token limits = deeper recursion")
    print()
    
    model = get_model()
    
    # Simulate a complex technical specification
    spec_document = """
    [API Specification v2.0]
    
    Section 1: Authentication & Authorization
    - OAuth 2.0 with PKCE for mobile apps
    - JWT tokens with RS256 signing
    - Refresh token rotation
    - Scope-based permissions
    
    Section 2: Rate Limiting
    - 1000 requests/hour for tier 1
    - 10000 requests/hour for tier 2
    - Burst allowance: 100 requests
    
    Section 3: Error Handling
    - Standard HTTP status codes
    - RFC 7807 Problem Details
    - Request ID for tracing
    
    Section 4: Pagination
    - Cursor-based for real-time data
    - Offset-based for static collections
    - Default page size: 50 items
    
    Section 5: Webhooks
    - Event types and signatures
    - Retry logic with exponential backoff
    - Idempotency keys
    
    Section 6: SDK Architecture
    - Auto-generated from OpenAPI
    - Retry middleware
    - Circuit breaker pattern
    """
    
    rlm = VLRAGGraphRLM(
        provider="nebius",
        model=model,
        temperature=0.0,
        max_depth=5,  # Can go deeper with Nebius
        max_iterations=25,
    )
    
    print(f"Model: {model}")
    print("Document: API Specification with 6 sections")
    print(f"Max recursion depth: {rlm.max_depth} (deeper than SambaNova)")
    print()
    
    query = "Analyze the security architecture across all sections. " \
            "How do authentication, rate limiting, and webhooks work together?"
    
    print(f"Query: {query}")
    print("-" * 70)
    
    result = rlm.completion(query, spec_document)
    print(f"\nResponse:\n{result.response[:500]}...")
    print(f"\nExecution time: {result.execution_time:.2f}s")
    print(f"Total LLM calls: {rlm.stats['llm_calls']}")
    print(f"Total iterations: {rlm.stats['iterations']}")


def main():
    """Run all advanced examples."""
    if not os.getenv("NEBIUS_API_KEY"):
        print("Error: NEBIUS_API_KEY not set")
        print("Get your API key from: https://tokenfactory.nebius.com")
        print("\nThen run:")
        print("  export NEBIUS_API_KEY=your_key_here")
        print("\nOptional environment variables:")
        print("  export NEBIUS_MODEL=z-ai/GLM-4.7")
        print("  export NEBIUS_CONTEXT_WINDOW=128000")
        return
    
    model = get_model()
    context_window = get_context_window()
    
    print("=" * 70)
    print("VL-RAG-Graph-RLM: Advanced Capabilities Demo (Nebius)")
    print("=" * 70)
    print()
    print("This template showcases BEYOND EXPERT features:")
    print("  ✓ Large context window (128K tokens)")
    print("  ✓ No daily token limits (unlike SambaNova)")
    print("  ✓ Vision RAG (text + image + video)")
    print("  ✓ Knowledge graph construction")
    print("  ✓ Hybrid search with RRF fusion")
    print("  ✓ Complete codebase analysis")
    print("  ✓ Deep recursive processing")
    print()
    print(f"Provider: Nebius Token Factory")
    print(f"Model: {model}")
    print(f"Context Window: {context_window:,} tokens")
    print("=" * 70)
    
    try:
        example_1_large_context_processing()
        example_2_multimodal_rag()
        example_3_knowledge_graph()
        example_4_hybrid_search()
        example_5_codebase_analysis()
        example_6_recursive_analysis()
        
        print("\n" + "=" * 70)
        print("All advanced examples completed!")
        print("=" * 70)
        print()
        print("Key advantages of Nebius Token Factory:")
        print("  - 128K context window (no 200K TPD limit)")
        print("  - Deep recursive processing enabled")
        print("  - GLM-4.7: Excellent for reasoning and coding")
        print()
        print("Next steps:")
        print("  1. Install with full features: pip install vl-rag-graph-rlm[all]")
        print("  2. Try with your own documents: pipeline.add_pdf('your_doc.pdf')")
        print("  3. Scale to larger datasets with Milvus vector store")
        print("  4. Use CLI: python src/vrlmrag.py --nebius <file_or_folder>")
        print()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
