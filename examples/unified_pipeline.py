"""
Unified Pipeline Example - VL-RAG-Graph-RLM Complete Workflow

This example demonstrates how all components work together:
1. Paddle OCR extracts text, tables, and images from PDFs
2. Qwen3-VL generates multimodal embeddings (text + images)
3. Hybrid search (dense + keyword) retrieves relevant documents
4. Qwen3-VL reranker improves result quality
5. VLRAGGraphRLM provides recursive reasoning
6. Cheap SOTA LLM generates the final answer

Requirements:
    pip install vl-rag-graph-rlm[qwen3vl]
    pip install paddlepaddle paddleocr pymupdf

    # Set API key
    export OPENROUTER_API_KEY=your_key_here
"""

import os
from pathlib import Path

from vl_rag_graph_rlm import MultimodalRAGPipeline, create_pipeline


def example_basic_usage():
    """Basic usage with default settings."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create pipeline with defaults
    # - Uses Qwen3-VL-Embedding-2B (free, local)
    # - Uses Qwen3-VL-Reranker-2B (free, local)
    # - Uses kimi/kimi-k2.5 via OpenRouter (cheap, high quality)
    pipeline = create_pipeline(
        llm_provider="openrouter",
        llm_model="kimi/kimi-k2.5",
        verbose=True
    )
    
    # Add a PDF document
    print("\nAdding PDF document...")
    doc_ids = pipeline.add_pdf(
        pdf_path="research_paper.pdf",
        extract_images=True,  # Extract figures and diagrams
        extract_tables=True   # Extract tables as structured data
    )
    print(f"Added {len(doc_ids)} document chunks")
    
    # Query the document
    print("\nQuerying...")
    result = pipeline.query(
        query="What are the main contributions of this paper?",
        top_k=5
    )
    
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nCost: ${result.total_cost:.4f}")
    print(f"Time: {result.execution_time:.2f}s")
    print(f"Sources used: {len(result.sources)}")


def example_research_paper_analysis():
    """Comprehensive research paper analysis with figures."""
    print("\n" + "=" * 60)
    print("Example 2: Research Paper Analysis with Figures")
    print("=" * 60)
    
    # Initialize with cost-optimized model routing
    pipeline = MultimodalRAGPipeline(
        llm_provider="openrouter",
        llm_model="kimi/kimi-k2.5",              # Strong model for main reasoning
        recursive_model="google/gemini-2.0-flash-001",  # Cheap model for sub-tasks
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",   # Free local embedding
        use_reranker=True,
        reranker_model="Qwen/Qwen3-VL-Reranker-2B",     # Free local reranker
        top_k=10,
        use_hybrid_search=True,                   # Dense + keyword search
        verbose=True
    )
    
    # Process multiple papers
    papers = [
        "paper1_attention_is_all_you_need.pdf",
        "paper2_bert.pdf",
        "paper3_gpt3.pdf"
    ]
    
    for paper in papers:
        if Path(paper).exists():
            print(f"\nProcessing {paper}...")
            doc_ids = pipeline.add_pdf(
                pdf_path=paper,
                extract_images=True,
                metadata={"type": "research_paper", "filename": paper}
            )
            print(f"  Added {len(doc_ids)} chunks")
    
    # Complex queries that reference figures
    questions = [
        "Compare the architecture diagrams across all papers. Which is most efficient?",
        "What are the training costs mentioned in each paper?",
        "Explain the attention mechanism visualization in Figure 2",
        "Summarize the evaluation metrics used and their results",
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = pipeline.query(question, top_k=8)
        print(f"A: {result.answer[:300]}...")
        print(f"   (Cost: ${result.total_cost:.4f}, Time: {result.execution_time:.2f}s)")


def example_technical_documentation():
    """Process technical docs with architecture diagrams."""
    print("\n" + "=" * 60)
    print("Example 3: Technical Documentation Analysis")
    print("=" * 60)
    
    pipeline = create_pipeline(
        llm_provider="openrouter",
        llm_model="deepseek/deepseek-chat",  # Good for technical content
        storage_path="./tech_docs_kb.pkl",    # Persist embeddings
        verbose=True
    )
    
    # Load existing knowledge base if available
    if Path("./tech_docs_kb.pkl").exists():
        print("Loading existing knowledge base...")
        pipeline.load()
    else:
        # Process documentation
        print("Processing documentation...")
        
        # Add PDF documentation
        pipeline.add_pdf(
            "api_reference.pdf",
            extract_images=True,
            metadata={"category": "api_docs"}
        )
        
        # Add architecture diagrams separately
        diagrams = Path("./architecture_diagrams").glob("*.png")
        for diagram in diagrams:
            pipeline.add_image(
                image_path=str(diagram),
                description=f"Architecture diagram: {diagram.stem}",
                metadata={"category": "diagram", "name": diagram.name}
            )
        
        # Add markdown documentation
        for md_file in Path("./docs").glob("*.md"):
            content = md_file.read_text()
            pipeline.add_text(
                content=content,
                metadata={"category": "markdown", "file": md_file.name}
            )
        
        # Save for future use
        pipeline.save()
    
    # Query about implementation
    result = pipeline.query(
        "How does the authentication flow work? Include the relevant diagram.",
        top_k=5
    )
    
    print(f"\nAnswer:\n{result.answer}")
    if result.images:
        print(f"\nReferenced images: {result.images}")


def example_multimodal_search():
    """Search using both text and image queries."""
    print("\n" + "=" * 60)
    print("Example 4: Multimodal Search")
    print("=" * 60)
    
    pipeline = create_pipeline(verbose=True)
    
    # Add documents
    pipeline.add_pdf("catalog.pdf", extract_images=True)
    
    # Search with text only
    print("\nText search: 'red dress'")
    text_results = pipeline.search("red dress", top_k=5)
    for r in text_results:
        print(f"  - {r.id}: score={r.composite_score:.2f}")
    
    # Search with image query
    print("\nImage search: query_image.jpg")
    image_results = pipeline.search(
        query="",
        query_image="query_image.jpg",
        top_k=5
    )
    for r in image_results:
        print(f"  - {r.id}: score={r.composite_score:.2f}")


def example_cost_optimized():
    """Most cost-effective setup."""
    print("\n" + "=" * 60)
    print("Example 5: Cost-Optimized Pipeline")
    print("=" * 60)
    
    # Use cheapest viable models
    pipeline = MultimodalRAGPipeline(
        llm_provider="openrouter",
        llm_model="google/gemini-2.0-flash-001",      # Very cheap, good quality
        recursive_model="solar-pro/solar-pro-3:free",  # Free tier for recursion
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",  # Free, local
        use_reranker=True,                             # Free, local
        max_depth=2,                                   # Limit recursion
        max_iterations=5,                              # Limit iterations
        verbose=True
    )
    
    # Process document
    pipeline.add_pdf("document.pdf", extract_images=False)  # Skip images for speed
    
    # Simple query
    result = pipeline.query("Summarize this document in 3 bullet points")
    
    print(f"\nAnswer: {result.answer}")
    print(f"Cost: ${result.total_cost:.4f} (target: <$0.01)")
    print(f"Time: {result.execution_time:.2f}s")


def example_with_paddle_ocr():
    """Demonstrate Paddle OCR integration for complex layouts."""
    print("\n" + "=" * 60)
    print("Example 6: Advanced PDF Processing with Paddle OCR")
    print("=" * 60)
    
    pipeline = create_pipeline(verbose=True)
    
    # Process complex PDF with tables and figures
    print("Processing complex PDF with Paddle OCR...")
    doc_ids = pipeline.add_pdf(
        pdf_path="complex_report.pdf",
        pages=[1, 2, 3, 10, 11, 12],  # Specific pages
        extract_images=True,
        extract_tables=True,
        use_ocr=True,  # Use Paddle OCR for layout analysis
        metadata={"project": "Q4_analysis", "department": "engineering"}
    )
    
    print(f"Added {len(doc_ids)} document segments")
    
    # Query about specific content
    result = pipeline.query(
        "What are the Q4 revenue numbers from Table 1?",
        top_k=3
    )
    
    print(f"\nAnswer: {result.answer}")
    print(f"\nSources:")
    for source in result.sources:
        print(f"  - Page {source['metadata'].get('page', 'unknown')}: "
              f"{source['content'][:100]}...")


def main():
    """Run all examples."""
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY not set. Set it to run examples.")
        print("export OPENROUTER_API_KEY=your_key_here")
        return
    
    # Run examples
    try:
        example_basic_usage()
    except FileNotFoundError as e:
        print(f"Skipping - file not found: {e}")
    
    try:
        example_research_paper_analysis()
    except FileNotFoundError as e:
        print(f"Skipping - file not found: {e}")
    
    try:
        example_technical_documentation()
    except FileNotFoundError as e:
        print(f"Skipping - file not found: {e}")
    
    try:
        example_multimodal_search()
    except FileNotFoundError as e:
        print(f"Skipping - file not found: {e}")
    
    example_cost_optimized()
    
    try:
        example_with_paddle_ocr()
    except FileNotFoundError as e:
        print(f"Skipping - file not found: {e}")


if __name__ == "__main__":
    main()
