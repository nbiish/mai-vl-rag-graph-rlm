"""Multimodal RAG Example using Qwen3-VL.

This example demonstrates how to use the Qwen3-VL embedding and reranker models
for multimodal retrieval-augmented generation. It supports:
- Text documents
- Images
- Videos
- PDF pages (converted to images)

Requirements:
    pip install unified-rlm[qwen3vl]

Or manually:
    pip install torch transformers pillow qwen-vl-utils pdf2image pymupdf
"""

import os
import tempfile
from pathlib import Path


def example_basic_text_rag():
    """Basic text-only RAG with Qwen3-VL embeddings."""
    from vl_rag_graph_rlm.rag.provider import MultimodalVLRAGGraphRLM
    
    print("=" * 60)
    print("Example 1: Basic Text RAG with Qwen3-VL")
    print("=" * 60)
    
    # Initialize multimodal RAG
    rag = MultimodalVLRAGGraphRLM(
        llm_provider="openrouter",  # or "openai", "anthropic", etc.
        llm_model="gpt-4o",
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",  # or 8B for better quality
        use_reranker=True,
        reranker_model="Qwen/Qwen3-VL-Reranker-2B",
        torch_dtype="bfloat16",  # Use float16 if bfloat16 not supported
    )
    
    # Add text documents
    documents = [
        "Climate change is caused by greenhouse gas emissions from burning fossil fuels.",
        "Renewable energy sources include solar, wind, and hydroelectric power.",
        "The Paris Agreement aims to limit global warming to 1.5 degrees Celsius.",
        "Electric vehicles produce zero direct emissions and reduce air pollution.",
    ]
    
    for i, doc in enumerate(documents):
        rag.add_text(doc, metadata={"source": f"doc_{i}"})
    
    # Query
    result = rag.query("What causes climate change?", top_k=3)
    
    print(f"\nQuery: What causes climate change?")
    print(f"\nAnswer: {result.content}")
    print(f"\nRetrieved {len(result.sources)} sources")
    
    return result


def example_pdf_rag():
    """RAG with PDF documents (pages converted to images)."""
    from vl_rag_graph_rlm.rag.provider import MultimodalVLRAGGraphRLM
    
    print("\n" + "=" * 60)
    print("Example 2: PDF RAG with Qwen3-VL")
    print("=" * 60)
    
    rag = MultimodalVLRAGGraphRLM(
        llm_provider="openrouter",
        llm_model="gpt-4o",
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",
        use_reranker=True,
    )
    
    # Add PDF (pages will be converted to images)
    pdf_path = "path/to/your/document.pdf"  # Replace with actual PDF path
    
    if os.path.exists(pdf_path):
        # Add all pages
        doc_ids = rag.add_pdf(pdf_path)
        print(f"Added {len(doc_ids)} pages from PDF")
        
        # Query
        result = rag.query("What is the main topic of this document?", top_k=5)
        
        print(f"\nAnswer: {result.content}")
    else:
        print(f"PDF not found at {pdf_path}")
        print("Skipping PDF example - please provide a valid PDF path")


def example_image_rag():
    """RAG with image documents."""
    from vl_rag_graph_rlm.rag.provider import MultimodalVLRAGGraphRLM
    
    print("\n" + "=" * 60)
    print("Example 3: Image RAG with Qwen3-VL")
    print("=" * 60)
    
    rag = MultimodalVLRAGGraphRLM(
        llm_provider="openrouter",
        llm_model="gpt-4o",
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",
        use_reranker=True,
    )
    
    # Add images with descriptions
    image_docs = [
        ("charts/sales_q1.png", "Q1 sales chart showing 20% growth"),
        ("charts/sales_q2.png", "Q2 sales chart showing 35% growth"),
        ("diagrams/architecture.png", "System architecture diagram"),
    ]
    
    for img_path, description in image_docs:
        if os.path.exists(img_path):
            rag.add_image(img_path, description=description)
            print(f"Added: {img_path}")
        else:
            print(f"Not found: {img_path}")
    
    # Query with text
    if any(os.path.exists(p) for p, _ in image_docs):
        result = rag.query("What were the sales figures?", top_k=3)
        print(f"\nAnswer: {result.content}")
    else:
        print("\nNo images found - skipping query")


def example_multimodal_query():
    """Query with both text and image."""
    from vl_rag_graph_rlm.rag.provider import MultimodalVLRAGGraphRLM
    
    print("\n" + "=" * 60)
    print("Example 4: Multimodal Query (Text + Image)")
    print("=" * 60)
    
    rag = MultimodalVLRAGGraphRLM(
        llm_provider="openrouter",
        llm_model="gpt-4o",
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",
    )
    
    # Add mixed content
    rag.add_text("The company headquarters is located in San Francisco.")
    rag.add_text("Our new product line includes smart watches and fitness trackers.")
    
    image_path = "product_photo.jpg"
    if os.path.exists(image_path):
        rag.add_image(image_path, description="Product photo of smart watch")
    
    # Query with both text and image
    query_image = "query_watch.jpg"
    if os.path.exists(query_image):
        result = rag.query_multimodal(
            "What products do we offer?",
            image_path=query_image,
            top_k=3
        )
        print(f"\nAnswer: {result.content}")
    else:
        print(f"\nQuery image not found: {query_image}")


def example_direct_store_usage():
    """Using the MultimodalVectorStore directly without RLM."""
    from vl_rag_graph_rlm.rag.multimodal_store import create_multimodal_store
    from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder
    
    print("\n" + "=" * 60)
    print("Example 5: Direct Vector Store Usage")
    print("=" * 60)
    
    # Create store with persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = os.path.join(tmpdir, "store.json")
        
        store = create_multimodal_store(
            model_name="Qwen/Qwen3-VL-Embedding-2B",
            storage_path=storage_path,
            use_reranker=False,
        )
        
        # Add documents
        store.add_text("Python is a popular programming language.")
        store.add_text("JavaScript is used for web development.")
        
        # Search
        results = store.search("What is Python used for?", top_k=2)
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.content[:50]}...")
        
        # Test persistence
        store2 = create_multimodal_store(
            model_name="Qwen/Qwen3-VL-Embedding-2B",
            storage_path=storage_path,
            use_reranker=False,
        )
        
        print(f"\nReloaded store has {len(store2.documents)} documents")


def example_custom_instruction():
    """Using custom embedding instructions for task-specific retrieval."""
    from vl_rag_graph_rlm.rag.provider import MultimodalVLRAGGraphRLM
    
    print("\n" + "=" * 60)
    print("Example 6: Custom Embedding Instructions")
    print("=" * 60)
    
    rag = MultimodalVLRAGGraphRLM(
        llm_provider="openrouter",
        llm_model="gpt-4o",
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",
    )
    
    # Add documents with custom instruction
    rag.add_text(
        "The quick brown fox jumps over the lazy dog.",
        metadata={"type": "pangram"}
    )
    
    # Search with custom instruction
    from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder
    
    embedder = create_qwen3vl_embedder("Qwen/Qwen3-VL-Embedding-2B")
    
    # Custom instruction for specific task
    instruction = "Find pangrams containing all letters of the alphabet."
    embedding = embedder.embed_text(
        "A sentence with every letter",
        instruction=instruction
    )
    
    print(f"\nCustom embedding generated with dimension: {len(embedding)}")
    print("Custom instructions help tailor embeddings for specific tasks.")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Qwen3-VL Multimodal RAG Examples")
    print("=" * 60)
    print("\nThese examples demonstrate Qwen3-VL integration for RLM.")
    print("Some examples require local files (PDFs, images) to be present.")
    print("\nMake sure you have the required dependencies installed:")
    print("  pip install unified-rlm[qwen3vl]")
    print("\nNote: First run will download model weights (~4-16GB)")
    print("=" * 60)
    
    # Run examples
    try:
        example_basic_text_rag()
    except Exception as e:
        print(f"\nError in text RAG example: {e}")
    
    try:
        example_pdf_rag()
    except Exception as e:
        print(f"\nError in PDF RAG example: {e}")
    
    try:
        example_image_rag()
    except Exception as e:
        print(f"\nError in image RAG example: {e}")
    
    try:
        example_multimodal_query()
    except Exception as e:
        print(f"\nError in multimodal query example: {e}")
    
    try:
        example_direct_store_usage()
    except Exception as e:
        print(f"\nError in direct store example: {e}")
    
    try:
        example_custom_instruction()
    except Exception as e:
        print(f"\nError in custom instruction example: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
