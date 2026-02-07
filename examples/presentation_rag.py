"""
PowerPoint RAG Example - Multimodal Presentation Analysis

This example demonstrates how to process PowerPoint presentations for RAG:
1. Extract text, tables, and speaker notes from slides
2. Extract embedded images from presentations
3. Generate multimodal embeddings for both text and images
4. Perform hybrid search across all content
5. Use RLM for expert reasoning about presentation content

Requirements:
    pip install vl-rag-graph-rlm[qwen3vl]
    pip install python-pptx Pillow

    # Set API key (any provider supported)
    export OPENROUTER_API_KEY=your_key_here
    # or
    export OPENAI_API_KEY=your_key_here
"""

import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import MultimodalRAGPipeline, create_pipeline


def example_basic_presentation_rag():
    """Basic PowerPoint processing with the Writing Tutorial presentation."""
    print("=" * 70)
    print("Example 1: Basic PowerPoint RAG")
    print("=" * 70)
    
    # Create pipeline with default settings
    pipeline = create_pipeline(
        llm_provider="openrouter",
        llm_model="kimi/kimi-k2.5",
        verbose=True
    )
    
    # Process the Writing Tutorial presentation
    pptx_path = "Writing Tutorial 2022.pptx"
    
    if not Path(pptx_path).exists():
        print(f"\nFile not found: {pptx_path}")
        print("Make sure the PowerPoint file is in the examples directory.")
        return
    
    print(f"\nProcessing PowerPoint: {pptx_path}")
    doc_ids = pipeline.add_presentation(
        pptx_path=pptx_path,
        extract_images=True,   # Extract embedded images
        extract_notes=True,    # Extract speaker notes
        extract_tables=True    # Extract tables
    )
    print(f"Added {len(doc_ids)} documents from presentation")
    
    # Query about the presentation content
    questions = [
        "What are the main topics covered in this writing tutorial?",
        "What tips are provided for structuring an essay?",
        "Are there any examples of good vs bad writing?",
        "What visual aids or diagrams are used to explain concepts?",
    ]
    
    for question in questions:
        print(f"\n{'='*70}")
        print(f"Q: {question}")
        print('='*70)
        result = pipeline.query(question, top_k=5)
        print(f"\nA: {result.answer}")
        print(f"\nSources used: {len(result.sources)}")
        if result.images:
            print(f"Images referenced: {len(result.images)}")
        print(f"Cost: ${result.total_cost:.4f} | Time: {result.execution_time:.2f}s")


def example_presentation_with_slides_filtering():
    """Process only specific slides from a presentation."""
    print("\n" + "=" * 70)
    print("Example 2: Selective Slide Processing")
    print("=" * 70)
    
    pipeline = create_pipeline(verbose=True)
    
    pptx_path = "Writing Tutorial 2022.pptx"
    if not Path(pptx_path).exists():
        print(f"\nFile not found: {pptx_path}")
        return
    
    # Process only first 5 slides
    print("\nProcessing only slides 1-5...")
    doc_ids = pipeline.add_presentation(
        pptx_path=pptx_path,
        slides=[1, 2, 3, 4, 5],  # Specific slides only
        extract_images=True,
        extract_notes=True
    )
    print(f"Added {len(doc_ids)} documents from slides 1-5")
    
    # Query about specific slides
    result = pipeline.query(
        "What is introduced in the first few slides?",
        top_k=3
    )
    print(f"\nAnswer: {result.answer}")


def example_multimodal_presentation_analysis():
    """
    Deep multimodal analysis of presentation content.
    
    This example shows how the RAG system can reason about both
text and images from the presentation together.
    """
    print("\n" + "=" * 70)
    print("Example 3: Multimodal Presentation Analysis")
    print("=" * 70)
    
    # Initialize with strong multimodal capabilities
    pipeline = MultimodalRAGPipeline(
        llm_provider="openrouter",
        llm_model="kimi/kimi-k2.5",
        recursive_model="google/gemini-2.0-flash-001",
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",
        use_reranker=True,
        top_k=10,
        use_hybrid_search=True,
        verbose=True
    )
    
    pptx_path = "Writing Tutorial 2022.pptx"
    if not Path(pptx_path).exists():
        print(f"\nFile not found: {pptx_path}")
        return
    
    print(f"\nProcessing presentation with multimodal analysis...")
    doc_ids = pipeline.add_presentation(
        pptx_path=pptx_path,
        extract_images=True,
        extract_notes=True,
        extract_tables=True
    )
    print(f"Added {len(doc_ids)} documents")
    
    # Complex multimodal queries
    questions = [
        "How do the visual diagrams complement the written explanations?",
        "What is the relationship between the images and the text on each slide?",
        "Explain the writing process as shown in both text and any flowcharts/diagrams",
        "What do the speaker notes add that isn't in the slide content?",
    ]
    
    for question in questions:
        print(f"\n{'='*70}")
        print(f"Q: {question}")
        print('='*70)
        result = pipeline.query(question, top_k=8)
        print(f"\nA: {result.answer}")
        
        # Show source information
        print("\nSources:")
        for i, source in enumerate(result.sources[:3], 1):
            meta = source['metadata']
            if meta.get('type') == 'presentation_slide':
                print(f"  {i}. Slide {meta.get('slide_number')} ({meta.get('layout', 'unknown layout')})")
            elif meta.get('type') == 'presentation_image':
                print(f"  {i}. Image from slide {meta.get('slide_number')}")
            if meta.get('has_notes'):
                print(f"     (includes speaker notes)")
            if meta.get('has_table'):
                print(f"     (includes table data)")


def example_presentation_comparison():
    """
    Compare content across multiple presentations.
    
    Note: This example assumes you have multiple .pptx files.
    It demonstrates the system's ability to reason about
    multiple presentations together.
    """
    print("\n" + "=" * 70)
    print("Example 4: Presentation Comparison (Multiple Files)")
    print("=" * 70)
    
    pipeline = create_pipeline(
        llm_provider="openrouter",
        llm_model="kimi/kimi-k2.5",
        storage_path="./presentations_kb.pkl",
        verbose=True
    )
    
    # Look for all .pptx files in the directory
    pptx_files = list(Path(".").glob("*.pptx"))
    
    if len(pptx_files) < 2:
        print(f"\nFound only {len(pptx_files)} presentation(s).")
        print("Add more .pptx files to demonstrate comparison.")
        
        # Still process the one we have
        if pptx_files:
            print(f"\nProcessing: {pptx_files[0]}")
            pipeline.add_presentation(
                pptx_path=pptx_files[0],
                extract_images=True,
                extract_notes=True,
                metadata={"presentation": pptx_files[0].stem}
            )
        return
    
    # Process all presentations
    for pptx_file in pptx_files:
        print(f"\nProcessing: {pptx_file}")
        pipeline.add_presentation(
            pptx_path=pptx_file,
            extract_images=True,
            extract_notes=True,
            metadata={"presentation": pptx_file.stem}
        )
    
    # Compare presentations
    result = pipeline.query(
        "Compare the main topics and teaching approaches across all presentations. "
        "What are the common themes and unique insights from each?",
        top_k=10
    )
    print(f"\nComparison Analysis:\n{result.answer}")


def example_cost_optimized_presentation_rag():
    """Most cost-effective setup for presentation analysis."""
    print("\n" + "=" * 70)
    print("Example 5: Cost-Optimized Presentation RAG")
    print("=" * 70)
    
    # Use cheapest viable models
    pipeline = MultimodalRAGPipeline(
        llm_provider="openrouter",
        llm_model="google/gemini-2.0-flash-001",      # Very cheap
        recursive_model="solar-pro/solar-pro-3:free",  # Free tier
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",  # Free, local
        use_reranker=True,
        max_depth=2,
        max_iterations=5,
        verbose=True
    )
    
    pptx_path = "Writing Tutorial 2022.pptx"
    if not Path(pptx_path).exists():
        print(f"\nFile not found: {pptx_path}")
        return
    
    # Skip image extraction for speed/cost
    print(f"\nProcessing {pptx_path} (text only, cost-optimized)...")
    doc_ids = pipeline.add_presentation(
        pptx_path=pptx_path,
        extract_images=False,  # Skip images for speed
        extract_notes=True     # Keep notes for context
    )
    print(f"Added {len(doc_ids)} text documents")
    
    # Simple query
    result = pipeline.query(
        "Summarize the key writing advice in bullet points",
        top_k=5
    )
    print(f"\nAnswer: {result.answer}")
    print(f"\nTotal Cost: ${result.total_cost:.4f}")
    print(f"Target: Keep under $0.01 per query")


def example_presentation_search_only():
    """Search presentation content without LLM generation."""
    print("\n" + "=" * 70)
    print("Example 6: Presentation Content Search")
    print("=" * 70)
    
    pipeline = create_pipeline(verbose=True)
    
    pptx_path = "Writing Tutorial 2022.pptx"
    if not Path(pptx_path).exists():
        print(f"\nFile not found: {pptx_path}")
        return
    
    print(f"\nProcessing {pptx_path}...")
    pipeline.add_presentation(
        pptx_path=pptx_path,
        extract_images=True,
        extract_notes=True
    )
    
    # Search for specific content
    search_terms = [
        "thesis statement",
        "paragraph structure",
        "citation",
        "introduction",
        "conclusion"
    ]
    
    print("\nSearching for key concepts:")
    for term in search_terms:
        results = pipeline.search(term, top_k=3)
        print(f"\n'{term}':")
        for r in results:
            meta = r.metadata
            if meta.get('type') == 'presentation_slide':
                print(f"  - Slide {meta.get('slide_number')} "
                      f"(score: {r.composite_score:.2f})")


def main():
    """Run all PowerPoint RAG examples."""
    # Check for API key
    if not any([
        os.getenv("OPENROUTER_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY")
    ]):
        print("Warning: No API key found. Set one of:")
        print("  export OPENROUTER_API_KEY=your_key")
        print("  export OPENAI_API_KEY=your_key")
        print("  export ANTHROPIC_API_KEY=your_key")
        return
    
    print("\n" + "=" * 70)
    print("PowerPoint RAG Examples - VL-RAG-Graph-RLM")
    print("=" * 70)
    print("\nThese examples demonstrate processing PowerPoint presentations")
    print("for multimodal RAG with text, images, tables, and speaker notes.")
    
    # Run examples
    examples = [
        ("Basic Presentation RAG", example_basic_presentation_rag),
        ("Selective Slide Processing", example_presentation_with_slides_filtering),
        ("Multimodal Analysis", example_multimodal_presentation_analysis),
        ("Presentation Comparison", example_presentation_comparison),
        ("Cost-Optimized RAG", example_cost_optimized_presentation_rag),
        ("Search Only", example_presentation_search_only),
    ]
    
    for name, func in examples:
        try:
            func()
        except FileNotFoundError as e:
            print(f"\nSkipping '{name}' - file not found: {e}")
        except ImportError as e:
            print(f"\nSkipping '{name}' - missing dependency: {e}")
            print("Install with: pip install python-pptx Pillow")
        except Exception as e:
            print(f"\nError in '{name}': {e}")
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
