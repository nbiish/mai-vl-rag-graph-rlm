# VL_RAG_GRAPH_RLM

Vision-Language RAG Graph Recursive Language Models - A comprehensive framework combining **Qwen3-VL multimodal embeddings**, **SOTA RAG with hybrid search**, **graph-based reasoning**, and **recursive LLM processing** for intelligent document analysis with PDF and image support.

## Overview

This unified toolkit processes documents with images using a pipeline of:

1. **Document Parsing** (Paddle OCR) - Extract text, tables, and images from PDFs
2. **Multimodal Embedding** (Qwen3-VL) - Generate embeddings for text + images
3. **Vector Search + Reranking** - Hybrid retrieval with semantic + keyword search
4. **Recursive Reasoning** (RLM) - Break complex queries into manageable chunks
5. **Cheap SOTA LLMs** - Generate final answers with modern cost-effective models

## Why This Stack?

| Component | Model | Cost | Why |
|-----------|-------|------|-----|
| **OCR/Layout** | Paddle OCR PP-StructureV3 | **FREE** (local) | Extracts text, tables, images from PDFs |
| **Embedding** | Qwen3-VL-Embedding-2B | **FREE** (local) | 2B params, SOTA on MMEB-v2 |
| **Reranking** | Qwen3-VL-Reranker-2B | **FREE** (local) | 2B params, beats much larger models |
| **LLM** | kimi-k2.5, claude-3.5-haiku, gpt-4o-mini | **~$0.10-0.30/M** | High quality, cheap via OpenRouter |
| **Reasoning** | DeepSeek-R1, o3-mini | **~$0.50-2/M** | Strong reasoning when needed |

## Installation

```bash
# Clone with all submodules
git clone --recursive https://github.com/yourusername/vl-rag-graph-rlm.git
cd vl-rag-graph-rlm

# Install core framework
uv pip install -e .

# Install with Qwen3-VL support (for local embeddings)
uv pip install -e ".[qwen3vl]"

# Install Paddle OCR dependencies for PDF processing
pip install paddlepaddle paddleocr pymupdf
```

## Quick Start: Complete Document Pipeline

### 1. Setup Environment

```bash
# .env file
OPENROUTER_API_KEY=your_key_here

# Optional: if using direct providers
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

### 2. Full Pipeline Example

```python
from vl_rag_graph_rlm import VLRAGGraphRLM
from vl_rag_graph_rlm.rag.qwen3vl import Qwen3VLEmbeddingProvider, Qwen3VLRerankerProvider
from vl_rag_graph_rlm.rag import MultimodalDocumentStore, RAGConfig
from pathlib import Path

# Step 1: Initialize local multimodal embedding + reranking (FREE)
print("Loading Qwen3-VL models...")
embedder = Qwen3VLEmbeddingProvider(
    model_name_or_path="Qwen/Qwen3-VL-Embedding-2B"  # 2B params, runs on CPU/GPU
)
reranker = Qwen3VLRerankerProvider(
    model_name_or_path="Qwen/Qwen3-VL-Reranker-2B"
)

# Step 2: Create multimodal document store
store = MultimodalDocumentStore(
    embedding_provider=embedder,
    reranker=reranker,
    storage_path="./knowledge_base.pkl"
)

# Step 3: Process PDF with Paddle OCR (extracts text + images)
print("Processing PDF...")
store.add_pdf(
    pdf_path="research_paper.pdf",
    extract_images=True,      # Extract figures/diagrams
    extract_tables=True,      # Extract tables as structured data
    ocr_engine="paddle"       # Use Paddle OCR for best accuracy
)

# Step 4: Add individual images if needed
store.add_image(
    image_path="diagram.png",
    description="System architecture diagram"
)

# Step 5: Initialize cheap SOTA LLM via OpenRouter
print("Initializing LLM...")
vlrag = VLRAGGraphRLM(
    provider="openrouter",
    model="kimi/kimi-k2.5",           # Excellent quality, ~$0.50/M tokens
    recursive_model="google/gemini-2.0-flash-001",  # Even cheaper for recursion
    max_depth=3
)

# Step 6: Query with automatic retrieval
print("Querying...")
result = vlrag.query_with_rag(
    query="Explain the methodology shown in Figure 3",
    document_store=store,
    top_k=5,
    use_vision=True  # Include image content in context
)

print(f"\nAnswer: {result.response}")
print(f"\nSources: {result.sources}")
print(f"Cost: ${result.usage_summary.total_cost:.4f}")
```

## Component Deep Dive

### Qwen3-VL Multimodal Embedding (Local, Free)

```python
from vl_rag_graph_rlm.rag.qwen3vl import Qwen3VLEmbeddingProvider

# Load 2B model (runs on CPU, faster on GPU)
embedder = Qwen3VLEmbeddingProvider(
    model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
    device="cuda"  # or "cpu", "mps" for Apple Silicon
)

# Embed text
text_emb = embedder.embed_text("The quick brown fox")

# Embed image
image_emb = embedder.embed_image("path/to/chart.png")

# Embed text + image together (multimodal)
mm_emb = embedder.embed_multimodal(
    text="What does this chart show?",
    image="path/to/chart.png"
)

# Batch processing for efficiency
documents = [
    {"text": "Document 1 content..."},
    {"text": "Caption", "image": "figure1.png"},
    {"image": "diagram.png"}
]
embeddings = embedder.embed_batch(documents)
```

### Qwen3-VL Reranker (Local, Free)

```python
from vl_rag_graph_rlm.rag.qwen3vl import Qwen3VLRerankerProvider

reranker = Qwen3VLRerankerProvider(
    model_name_or_path="Qwen/Qwen3-VL-Reranker-2B"
)

# Rerank retrieved documents
query = "What are the experimental results?"
documents = [
    {"text": "Results section..."},
    {"text": "Figure 5 shows...", "image": "results_chart.png"},
    {"text": "Related work..."}
]

scores = reranker.rerank(query, documents)
# Returns ranked list with relevance scores
```

### Paddle OCR Document Processing (Local, Free)

```python
from vl_rag_graph_rlm.rag.multimodal_store import extract_pdf_with_paddle

# Extract everything from PDF: text, tables, images, layout
sections = extract_pdf_with_paddle(
    pdf_path="document.pdf",
    extract_images=True,
    extract_tables=True,
    language="en",  # or "ch" for Chinese
    save_images_to="./extracted_images"
)

# Returns structured sections:
# - text blocks with page numbers
# - table data as markdown
# - image paths with captions
# - layout metadata (headers, paragraphs, etc.)

for section in sections:
    print(f"Page {section.page_num}: {section.type}")
    if section.type == "image":
        print(f"  Image: {section.image_path}")
        print(f"  Caption: {section.caption}")
```

### RAG with Hybrid Search

```python
from vl_rag_graph_rlm.rag import (
    MultimodalDocumentStore,
    RAGConfig,
    ReciprocalRankFusion,
    MultiFactorReranker
)

# Configure hybrid search
config = RAGConfig(
    top_k=10,
    dense_weight=4.0,      # Vector similarity weight
    keyword_weight=1.0,    # BM25 keyword weight
    rerank_top_k=5,        # Rerank top N results
    context_format="citations"  # Include source citations
)

# Create store with local embeddings
store = MultimodalDocumentStore(
    embedding_provider=embedder,
    reranker=reranker,
    config=config
)

# Add documents with automatic chunking
store.add_document(
    content="Long document text...",
    images=["figure1.png", "figure2.png"],
    metadata={"source": "paper.pdf", "page": 1}
)

# Query with automatic retrieval
results = store.query(
    query="Show me the performance metrics",
    top_k=10,
    use_hybrid=True,       # Dense + keyword search
    use_reranking=True     # Qwen3-VL reranker
)
```

### Recursive LLM (RLM) for Complex Queries

```python
from vl_rag_graph_rlm import VLRAGGraphRLM

# Cost-optimized: strong model for root, cheap for recursion
vlrag = VLRAGGraphRLM(
    provider="openrouter",
    model="kimi/kimi-k2.5",              # Main reasoning
    recursive_model="google/gemini-2.0-flash-001",  # 10x cheaper recursion
    max_depth=4,
    max_iterations=20
)

# Process large document recursively
result = vlrag.completion(
    query="Summarize the key findings and identify contradictions",
    context=long_document,
    recursive_strategy="map_reduce"  # Break into chunks, process recursively
)

print(f"Answer: {result.response}")
print(f"Iterations: {result.iterations}")
print(f"Depth: {result.depth}")
```

## Recommended Cheap SOTA Models (via OpenRouter)

| Model | Use Case | Price | Quality |
|-------|----------|-------|---------|
| `kimi/kimi-k2.5` | General tasks | $0.50/M | Excellent |
| `google/gemini-2.0-flash-001` | Fast, cheap | $0.15/M | Very Good |
| `google/gemini-2.0-flash-thinking-exp-1219` | Reasoning | $0.20/M | Excellent |
| `deepseek/deepseek-chat` | Coding, analysis | $0.50/M | Excellent |
| `deepseek/deepseek-r1` | Complex reasoning | $0.60/M | Excellent |
| `anthropic/claude-3.5-haiku` | Quick tasks | $0.25/M | Very Good |
| `openai/gpt-4o-mini` | Balanced | $0.15/M | Very Good |

## Complete Examples

### Example 1: Research Paper Analysis

```python
"""
Process a research paper with figures, tables, and equations.
Answer questions about any part of the document.
"""
from vl_rag_graph_rlm import VLRAGGraphRLM
from vl_rag_graph_rlm.rag.qwen3vl import Qwen3VLEmbeddingProvider, Qwen3VLRerankerProvider
from vl_rag_graph_rlm.rag import MultimodalDocumentStore

# 1. Initialize (all local except LLM)
embedder = Qwen3VLEmbeddingProvider("Qwen/Qwen3-VL-Embedding-2B")
reranker = Qwen3VLRerankerProvider("Qwen/Qwen3-VL-Reranker-2B")
store = MultimodalDocumentStore(embedder, reranker)

# 2. Process PDF - extracts text, tables, figures
store.add_pdf("attention_is_all_you_need.pdf", extract_images=True)

# 3. Initialize cheap LLM
vlrag = VLRAGGraphRLM(
    provider="openrouter",
    model="kimi/kimi-k2.5",
    recursive_model="google/gemini-2.0-flash-001"
)

# 4. Answer questions about text AND figures
questions = [
    "What is the BLEU score improvement over RNN enc-dec?",
    "Explain the Multi-Head Attention mechanism shown in Figure 2",
    "Compare training times between models in Table 2",
    "What do the attention visualizations in Figure 3 show?"
]

for q in questions:
    result = vlrag.query_with_rag(q, store, use_vision=True)
    print(f"\nQ: {q}")
    print(f"A: {result.response[:200]}...")
```

### Example 2: Technical Documentation with Diagrams

```python
"""
Process technical docs with architecture diagrams, flowcharts, and code.
"""
from vl_rag_graph_rlm.rag.multimodal_store import extract_technical_docs

# Extract from multiple sources
docs = extract_technical_docs([
    "api_reference.pdf",
    "architecture_diagrams/",
    "README.md"
])

# Build knowledge base
store = MultimodalDocumentStore(embedder, reranker)
for doc in docs:
    if doc.type == "image":
        # Use multimodal embedding for diagrams
        store.add_image(doc.path, description=doc.ocr_text)
    else:
        store.add_document(doc.content, metadata=doc.meta)

# Query about implementation
result = vlrag.query_with_rag(
    "How does the authentication flow work? Show the relevant diagram.",
    store,
    return_images=True  # Include relevant images in response
)
```

### Example 3: Multi-Document Comparison

```python
"""
Compare multiple papers or documents on the same topic.
"""
# Add multiple papers to same store
store.add_pdf("paper_v1.pdf", metadata={"version": "v1"})
store.add_pdf("paper_v2.pdf", metadata={"version": "v2"})
store.add_pdf("competitor_paper.pdf", metadata={"version": "competitor"})

# Ask comparative questions
result = vlrag.query_with_rag(
    "Compare the evaluation metrics used across all papers. "
    "Which paper has the most comprehensive evaluation?",
    store,
    top_k=15  # Retrieve more for comparison
)
```

## Advanced Configuration

### Custom Model Routing

```python
vlrag = VLRAGGraphRLM(
    provider="openrouter",
    model="kimi/kimi-k2.5",
    # Different models for different recursion depths
    recursive_model="google/gemini-2.0-flash-001",
    # Fallback if primary is down
    fallback_model="anthropic/claude-3.5-haiku",
    # Cost limits
    max_cost_per_query=0.10  # USD
)
```

### Batch Processing

```python
# Process multiple documents efficiently
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

for doc in documents:
    store.add_pdf(doc)

# Batch query
questions = ["Q1", "Q2", "Q3"]
results = vlrag.batch_query_with_rag(questions, store)
```

### Async Support

```python
import asyncio

async def process_documents():
    # Async document processing
    await store.a_add_pdf("large_doc.pdf")
    
    # Async querying
    result = await vlrag.aquery_with_rag("Question?", store)
    return result

result = asyncio.run(process_documents())
```

## Cost Optimization Tips

1. **Use local models for embedding/reranking** (Qwen3-VL 2B) - FREE
2. **Cache embeddings** to avoid re-computing - FREE after first run
3. **Use cheaper models for recursion** (Flash, Haiku) - 10x cheaper
4. **Start with fast models**, escalate only if needed
5. **Use smaller context windows** when possible

```python
# Example: Optimized pipeline
vlrag = VLRAGGraphRLM(
    provider="openrouter",
    model="google/gemini-2.0-flash-001",  # Start cheap
    recursive_model="deepseek/deepseek-chat",  # Escalate if needed
    max_depth=2  # Limit recursion
)

# Typical cost per query: $0.01-0.05
```

## Performance Benchmarks

| Component | Model | MMEB-v2 | Speed |
|-----------|-------|---------|-------|
| Embedding | Qwen3-VL-2B | 73.2% | ~50 docs/sec (GPU) |
| Reranking | Qwen3-VL-Reranker-2B | +5.8 pts | ~20 pairs/sec (GPU) |
| vs | text-embedding-3-large | 58.9% | ~100 docs/sec (API) |
| vs | Claude-3.5-Sonnet | N/A (API only) | ~$3/M |

## Troubleshooting

### Qwen3-VL models won't load
```bash
# Install dependencies
pip install torch transformers pillow qwen-vl-utils

# For GPU acceleration
pip install flash-attn --no-build-isolation
```

### Paddle OCR errors
```bash
# Install PaddlePaddle
pip install paddlepaddle paddleocr

# For GPU
pip install paddlepaddle-gpu
```

### Out of memory
```python
# Use smaller batch sizes
embedder = Qwen3VLEmbeddingProvider(
    model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
    max_length=2048  # Reduce from default 8192
)

# Or use CPU
embedder = Qwen3VLEmbeddingProvider(device="cpu")
```


## How It Works

### Complete Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  PDF/Image  │ -> │Paddle OCR   │ -> │Qwen3-VL     │ -> │Hybrid Search│ -> │Cheap SOTA   │
│  Documents  │    │(Layout +    │    │(Multimodal  │    │+ Reranking  │    │LLM via      │
│             │    │  Extraction)│    │  Embedding) │    │             │    │OpenRouter   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                   │                   │                   │                   │
      │                   │                   │                   │                   │
   Input            Text + Images       Vector Store         Retrieved         Final
   Sources          Extracted           Built                Context           Answer
```

### The RLM Workflow

When processing complex queries:

1. **Context Analysis**: The RLM analyzes the context size and complexity
2. **Code Generation**: Generates Python code to explore the context
3. **Safe Execution**: Executes code in a RestrictedPython sandbox
4. **Recursive Calls**: Breaks large tasks into sub-tasks using `recursive_llm()`
5. **Answer Synthesis**: Combines results and returns via `FINAL()` or `FINAL_VAR()`

## API Reference

### VLRAGGraphRLM Class

```python
VLRAGGraphRLM(
    provider: str,              # API provider name (openrouter, openai, etc.)
    model: str,                 # Model name (kimi-k2.5, claude-3.5-haiku, etc.)
    recursive_model: str,       # Optional: different model for recursive calls
    api_key: str,               # Optional: API key (or use env var)
    api_base: str,              # Optional: Custom API base URL
    max_depth: int = 3,         # Maximum recursion depth
    max_iterations: int = 10,   # Max iterations per call
    temperature: float = 0.0,   # Sampling temperature
)
```

### Methods

- `completion(query, context)` - Synchronous completion with recursive processing
- `acompletion(query, context)` - Asynchronous completion
- `query_with_rag(query, document_store, **kwargs)` - RAG-enhanced query with automatic retrieval
- `batch_query_with_rag(queries, document_store)` - Batch process multiple queries
- `stats` - Property returning execution statistics and costs

### MultimodalDocumentStore Class

```python
MultimodalDocumentStore(
    embedding_provider: Qwen3VLEmbeddingProvider,  # Local embedding model
    reranker: Qwen3VLRerankerProvider,             # Local reranker
    config: RAGConfig = None,                      # Search configuration
    storage_path: str = None                       # Path for persistence
)
```

### Methods

- `add_document(content, images, metadata)` - Add text document with optional images
- `add_pdf(pdf_path, extract_images, extract_tables)` - Process PDF with Paddle OCR
- `add_image(image_path, description)` - Add single image with description
- `query(query, top_k, use_hybrid, use_reranking)` - Search documents
- `save()` / `load()` - Persist/restore the knowledge base

### Qwen3VLEmbeddingProvider Class

```python
Qwen3VLEmbeddingProvider(
    model_name_or_path: str = "Qwen/Qwen3-VL-Embedding-2B",
    device: str = "cuda",          # "cuda", "cpu", or "mps"
    max_length: int = 8192,
    torch_dtype = None,
    default_instruction: str = "Represent the user's input."
)
```

### Methods

- `embed_text(text)` - Embed text only
- `embed_image(image_path)` - Embed image only
- `embed_multimodal(text, image)` - Embed text + image together
- `embed_batch(documents)` - Batch process multiple documents

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by **Paddle-ERNIE-RAG** - Hybrid search and multimodal RAG implementation
- Inspired by **Qwen3-VL-Embedding** - Vision-language embedding and reranking techniques
- Based on [alexzhang13/rlm](https://github.com/alexzhang13/rlm) - Comprehensive RLM framework
- Based on [ysz/recursive-llm](https://github.com/ysz/recursive-llm) - Minimal recursive LLM implementation
