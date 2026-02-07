# VL-RAG-Graph-RLM Project Template

A complete template for building **beyond expert** RAG applications with unlimited context, vision support, and knowledge graphs.

## Features

- **Unlimited Context**: Recursive processing handles documents of ANY size
- **Vision RAG**: Process PDFs with images, screenshots, diagrams alongside text
- **Knowledge Graphs**: Extract entities/relationships from unstructured data
- **Hybrid Search**: Dense + keyword + RRF fusion + composite reranking
- **SOTA Models**: DeepSeek-V3.2 via SambaNova (200+ tok/sec, 128K context)

## Quick Start

### 1. Clone and Setup

```bash
# Clone this template
git clone <your-repo> my-rag-project
cd my-rag-project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
export SAMBANOVA_API_KEY=your_key_here
# Get key: https://cloud.sambanova.ai
```

### 3. Run Examples

```bash
python src/main.py
```

## Project Structure

```
my-rag-project/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config.py               # Configuration
│   ├── processors/             # Document processors
│   │   ├── __init__.py
│   │   ├── pdf_processor.py    # PDF + image extraction
│   │   ├── code_processor.py   # Codebase analysis
│   │   └── graph_builder.py    # Knowledge graph construction
│   └── pipelines/              # RAG pipelines
│       ├── __init__.py
│       ├── multimodal_rag.py   # Vision + text RAG
│       └── unlimited_context.py # Recursive processing
├── data/
│   ├── documents/              # Input documents
│   ├── images/                 # Images/diagrams
│   └── vector_store/           # Persistent storage
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## Usage Examples

### Process Large Technical Manual

```python
from src.pipelines.multimodal_rag import MultimodalRAGPipeline

pipeline = MultimodalRAGPipeline()

# Add 500-page PDF with 50 figures
pipeline.add_pdf("technical_manual.pdf", extract_images=True)

# Query across text AND figures
result = pipeline.query(
    "Explain the circuit diagram in Figure 12 and how it relates to the text description"
)
```

### Analyze Entire Codebase

```python
from src.pipelines.unlimited_context import CodebaseAnalyzer

analyzer = CodebaseAnalyzer()
analyzer.index_codebase("./my-project")

# Query across all files + diagrams
result = analyzer.query("""
    Find all authentication-related code and explain how it works
    with reference to the architecture diagram in docs/
""")
```

### Build Knowledge Graph

```python
from src.processors.graph_builder import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder()
builder.extract_from_documents("./research_papers/")

# Complex multi-hop queries
result = builder.query("""
    What techniques proposed in paper A were later
    improved by authors of paper B, and what were the results?
""")
```

## Advanced Configuration

### Custom Embedding Models

```python
# Use Qwen3-VL for multimodal (text + image + video)
from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder

embedder = create_qwen3vl_embedder("Qwen/Qwen3-VL-Embedding-2B")
```

### Milvus Vector Store (Production)

```python
from vl_rag_graph_rlm.rag import MilvusVectorStore

store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="my_documents",
    embedding_client=embedder
)
```

### Recursive RLM with Custom Depth

```python
from vl_rag_graph_rlm import VLRAGGraphRLM

rlm = VLRAGGraphRLM(
    provider="sambanova",
    model="DeepSeek-V3.2",
    max_depth=10,          # Handle 10x context window
    max_iterations=100,    # Up to 100 recursive calls
)
```

## Dependencies

Core:
- vl-rag-graph-rlm >= 0.1.0
- python-dotenv >= 1.0.0

Optional (for full features):
- torch >= 2.0.0
- transformers >= 4.40.0
- pillow >= 10.0.0
- pymilvus >= 2.4.0

See `requirements.txt` for complete list.

## License

MIT
