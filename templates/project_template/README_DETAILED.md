# VL-RAG-Graph-RLM Project Template

A complete, standalone project template for building **beyond expert** RAG applications with unlimited context, vision support, and knowledge graphs.

## What's Included

This template provides a complete, working project structure that you can clone and immediately use for:

- **Unlimited Context**: Process documents of ANY size via recursive chunking
- **Vision RAG**: Handle PDFs with images, screenshots, diagrams alongside text  
- **Knowledge Graphs**: Extract entities/relationships from unstructured data
- **Hybrid Search**: Dense + keyword + RRF fusion + composite reranking
- **SOTA Models**: DeepSeek-V3.2 via SambaNova (200+ tok/sec, 128K context)

## Quick Start (New Project)

```bash
# 1. Clone this template to your new project
git clone https://github.com/yourusername/vl-rag-graph-template my-project
cd my-project
rm -rf .git  # Remove template git history
git init     # Start fresh

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env with your keys

# 5. Run the demo
python src/main.py
```

## Project Structure

```
my-project/
├── src/
│   ├── __init__.py
│   ├── main.py              # Entry point with all examples
│   ├── config.py            # Configuration management
│   ├── processors/          # Document processing modules
│   │   ├── __init__.py
│   │   ├── pdf_processor.py # Extract text + images from PDFs
│   │   ├── code_analyzer.py # Parse codebases
│   │   └── kg_builder.py    # Knowledge graph construction
│   ├── pipelines/           # RAG pipeline implementations
│   │   ├── __init__.py
│   │   ├── multimodal_rag.py    # Vision + text RAG
│   │   └── recursive_rag.py     # Unlimited context processing
│   └── utils/               # Helper utilities
│       ├── __init__.py
│       └── embeddings.py    # Embedding model utilities
├── data/
│   ├── input/               # Your documents go here
│   ├── processed/           # Extracted content
│   └── vector_store/        # Vector DB persistence
├── notebooks/               # Jupyter examples
├── tests/
├── .env.example
├── requirements.txt
├── setup.py
└── README.md
```

## Usage Examples

### Example 1: Process Large Technical Manual with Figures

```python
from src.pipelines.multimodal_rag import MultimodalRAGPipeline

pipeline = MultimodalRAGPipeline(
    provider="sambanova",
    model="DeepSeek-V3.2",
)

# Add 500-page PDF with 50 figures
pipeline.add_pdf("data/input/technical_manual.pdf", extract_images=True)

# Query across text AND figures
result = pipeline.query(
    "Explain the circuit diagram in Figure 12 and relate it to the text"
)
print(result.answer)
```

### Example 2: Analyze Entire Codebase + Diagrams

```python
from src.pipelines.recursive_rag import CodebaseAnalyzer

analyzer = CodebaseAnalyzer()
analyzer.index_codebase("/path/to/my-project")

result = analyzer.query("""
    Find the authentication flow and explain it with 
    reference to the architecture diagram in docs/
""")
```

### Example 3: Build Knowledge Graph from Research Papers

```python
from src.processors.kg_builder import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder()
builder.extract_from_directory("data/input/papers/")

result = builder.query("""
    What methods from paper A were improved by authors 
    of paper B, and what were the quantitative results?
""")
```

### Example 4: Unlimited Context Document Processing

```python
from vl_rag_graph_rlm import VLRAGGraphRLM

rlm = VLRAGGraphRLM(
    provider="sambanova",
    model="DeepSeek-V3.2",
    max_depth=10,
    max_iterations=100,
)

with open("huge_document.txt") as f:
    huge_doc = f.read()

result = rlm.completion(
    "Summarize the key findings across all sections",
    huge_doc
)
```

## Dependencies

Core:
```
vl-rag-graph-rlm >= 0.1.0
python-dotenv >= 1.0.0
```

Optional (multimodal):
```
torch >= 2.0.0
transformers >= 4.40.0
pillow >= 10.0.0
```

Optional (production):
```
pymilvus >= 2.4.0
```

Full installation:
```bash
pip install vl-rag-graph-rlm[all]
```

## License

MIT
