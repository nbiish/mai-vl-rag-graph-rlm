# VL-RAG-Graph-RLM

**Vision-Language RAG Graph Recursive Language Models** — a unified multimodal document analysis framework combining **Qwen3-VL embeddings**, **hybrid RAG with RRF fusion**, **cross-attention reranking**, **knowledge graph extraction**, and **recursive LLM reasoning** across **17 LLM provider templates** (15 CLI providers + 2 generic compatible).

## The Six Pillars

| # | Pillar | Component | Cost |
|---|--------|-----------|------|
| 1 | **VL** | Qwen3-VL-Embedding-2B — unified text + image embeddings | FREE (local) |
| 2 | **RAG** | Hybrid search (dense cosine + keyword) with RRF fusion | FREE (local) |
| 3 | **Reranker** | Qwen3-VL-Reranker-2B — cross-attention relevance scoring | FREE (local) |
| 4 | **Graph** | Knowledge graph extraction via RLM | LLM cost |
| 5 | **RLM** | Recursive Language Model with sandboxed REPL | LLM cost |
| 6 | **Report** | Markdown report with sources, scores, and metadata | FREE |

## Installation

```bash
# Clone
git clone https://github.com/nbiish/mai-vl-rag-graph-rlm.git
cd mai-vl-rag-graph-rlm

# Install core
uv pip install -e .

# Install with Qwen3-VL multimodal support
uv pip install -e ".[qwen3vl]"

# Install everything
uv pip install -e ".[all]"
```

## Quick Start

### CLI

```bash
# Set your API key
export SAMBANOVA_API_KEY=your_key_here

# Process a PowerPoint presentation
vrlmrag --provider sambanova presentation.pptx

# Process with custom query and save report
vrlmrag --provider nebius document.pptx -q "Summarize key findings" -o report.md

# Override model and tune recursion
vrlmrag --provider openrouter ./docs --model kimi/kimi-k2.5 --max-depth 5

# List all providers and API key status
vrlmrag --list-providers

# Show version
vrlmrag --version
```

### Python API — High-Level Pipeline

```python
from vl_rag_graph_rlm import create_pipeline

pipeline = create_pipeline(
    llm_provider="sambanova",
    embedding_model="Qwen/Qwen3-VL-Embedding-2B",
    use_reranker=True,
)

pipeline.add_pptx("presentation.pptx", extract_images=True)
result = pipeline.query("What are the main topics covered?")
print(result.answer)
```

### Python API — Manual Pipeline

```python
from vl_rag_graph_rlm import VLRAGGraphRLM
from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder, create_qwen3vl_reranker
from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore
from vl_rag_graph_rlm.rag import ReciprocalRankFusion

# 1. VL — Embed text + images
embedder = create_qwen3vl_embedder(device="mps")  # or "cuda", "cpu"
store = MultimodalVectorStore(embedding_provider=embedder)
store.add_text("Machine learning uses statistical methods...", metadata={"page": 1})
store.add_image("diagram.png", description="Architecture diagram")

# 2. RAG — Hybrid search
dense_results = store.search("How does ML work?", top_k=20)

# 3. Reranker — Cross-attention reranking
reranker = create_qwen3vl_reranker(device="mps")
docs = [{"text": store.get(r.id).content} for r in dense_results if store.get(r.id)]
reranked = reranker.rerank(query={"text": "How does ML work?"}, documents=docs)

# 4. Graph — Knowledge graph extraction
rlm = VLRAGGraphRLM(provider="sambanova", temperature=0.0)
kg = rlm.completion("Extract entities and relationships.", context)

# 5. RLM — Recursive query answering
result = rlm.completion("How does ML work?", context)
print(result.response)
```

## Supported Providers

| Provider | Default Model | Context | Token Limits | Notes |
|----------|--------------|---------|--------------|-------|
| `sambanova` | DeepSeek-V3.2 | 128K | **200K TPD** ⚠️ | 200+ tok/s, requires low context budget |
| `nebius` | MiniMax-M2.1 | 128K | Unlimited | No daily token limits |
| `openrouter` | minimax-m2.1 | varies | Per-model | 200+ models, pay-per-token |
| `openai` | gpt-4o-mini | 128K | Per-tier | Generous limits |
| `anthropic` | claude-3-5-haiku | 200K | Per-tier | Generous limits |
| `gemini` | gemini-1.5-flash | 1M | Per-tier | Very high context |
| `groq` | llama-3.3-70b-versatile | 128K | Per-tier | Ultra-fast inference |
| `deepseek` | deepseek-chat | 128K | Per-tier | Strong reasoning |
| `mistral` | mistral-large | 128K | Per-tier | Generous limits |
| `fireworks` | llama-3.1-70b | 128K | Per-tier | Serverless |
| `together` | llama-3.1-70b-turbo | 128K | Per-tier | Generous limits |
| `zenmux` | ernie-5.0-thinking | varies | Per-model | Chinese AI models |
| `zai` | glm-4.7 | 128K | Per-tier | Zhipu AI |
| `azure_openai` | gpt-4o | 128K | Per-deployment | Enterprise |
| `cerebras` | llama-3.3-70b | 128K | Per-tier | Ultra-fast wafer-scale |
| `openai_compatible` | user-configured | varies | Per-provider | Generic OpenAI-compatible |
| `anthropic_compatible` | user-configured | varies | Per-provider | Generic Anthropic-compatible |

### Token Budget Architecture

**Important:** All providers except SambaNova have generous or unlimited token limits:

- **Nebius**: No daily limits — can use full 128K context
- **OpenRouter, OpenAI, Anthropic, Gemini, Groq, etc.**: Per-tier limits that are typically generous
- **SambaNova**: **200K TPD free tier** — requires careful context budgeting

The system automatically adjusts context budgets per provider:

| Provider | Context Budget | Rationale |
|----------|---------------|-----------|
| SambaNova | 8,000 chars (~2K tokens) | Fits within 200K TPD for multiple queries |
| Nebius | 100,000 chars (~25K tokens) | No daily limits, use full context |
| Others | 32,000-64,000 chars | Balanced for typical tier limits |

To upgrade SambaNova limits, consider the Developer tier (12K RPD, no TPD limit).

## CLI Reference

```
usage: vrlmrag [-h] [--version] [--list-providers] [--provider NAME]
               [--query QUERY] [--output OUTPUT] [--model MODEL]
               [--max-depth N] [--max-iterations N] [PATH]
```

| Flag | Short | Description |
|------|-------|-------------|
| `--provider NAME` | `-p` | LLM provider (required) |
| `PATH` | | File or folder to process (PPTX, TXT, MD) |
| `--query QUERY` | `-q` | Custom query (default: auto-generated) |
| `--output PATH` | `-o` | Output markdown report path |
| `--model MODEL` | `-m` | Override default model |
| `--max-depth N` | | RLM recursion depth (default: 3) |
| `--max-iterations N` | | RLM iterations per call (default: 10) |
| `--list-providers` | | Show all providers + API key status |
| `--version` | `-V` | Print version |

Backward-compatible: `--samba-nova PATH`, `--nebius PATH`

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Each provider uses `{PROVIDER}_API_KEY`, with optional `{PROVIDER}_MODEL` and `{PROVIDER}_RECURSIVE_MODEL` overrides. See `.env.example` for all options.

## How It Works

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Documents   │ →  │  Qwen3-VL    │ →  │ Hybrid Search│ →  │  RLM         │
│  PPTX/TXT/MD │    │  Embedding   │    │ + RRF Fusion │    │  Recursive   │
│  + Images    │    │  (2B, local) │    │ + Reranking  │    │  Reasoning   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                           ↓                    ↓                    ↓
                    Unified Vector       Retrieved Context    Markdown Report
                    Space (text+img)     with Scores          + Knowledge Graph
```

### RLM Workflow

1. **Context Analysis** — RLM analyzes retrieved context
2. **Code Generation** — Generates Python code to explore the data
3. **Safe Execution** — Runs code in a RestrictedPython sandbox
4. **Recursive Calls** — Breaks complex tasks into sub-tasks via `recursive_llm()`
5. **Answer Synthesis** — Returns final answer via `FINAL()` or `FINAL_VAR()`

## API Reference

### VLRAGGraphRLM

```python
VLRAGGraphRLM(
    provider: str,              # Provider name (sambanova, nebius, openrouter, ...)
    model: str = None,          # Model name (auto-detected from env/defaults)
    recursive_model: str = None,# Cheaper model for recursive sub-queries
    api_key: str = None,        # API key (or use env var)
    max_depth: int = 3,         # Maximum recursion depth
    max_iterations: int = 10,   # Max iterations per call
    temperature: float = 0.0,   # Sampling temperature
)
```

**Methods:**
- `completion(query, context)` → `CompletionResult` — Recursive completion with REPL

### MultimodalVectorStore

```python
MultimodalVectorStore(
    embedding_provider=embedder,  # Qwen3-VL embedding provider
    storage_path="store.json",    # JSON persistence path
)
```

**Methods:**
- `add_text(content, metadata)` — Embed and store text
- `add_image(image_path, description, metadata)` — Embed and store image
- `search(query, top_k)` → `List[SearchResult]` — Dense vector search
- `get(doc_id)` → `Document` — Retrieve document by ID

### Factory Functions

```python
from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder, create_qwen3vl_reranker

embedder = create_qwen3vl_embedder(device="mps")  # or "cuda", "cpu"
reranker = create_qwen3vl_reranker(device="mps")
```

## Provider Templates

Every provider has a ready-to-use template in `templates/`:

```bash
python templates/provider_sambanova.py --input document.pptx
python templates/provider_nebius.py --manual
python templates/provider_openrouter.py --input doc.pdf --query "Summarize"
```

Each template demonstrates all six pillars with both `create_pipeline()` and manual step-by-step examples.

## Troubleshooting

### Qwen3-VL won't load
```bash
pip install torch transformers>=5.1.0 qwen-vl-utils>=0.0.14 pillow torchvision
```

### Out of memory
```python
# Use CPU instead of GPU/MPS
embedder = create_qwen3vl_embedder(device="cpu")
```

### google-generativeai deprecation warning
```bash
# Known issue — migration to google-genai planned for v0.2.0
```

## License

MIT License

## Acknowledgments

- **Qwen3-VL-Embedding** — Vision-language embedding and reranking (Qwen team)
- **Paddle-ERNIE-RAG** — Hybrid search and multimodal RAG patterns
- [alexzhang13/rlm](https://github.com/alexzhang13/rlm) — Comprehensive RLM framework
- [ysz/recursive-llm](https://github.com/ysz/recursive-llm) — Minimal recursive LLM implementation
