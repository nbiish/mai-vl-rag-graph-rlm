# VL_RAG_GRAPH_RLM

Vision-Language RAG Graph Recursive Language Models - A unified framework combining multimodal RAG, graph-based reasoning, and recursive LLM processing with support for **OpenRouter**, **ZenMux**, **z.ai**, and 100+ providers via LiteLLM.

## Features

- **Multiple Provider Support**: OpenRouter, ZenMux, z.ai, OpenAI, Anthropic, Gemini, Azure, and 100+ more via LiteLLM
- **Recursive Processing**: Automatically breaks down large contexts into manageable chunks
- **Safe Code Execution**: Uses RestrictedPython for secure REPL execution
- **Model Routing**: Use different models for root vs recursive calls (cost optimization)
- **Usage Tracking**: Track tokens and costs across all providers
- **Async Support**: Full async/await support for concurrent processing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vl-rag-graph-rlm.git
cd vl-rag-graph-rlm

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

### OpenRouter

```python
from vl_rag_graph_rlm import RLM
import os

rlm = VLRAGGraphRLM(
    provider="openrouter",
    model="anthropic/claude-3.5-sonnet",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

result = vlrag.completion(
    query="What are the key findings?",
    context="Your long document here..."
)

print(result.response)
```

### ZenMux

```python
rlm = VLRAGGraphRLM(
    provider="zenmux",
    model="gpt-4o",
    api_key=os.getenv("ZENMUX_API_KEY")
)

result = vlrag.completion("Analyze this", context=data)
print(result.response)
```

### z.ai

```python
rlm = VLRAGGraphRLM(
    provider="zai",
    model="claude-3-opus",
    api_key=os.getenv("ZAI_API_KEY")
)

result = vlrag.completion("Extract insights", context=report)
print(result.response)
```

## RAG (Retrieval-Augmented Generation)

VL_RAG_GRAPH_RLM now includes SOTA RAG capabilities inspired by Paddle-ERNIE-RAG:

- **Hybrid Search**: Combines dense vector + keyword search with RRF fusion
- **Multi-factor Reranking**: Fuzzy matching + keyword coverage + semantic scoring
- **Provider-agnostic Embeddings**: Works with any existing provider
- **Vision Support**: Analyze images alongside text

### Quick Start with RAG

```python
from vl_rag_graph_rlm.rag import RAGEnhancedVLRAGGraphRLM

# Initialize with separate LLM and embedding providers
rag = RAGEnhancedVLRAGGraphRLM(
    llm_provider="openrouter",
    llm_model="gpt-4o",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small"
)

# Add documents
rag.add_document(
    "Your document content here...",
    metadata={"source": "document.pdf", "page": 1}
)

# Query with automatic retrieval
result = rag.query("What is the main topic?")
print(result.response)
print(f"Sources: {result.sources}")
```

### Advanced RAG Usage

```python
from vl_rag_graph_rlm.rag import RAGContextProvider, create_vector_store, RAGConfig

# Custom configuration
config = RAGConfig(
    top_k=10,
    dense_weight=4.0,      # Weight for vector search
    keyword_weight=1.0,    # Weight for keyword search
    context_format="citations"  # "simple", "detailed", or "citations"
)

# Create store with specific embedding model
store = create_vector_store(
    provider="openai",
    model="text-embedding-3-large",
    storage_path="./my_knowledge_base.json"
)

# Create provider
rag = RAGContextProvider(store, config)

# Retrieve context
context = rag.retrieve("Your query here")

# Use with RLM
from vl_rag_graph_rlm import RLM
rlm = VLRAGGraphRLM(provider="openrouter", model="claude-3.5-sonnet")
result = vlrag.completion("Summarize", context=context)
```

### Vision/Multimodal RAG

```python
from vl_rag_graph_rlm.vision import VisionRAG

# Initialize vision-capable RAG
vrag = VisionRAG(
    llm_provider="openai",
    llm_model="gpt-4o",
    embedding_provider="openai"
)

# Add document with images
vrag.add_document(
    text="Document text content...",
    images=["figure1.png", "chart2.png"],
    metadata={"source": "report.pdf"}
)

# Query that may use both text and images
result = vrag.query("Explain the trends in the figures", use_vision=True)
```

## Configuration

### Environment Variables

Set these in your `.env` file or environment:

```bash
# OpenRouter
OPENROUTER_API_KEY=your_openrouter_key

# ZenMux
ZENMUX_API_KEY=your_zenmux_key

# z.ai
ZAI_API_KEY=your_zai_key

# OpenAI (for direct OpenAI usage)
OPENAI_API_KEY=your_openai_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Gemini
GOOGLE_API_KEY=your_google_key
```

### Advanced Usage

```python
from vl_rag_graph_rlm import RLM

# Cost-optimized setup: strong model for root, cheap model for recursion
rlm = VLRAGGraphRLM(
    provider="openrouter",
    model="anthropic/claude-3.5-sonnet",      # Strong model for root
    recursive_model="openai/gpt-4o-mini",     # Cheaper for recursive calls
    max_depth=5,                               # Max recursion depth
    max_iterations=20,                         # Max iterations per call
    temperature=0.0                            # Deterministic output
)

# Process large document
with open("large_document.txt") as f:
    document = f.read()

result = vlrag.completion(
    query="Summarize the key points and identify action items",
    context=document
)

print(f"Answer: {result.response}")
print(f"Provider: {result.provider}")
print(f"Model: {result.model}")
print(f"Time: {result.execution_time:.2f}s")
print(f"Usage: {result.usage_summary.to_dict()}")
```

### Async Usage

```python
import asyncio
from vl_rag_graph_rlm import RLM

async def process():
    rlm = VLRAGGraphRLM(provider="openrouter", model="gpt-4o")
    result = await rlm.acompletion("Analyze this", context=data)
    return result.response

answer = asyncio.run(process())
```

## Supported Providers

| Provider | Key Feature | Base URL |
|----------|-------------|----------|
| `openrouter` | Access to 100+ models | https://openrouter.ai/api/v1 |
| `zenmux` | Optimized routing | https://api.zenmux.ai/v1 |
| `zai` | z.ai platform | https://api.z.ai/v1 |
| `openai` | Direct OpenAI API | https://api.openai.com/v1 |
| `anthropic` | Claude models | Native Anthropic API |
| `gemini` | Google Gemini | Native Gemini API |
| `litellm` | Universal (100+ providers) | Varies |

## Recommended Models (January 2026)

### 🏆 Best Value Models (Cheap + Capable)

| Model | Provider | Price (per 1M tokens) | Best For |
|-------|----------|----------------------|----------|
| **GLM 4.7-Flash** | z.ai, OpenRouter | **FREE** / $0.07 | Coding, agents, fast inference |
| **DeepSeek V3.2** | OpenRouter | ~$0.50 | General reasoning, coding |
| **MiniMax M2.1** | OpenRouter | **FREE** / $0.15 | Agentic coding (half the cost of GLM 4.7) |
| **Devstral 2** | OpenRouter | $0.05/$0.22 | Multi-file orchestration |
| **NVIDIA Nemotron 3 Nano** | OpenRouter | $0.06/$0.24 | Agentic AI tasks |
| **Gemini 2.5 Flash-Lite** | Gemini | ~$0.15 | Fast, cost-efficient tasks |
| **GPT-5-mini** | OpenAI | $0.25/$2.00 | Budget OpenAI option |

### 🎯 Premium Models (Quality)

| Model | Provider | Price (per 1M tokens) | Best For |
|-------|----------|----------------------|----------|
| **Claude 3.5 Sonnet** | OpenRouter, Anthropic | $3.00/$15.00 | General tasks, reasoning |
| **GPT-4o** | OpenAI, OpenRouter | $2.50/$10.00 | Balanced performance |
| **Gemini 2.5 Pro** | Gemini | $1.25/$10.00 | Long context (50% cheaper than GPT-4o) |
| **GLM 4.5** | z.ai, OpenRouter | $0.35/$1.55 | Cheap alternative to Sonnet |
| **GLM 4.6** | z.ai | $0.35/$1.50 | Enhanced reasoning |

### 🚀 Frontier Models (Cutting Edge)

| Model | Provider | Price (per 1M tokens) | Best For |
|-------|----------|----------------------|----------|
| **Claude Opus 4.5** | OpenRouter | $5.00/$25.00 | Complex software engineering |
| **GPT-5.2 Pro** | OpenAI | $21.00/$168.00 | 400K context, deep reasoning |
| **GPT-5.1 Codex Max** | OpenAI | $2.00/$8.00 | Agentic coding |
| **Grok 4.1 Fast** | ZenMux (xAI) | Varies | Tool calling, customer support |

### 💡 Provider-Specific Recommendations

**OpenRouter:**
- **Budget**: GLM 4.7-Flash (free), DeepSeek V3.2, MiniMax M2.1
- **Balanced**: Claude 3.5 Sonnet, GPT-4o, GLM 4.5
- **Premium**: Claude Opus 4.5, GPT-5.2 Pro

**z.ai:**
- **FREE**: GLM 4.7-Flash (30B, MIT license) - excellent for coding
- **Cheap**: GLM 4.5 ($0.35/$1.55), GLM 4.6 ($0.35/$1.50)
- **Multimodal**: GLM 4.6V ($0.30/$0.90)

**ZenMux:**
- **FREE**: GLM 4.6V Flash (z-ai)
- **Chinese**: ERNIE-5.0 (Baidu) - strong multimodal
- **Fast**: xAI Grok 4 Fast (2M token context)
- **Agentic**: Grok 4.1 Fast (tool calling)

**OpenAI:**
- **Budget**: GPT-5-mini ($0.25/$2.00)
- **Standard**: GPT-4o ($2.50/$10.00)
- **Coding**: GPT-5.1-Codex-mini
- **Advanced**: GPT-5 ($1.25/$10.00), GPT-5.2 Pro ($21/$168)

**Gemini:**
- **Cheapest**: Gemini 2.5 Flash-Lite
- **Fast**: Gemini 3 Flash (faster than 2.5 Pro)
- **Capable**: Gemini 2.5 Pro ($1.25/$10.00)
- **Latest**: Gemini 3 Pro ($2.00/$12.00)

## How It Works

1. **Context Analysis**: The RLM analyzes the context size and complexity
2. **Code Generation**: Generates Python code to explore the context
3. **Safe Execution**: Executes code in a RestrictedPython sandbox
4. **Recursive Calls**: Breaks large tasks into sub-tasks using `recursive_llm()`
5. **Answer Synthesis**: Combines results and returns via `FINAL()` or `FINAL_VAR()`

## API Reference

### RLM Class

```python
RLM(
    provider: str,           # API provider name
    model: str,              # Model name
    recursive_model: str,    # Optional: different model for recursive calls
    api_key: str,            # Optional: API key (or use env var)
    api_base: str,           # Optional: Custom API base URL
    max_depth: int = 3,      # Maximum recursion depth
    max_iterations: int = 10, # Max iterations per call
    temperature: float = 0.0, # Sampling temperature
)
```

### Methods

- `completion(query, context)` - Synchronous completion
- `acompletion(query, context)` - Asynchronous completion
- `stats` - Property returning execution statistics

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by **Paddle-ERNIE-RAG** - Hybrid search and multimodal RAG implementation
- Inspired by **Qwen3-VL-Embedding** - Vision-language embedding and reranking techniques
- Based on [alexzhang13/rlm](https://github.com/alexzhang13/rlm) - Comprehensive RLM framework
- Based on [ysz/recursive-llm](https://github.com/ysz/recursive-llm) - Minimal recursive LLM implementation
