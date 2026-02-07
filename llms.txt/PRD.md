# PRD — VL-RAG-Graph-RLM

## Project Overview

- **Name:** VL-RAG-Graph-RLM (Vision-Language RAG Graph Recursive Language Model)
- **Version:** 0.1.0
- **Description:** Unified multimodal document analysis framework combining vision-language embeddings, retrieval-augmented generation, knowledge graph extraction, and recursive language model reasoning.
- **Purpose:** Process documents (PPTX, PDF, TXT, MD) with images through a full multimodal pipeline — embed text and images in a unified vector space, retrieve with hybrid search, rerank with cross-attention, build knowledge graphs, and answer queries with recursive reasoning.
- **UX:** CLI (`vrlmrag --provider <name> <path>`), Python API (`MultimodalRAGPipeline`, `VLRAGGraphRLM`), and 17 provider templates.

## Architecture — The Six Pillars

Every template and integration **must** exercise all six pillars:

### 1. VL — Vision-Language Embeddings (Qwen3-VL)
- **Embedder:** `Qwen/Qwen3-VL-Embedding-2B` — unified text + image embedding in a single vector space
- **Reranker:** `Qwen/Qwen3-VL-Reranker-2B` — cross-attention relevance scoring for query-document pairs
- **Multimodal:** Images extracted from PPTX/PDF are embedded alongside text chunks
- **Device:** MPS (Apple Silicon), CUDA, or CPU fallback

### 2. RAG — Retrieval-Augmented Generation
- **Dense Search:** Cosine similarity over Qwen3-VL embeddings
- **Keyword Search:** Token-overlap scoring for lexical matching
- **Hybrid Fusion:** Reciprocal Rank Fusion (RRF) combines dense + keyword with configurable weights
- **Vector Store:** `MultimodalVectorStore` with JSON persistence

### 3. Reranker — Multi-Stage Reranking
- **Stage 1:** RRF fusion (dense 4.0 weight + keyword 1.0 weight)
- **Stage 2:** Qwen3-VL cross-attention reranking (top-15 candidates → top-5)
- **Stage 3:** `MultiFactorReranker` — fuzzy matching, keyword coverage, semantic similarity, length normalization, proper noun bonus
- **Fallback:** `CompositeReranker` when Qwen3-VL is unavailable

### 4. Graph — Knowledge Graph Extraction
- RLM extracts entities, concepts, and relationships from document context
- Graph context feeds into query answering for structured reasoning
- Larger context windows (Nebius 128K, SambaNova 128K) enable richer graph extraction

### 5. RLM — Recursive Language Model
- **Core:** `VLRAGGraphRLM` with configurable `max_depth` (default 3) and `max_iterations` (default 10)
- **REPL:** Safe Python execution environment with `re`, `context`, `query`, and `recursive_llm` available
- **Recursive Calls:** Sub-queries spawn child RLM instances at increased depth with cheaper recursive models
- **Providers:** 17 supported (SambaNova, Nebius, OpenRouter, OpenAI, Anthropic, Gemini, Groq, Cerebras, Generic OpenAI, Generic Anthropic, etc.)

### 6. Pipeline — Unified `MultimodalRAGPipeline`
- Single high-level API combining all pillars
- `pipeline.add_pdf()`, `pipeline.add_pptx()`, `pipeline.add_text()`
- `pipeline.query()` → automatic retrieval, reranking, and recursive reasoning
- Cost tracking and execution metrics

## Supported Providers

| Provider | Default Model | Context | Rate Limits |
|----------|--------------|---------|-------------|
| SambaNova | DeepSeek-V3.2 | 128K | 200K TPD (free) |
| Nebius | MiniMax-M2.1 | 128K | No daily limits |
| OpenRouter | minimax-m2.1 | varies | Per-model |
| OpenAI | gpt-4o-mini | 128K | Per-tier |
| Anthropic | claude-3-5-haiku | 200K | Per-tier |
| Gemini | gemini-1.5-flash | 1M | Per-tier |
| Groq | llama-3.1-70b | 128K | Per-tier |
| DeepSeek | deepseek-chat | 128K | Per-tier |
| ZenMux | ernie-5.0-thinking | varies | Per-model |
| z.ai | glm-4.7 | 128K | Per-tier |
| Mistral | mistral-large | 128K | Per-tier |
| Fireworks | llama-3.1-70b | 128K | Per-tier |
| Together | llama-3.1-70b-turbo | 128K | Per-tier |
| Azure OpenAI | gpt-4o | 128K | Per-deployment |
| Cerebras | llama-4-scout-17b-16e-instruct | 128K | Per-tier | Ultra-fast wafer-scale |
| Generic OpenAI | (user-configured) | varies | Per-provider |
| Generic Anthropic | (user-configured) | varies | Per-provider |

## CLI

```bash
# Unified provider flag (17 providers supported)
vrlmrag --provider sambanova document.pptx
vrlmrag --provider nebius document.pdf --output report.md
vrlmrag --provider openrouter ./folder --query "Summarize key findings"
vrlmrag --provider gemini paper.pdf --model gemini-1.5-pro

# Utility commands
vrlmrag --list-providers          # Show all providers + API key status
vrlmrag --version                 # Print version
vrlmrag --help                    # Full usage

# RLM tuning
vrlmrag --provider nebius doc.pptx --max-depth 5 --max-iterations 25

# Backward-compatible aliases
vrlmrag --samba-nova document.pptx
vrlmrag --nebius document.pdf
```

## Short-term Goals

> READ `llms.txt/TODO.md`

## Codebase Requirements

> READ `llms.txt/RULES.md`

## Architecture Details

> READ `llms.txt/ARCHITECTURE.md`
