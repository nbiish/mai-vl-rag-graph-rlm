# PRD — VL-RAG-Graph-RLM

## Project Overview

- **Name:** VL-RAG-Graph-RLM (Vision-Language RAG Graph Recursive Language Model)
- **Version:** 0.1.1
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

| Provider | Default Model | Also Available | Context | Rate Limits |
|----------|--------------|----------------|---------|-------------|
| SambaNova | DeepSeek-V3.2 | DeepSeek-V3.1, gpt-oss-120b, Qwen3-235B, Llama-4-Maverick | 128K | 200K TPD (free) |
| Nebius | MiniMaxAI/MiniMax-M2.1 | zai-org/GLM-4.7-FP8, DeepSeek-R1, Nemotron-Ultra-253B | 128K | No daily limits |
| OpenRouter | minimax/minimax-m2.1 | 400+ models incl. GPT-5.3, Claude Opus 4.6, Gemini 3 | varies | Per-model |
| OpenAI | gpt-4o-mini | gpt-4o, gpt-5.2, gpt-5.3-codex | 128K | Per-tier |
| Anthropic | claude-3-5-haiku | claude-sonnet-4, claude-opus-4.6 | 200K | Per-tier |
| Gemini | gemini-1.5-flash | gemini-3-pro, gemini-3-flash | 1M | Per-tier |
| Groq | moonshotai/kimi-k2-instruct-0905 | openai/gpt-oss-120b, llama-4-maverick, qwen3-32b | 128K | Per-tier |
| DeepSeek | deepseek-chat | deepseek-reasoner (R1) | 128K | Per-tier |
| ZenMux | moonshotai/kimi-k2.5 | 59+ models, provider/model format | varies | Per-model |
| z.ai | glm-4.7 | glm-4.7-flash, glm-4.5-air (Coding Plan) | 128K | Coding Plan / Per-tier |
| Mistral | mistral-large-latest | mistral-large-3 (675B MoE) | 128K | Per-tier |
| Fireworks | llama-v3p1-70b-instruct | various open-source models | 128K | Per-tier |
| Together | Meta-Llama-3.1-70B-Instruct-Turbo | various open-source models | 128K | Per-tier |
| Azure OpenAI | gpt-4o | enterprise GPT deployments | 128K | Per-deployment |
| Cerebras | zai-glm-4.7 | gpt-oss-120b (~3000 tok/s), qwen-3-235b | 128K | Per-tier |
| Generic OpenAI | (user-configured) | any OpenAI-compatible endpoint | varies | Per-provider |
| Generic Anthropic | (user-configured) | any Anthropic-compatible endpoint | varies | Per-provider |

> **Model data queried from live APIs on Feb 7, 2026.** Cerebras deprecating llama-3.3-70b and qwen-3-32b on Feb 16, 2026.

## Universal Model Fallback

Every provider has **automatic model fallback** built into the base client class:

1. **Primary model** is attempted first (e.g., `DeepSeek-V3.2` on SambaNova)
2. On **any error** (rate limit, token limit, downtime, network), the client automatically retries with a **fallback model** on the same provider
3. If the fallback also fails, the error propagates to the **provider hierarchy** for cross-provider fallback

**Fallback map (overridable via `{PROVIDER}_FALLBACK_MODEL`):**
- `sambanova`: DeepSeek-V3.2 → DeepSeek-V3.1
- `groq`: kimi-k2-instruct → llama-3.3-70b-versatile  
- `cerebras`: zai-glm-4.7 → gpt-oss-120b
- `nebius`: MiniMax-M2.1 → GLM-4.7-FP8
- `openrouter`: minimax-m2.1 → deepseek-v3.2
- `zenmux`: kimi-k2.5 → glm-4.7
- `zai`: glm-4.7 → glm-4.5-air (plus endpoint fallback: Coding Plan → Normal)
- `openai`: gpt-4o-mini → gpt-4o
- `mistral`: mistral-large → mistral-small
- `deepseek`: deepseek-chat → deepseek-reasoner

**Result:** Up to `2 × N` chances per API call (2 models × N providers with keys).

## CLI

```bash
# Auto mode — no --provider needed, uses hierarchy fallback
vrlmrag document.pptx
vrlmrag ./folder --query "Summarize key findings"

# Explicit provider
vrlmrag --provider sambanova document.pptx
vrlmrag --provider nebius document.pdf --output report.md
vrlmrag --provider gemini paper.pdf --model gemini-3-pro

# Hierarchy management
vrlmrag --show-hierarchy          # Show fallback order + availability
vrlmrag --list-providers          # Show all providers + API key status

# RLM tuning
vrlmrag --provider nebius doc.pptx --max-depth 5 --max-iterations 25

# Interactive mode — load VL models once, query continuously
vrlmrag --interactive presentation.pptx
vrlmrag -i ./codebase
vrlmrag -i                            # Start empty, /add docs later

# Collections — named persistent knowledge stores
vrlmrag -c research --add ./papers/              # add docs to collection
vrlmrag -c research -q "Key findings?"           # query a collection
vrlmrag -c research -c code -q "How?"            # blend collections
vrlmrag -c research -i                           # interactive w/ collection
vrlmrag --collection-list                        # list all collections
vrlmrag -c research --collection-info            # show collection details
vrlmrag -c research --collection-delete          # delete a collection
```

- **UX:** `--provider` defaults to `auto`. The hierarchy is editable via `PROVIDER_HIERARCHY` in `.env`.
- **Fallback:** If a provider fails (rate limit, auth, network), the system falls through to the next available provider automatically.
- **SDK priority:** If `OPENAI_COMPATIBLE_API_KEY` or `ANTHROPIC_COMPATIBLE_API_KEY` is set, those custom endpoints are automatically prepended as the highest-priority providers (user explicitly configured a custom SDK endpoint).
- **Interactive mode:** `--interactive` / `-i` loads VL models once, then provides a REPL for continuous querying. Supports `/add <path>` for incremental document addition, `/kg` to inspect the knowledge graph, and `/save` to export reports. Knowledge graph and embeddings persist to disk across sessions.
- **Persistent embeddings:** All modes persist embeddings and knowledge graph to `.vrlmrag_store/`. Re-runs skip already-embedded content (SHA-256 dedup). The KG merges across runs and is prepended to every query. Works with any provider/model combo.
- **Named collections:** `-c <name>` creates persistent knowledge stores at `collections/<name>/` inside the codebase. Add docs with `--add`, query with `-q`, blend with `-c A -c B`. Fully scriptable — no interaction required. Collections can also back interactive sessions (`-c <name> -i`).

## Named Persistent Collections

Collections are the system's primary mechanism for building reusable, scriptable knowledge bases. They solve the problem of needing to re-embed and re-process documents every time you want to query them.

### Design Goals

1. **Location-independent:** Collections live inside the codebase at `collections/<name>/`. Query from any directory.
2. **Scriptable:** Every operation is a single CLI command. No interaction required. Pipe-friendly output.
3. **Blendable:** Merge multiple collections' embeddings and knowledge graphs for cross-domain queries.
4. **Incremental:** Only new content gets embedded (SHA-256 dedup). KG merges, never overwrites.
5. **Provider-agnostic:** Embeddings are local Qwen3-VL. Any LLM provider can query any collection.

### Use Cases

| Scenario | Commands |
|----------|----------|
| **Research library** | `-c papers --add ./papers/` then `-c papers -q "Summarize findings"` |
| **Codebase documentation** | `-c code --add ./src/` then `-c code -q "How does auth work?"` |
| **Cross-domain analysis** | `-c papers -c code -q "How does the code implement the paper?"` |
| **Meeting knowledge base** | `-c meetings --add ./notes/` after each meeting, query anytime |
| **CI/CD integration** | Script: `vrlmrag -c docs --add ./docs/ && vrlmrag -c docs -q "..." -o report.md` |
| **Interactive exploration** | `-c research -i` → load collection, query continuously, `/add` more docs |

### Storage Layout

```
<project_root>/collections/
├── research/
│   ├── collection.json       # metadata (name, description, sources, counts)
│   ├── embeddings.json       # Qwen3-VL embeddings (text + images)
│   └── knowledge_graph.md    # accumulated KG across all additions
├── codebase/
│   ├── collection.json
│   ├── embeddings.json
│   └── knowledge_graph.md
└── .gitignore                # collection data is local-only
```

### How It Works

1. **`--add`** processes documents through `DocumentProcessor`, embeds with Qwen3-VL (dedup via SHA-256), extracts KG via RLM, and persists everything to the collection directory.
2. **`-q`** loads the collection's embeddings and KG, runs the full 6-pillar pipeline (`_run_vl_rag_query()`), and prints/saves the result.
3. **Blending** (`-c A -c B`) merges vector stores (doc-ID dedup) and knowledge graphs (`---` separator) before running the pipeline.
4. **`-i`** starts an interactive session using the collection's directory as the `--store-dir`.

## Scripting & Automation

The CLI is designed for scripting. Every command exits cleanly with appropriate exit codes and can be piped or redirected.

```bash
#!/bin/bash
# Build a knowledge base from multiple sources
vrlmrag -c project --add ./docs/ --collection-description "Project documentation"
vrlmrag -c project --add ./specs/requirements.pdf
vrlmrag -c project --add ./meeting-notes/

# Generate a report
vrlmrag -c project -q "Summarize the project status and key decisions" -o status.md

# Cross-reference with code
vrlmrag -c code --add ./src/
vrlmrag -c project -c code -q "Are there gaps between specs and implementation?" -o gaps.md

# List what we have
vrlmrag --collection-list
```

## Accuracy-First Query Pipeline

All queries — default, interactive, and collection — route through `_run_vl_rag_query()`:

| Stage | Component | Parameters |
|-------|-----------|------------|
| 1. Dense search | Qwen3-VL embedding (cosine sim) | `top_k=50`, `_QUERY_INSTRUCTION` |
| 2. Keyword search | Token-overlap scoring | `top_k=50` |
| 3. Fusion | Reciprocal Rank Fusion | `k=60`, weights `[4.0, 1.0]` |
| 4. Reranking | Qwen3-VL cross-attention | `30` candidates → `10` final |
| 5. KG augmentation | Persisted knowledge graph | Up to `8000` chars |
| 6. RLM completion | Recursive Language Model | `max_depth=3`, `max_iterations=10` |

Embedding instruction pairing: `_DOCUMENT_INSTRUCTION` for ingestion, `_QUERY_INSTRUCTION` for search.

## Short-term Goals

> READ `llms.txt/TODO.md`

## Codebase Requirements

> READ `llms.txt/RULES.md`

## Architecture Details

> READ `llms.txt/ARCHITECTURE.md`
