# VL-RAG-Graph-RLM Documentation

Version: 0.2.0 (Feb 12, 2026)

## Overview

This folder contains comprehensive documentation for the **VL-RAG-Graph-RLM** (Vision-Language RAG Graph Recursive Language Model) framework — a unified multimodal document analysis system with **named persistent knowledge collections**, **accuracy-first retrieval**, and **18 LLM provider templates** with automatic fallback.

The system processes documents (PPTX, PDF, TXT, MD, Video, Audio) through a full 6-pillar pipeline: Qwen3-VL vision-language embeddings → hybrid RAG with RRF fusion → cross-attention reranking → knowledge graph extraction → recursive LLM reasoning → markdown report generation. All model loading uses **sequential load-use-free** memory management (peak ~6.7 GB on a 40-min video).

## What's New (v0.2.0 — Feb 12, 2026)

### 🧪 Modal Research Provider (New!)
**Free GLM-5 745B frontier inference** via Modal Research's OpenAI-compatible endpoint:
- **Model:** `zai-org/GLM-5-FP8` — 745B parameters (44B active), MoE architecture, MIT license
- **Endpoint:** `https://api.us-west-2.modal.direct/v1` — runs on 8×B200 GPUs via SGLang
- **Performance:** 30-75 tok/s per user, frontier-class reasoning
- **Status:** Experimental (free tier: 1 concurrent request, may have downtime)
- **Get key:** https://modal.com/glm-5-endpoint

```bash
vrlmrag document.pptx --provider modalresearch
# Or use auto mode — modalresearch is first in hierarchy
vrlmrag document.pptx
```

### 🔑 Fallback API Key System (New!)
**Multi-account support** with automatic fallback when primary keys fail:
- **Pattern:** `{PROVIDER}_API_KEY_FALLBACK` — every provider supports this suffix
- **Use cases:** Credit distribution, rate limit mitigation, account redundancy
- **Four-tier resilience:** Primary key → Fallback key → Model fallback → Provider hierarchy

```bash
# Example: Two OpenRouter accounts
OPENROUTER_API_KEY=sk-or-v1-primary-key
OPENROUTER_API_KEY_FALLBACK=sk-or-v1-secondary-key
```

**Implementation:**
- All OpenAI-compatible providers (14+ providers via `OpenAICompatibleClient`)
- Anthropic/AnthropicCompatible clients
- Gemini client
- Fallback key promoted to primary after successful retry (session persistence)

### 🎯 Omni Model Fallback Chain (New!)
Three-tier resilient multimodal processing for images, audio, and video:
- **Primary:** ZenMux `inclusionai/ming-flash-omni-preview` — text, image, audio, video
- **Secondary:** ZenMux `gemini/gemini-3-flash-preview` — fallback for all modalities
- **Tertiary:** OpenRouter `google/gemini-3-flash-preview` — final omni fallback
- **Legacy VLM:** OpenRouter `moonshotai/kimi-k2.5` — images/video only (no audio)

Audio transcription now routes through the full omni chain — no more silent failures when the primary omni model is unavailable.

### 📦 Collection Management (New!)
- **Export/Import** — `--collection-export PATH` and `--collection-import PATH` for portable tar.gz archives
- **Collection Merge** — `--collection-merge SRC` merges one collection into another
- **Collection Tagging** — `--collection-tag TAG` and `--collection-untag TAG` for organization
- **Collection Search** — `--collection-search QUERY` and `--collection-search-tags TAGS` to find collections
- **Statistics Dashboard** — `--collection-stats` and `--global-stats` for detailed analytics

### 🔍 RAG Improvements
- **BM25 keyword search** — Replaced simple token-overlap with state-of-the-art BM25 algorithm via `rank-bm25`
- **Graph-augmented retrieval** — `--graph-augmented` traverses KG edges for context expansion (`--graph-hops N`)
- **Multi-query retrieval** — `--multi-query` generates sub-queries via RLM for broader recall
- **Configurable RRF weights** — `--rrf-dense-weight` and `--rrf-keyword-weight` tune fusion balance
- **SQLite backend** — `--use-sqlite` flag enables persistent vector store with better performance

### 📊 Output & UX Enhancements
- **JSON output** — `--format json` for machine-readable results (default: markdown)
- **Log level control** — `--verbose` and `--quiet` for output verbosity
- **Progress bars** — tqdm integration for embedding/search operations

### 🎯 Smart Defaults (New!)
- **Configuration profiles** — `--profile {fast,balanced,thorough,comprehensive}` presets
- **Comprehensive by default** — All best features enabled automatically (multi-query, graph-augmented, deep reasoning)
- **API hierarchy default** — Provider auto-fallback enabled by default (set keys in .env)
- **MCP streamlined server** — 4 consolidated tools instead of 11+ for reduced context usage

### 🤖 New Providers
- **Ollama** — Local LLM inference support (`--provider ollama`)

### 📄 Enhanced Document Processing
- **PDF support** — PyMuPDF extracts text and images from PDF documents
- **DOCX support** — python-docx extracts text and tables from Word documents
- **CSV/Excel support** — Tabular data ingestion with natural language row chunking
- **Sliding window chunking** — Configurable `--chunk-size` and `--chunk-overlap`

### Knowledge Graph Enhancements
- **Graph visualization** — `--export-graph PATH` exports to Mermaid, Graphviz (DOT), or NetworkX formats
- **Graph statistics** — `--graph-stats` shows entity counts, relationship stats, type distribution
- **Entity deduplication** — `--deduplicate-kg` merges similar entities with configurable `--dedup-threshold`
- **NetworkX serialization** — Export structured graphs for external analysis

### Model Management
- **Model comparison** — `--model-compare OLD_MODEL` compares embeddings between model versions
- **Compatibility checking** — `--check-model MODEL` verifies collection compatibility before migration
- **Quality assessment** — `--quality-check` RLM-powered evaluation of embedding retrieval quality

```bash
# Process PDF with sliding window chunking
vrlmrag document.pdf --chunk-size 500 --chunk-overlap 50

# Export knowledge graph as Mermaid diagram
vrlmrag -c research --export-graph graph.mmd --graph-format mermaid

# Show graph statistics and deduplication report
vrlmrag -c research --graph-stats --dedup-report

# Run with multi-query retrieval for better recall
vrlmrag ./docs -q "Key findings?" --multi-query
```

### Named Persistent Collections (v0.1.1)
Build named, location-independent knowledge stores that persist inside the codebase.

```bash
vrlmrag -c research --add ./papers/          # add docs to a collection
vrlmrag -c research -q "Key findings?"       # query a collection
vrlmrag -c research -c code -q "How?"        # blend multiple collections
```

## Documentation Files

| File | Purpose |
|------|---------|
| **[README.md](README.md)** | Documentation index — quick navigation, what's new, key capabilities |
| **[PRD.md](PRD.md)** | Product Requirements — six-pillar architecture, 17 providers, CLI, collections, future plans |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture — diagrams, component map, pipeline flow, collection internals, CLI reference |
| **[RULES.md](RULES.md)** | Coding standards — always/never patterns, collection rules, device detection, provider-specific rules |
| **[TODO.md](TODO.md)** | Roadmap — v0.2.0 plans, collection enhancements, completed items |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Contributor guide — adding providers, extending collections, testing |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history — v0.1.0 initial, v0.1.1 collections, v0.1.2 audio/video/memory |
| **[SECURITY.md](../SECURITY.md)** | Local security orchestration — secret scanning, sanitization, OWASP compliance |

## Quick Navigation

### For Users
- **Getting started:** [PRD.md](PRD.md) → system overview, CLI examples, provider list
- **CLI reference:** [ARCHITECTURE.md](ARCHITECTURE.md) → all flags, collection commands, environment variables
- **Collections:** [ARCHITECTURE.md § Named Persistent Collections](ARCHITECTURE.md) → storage layout, blending, scripting
- **What's new:** [CHANGELOG.md](CHANGELOG.md) → v0.1.1 features

### For Contributors
- **Adding providers:** [CONTRIBUTING.md](CONTRIBUTING.md) → 5-step guide with template
- **Extending collections:** [CONTRIBUTING.md](CONTRIBUTING.md) → collection manager API
- **Coding standards:** [RULES.md](RULES.md) → always/never patterns, collection rules
- **Roadmap:** [TODO.md](TODO.md) → what's planned, what's done

### For Developers
- **Architecture deep-dive:** [ARCHITECTURE.md](ARCHITECTURE.md) → system diagram, data flow, component map
- **Collection internals:** [ARCHITECTURE.md](ARCHITECTURE.md) → `collections.py` API, metadata schema, blending mechanics
- **Pipeline flow:** [ARCHITECTURE.md](ARCHITECTURE.md) → `_run_vl_rag_query()` template pattern with retrieval instructions
- **Provider rules:** [RULES.md](RULES.md) → device detection, Qwen3-VL patterns, fallback behavior

## The Six-Pillar Architecture

Every template, query, and collection operation exercises all six pillars:

| # | Pillar | Component | Cost |
|---|--------|-----------|------|
| 1 | **VL** | Qwen3-VL-Embedding-2B — unified text + image + video + audio embeddings | FREE (local) |
| 2 | **RAG** | Hybrid search (dense cosine + keyword) with RRF fusion | FREE (local) |
| 3 | **Reranker** | Qwen3-VL-Reranker-2B — cross-attention relevance scoring | FREE (local) |
| 4 | **Graph** | Knowledge graph extraction via RLM (typed entities + relationships) | LLM cost |
| 5 | **RLM** | Recursive Language Model with sandboxed REPL | LLM cost |
| 6 | **Report** | Markdown report with sources, scores, and metadata | FREE |

See **[PRD.md](PRD.md)** for the full architecture specification or **[ARCHITECTURE.md](ARCHITECTURE.md)** for implementation details.

## Key Capabilities

### Three Operating Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Default** | `vrlmrag <path>` | Process docs → embed → query → report |
| **Interactive** | `vrlmrag -i <path>` | Load VL models once, query continuously, `/add` docs on the fly |
| **Collection** | `vrlmrag -c <name> -q "..."` | Query named persistent knowledge stores, blend multiple collections |

### Persistence & Deduplication

All modes persist embeddings and knowledge graphs automatically:
- **Path-local store:** `.vrlmrag_store/` next to input (default and interactive modes)
- **Named collections:** `collections/<name>/` inside the codebase (collection mode)
- **SHA-256 dedup:** Only new/changed content gets re-embedded
- **KG merging:** Knowledge graph grows across runs, never overwrites
- **Provider-agnostic:** Embeddings are local Qwen3-VL; any LLM provider can query any store

### Provider Resilience

18 providers with automatic **four-tier fallback**:
1. **API key fallback** — primary key fails → retry with `{PROVIDER}_API_KEY_FALLBACK` (same provider, different account)
2. **Model fallback** — primary model fails → retry with fallback model (same provider, same key)
3. **Provider fallback** — all retries fail → try next provider in hierarchy
4. **z.ai five-tier** — Coding Plan endpoint → Normal endpoint → fallback key → model fallback → hierarchy

Default hierarchy: `modalresearch → sambanova → nebius → groq → cerebras → zai → zenmux → openrouter → ...`

## Future Plans

See **[TODO.md](TODO.md)** for the full roadmap. Key upcoming features:

- **Collection enhancements:** Remote sync, multi-user access
- **Testing:** Integration tests, CI pipeline, benchmark suite
- **Providers:** vLLM (self-hosted), more local LLM options
- **Advanced RAG:** Hybrid fusion, query expansion, cross-collection search

## Version

Current release: **v0.2.0** (2026-02-12)

See **[CHANGELOG.md](CHANGELOG.md)** for full release notes.

## Project Links

- Main README: `../README.md`
- Collections module: `../src/vl_rag_graph_rlm/collections.py`
- CLI entry point: `../src/vrlmrag.py`
- Templates: `../templates/`
- Source Code: `../src/vl_rag_graph_rlm/`
