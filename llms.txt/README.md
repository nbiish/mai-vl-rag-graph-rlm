# VL-RAG-Graph-RLM Documentation

Version: 0.1.1 (Feb 8, 2026)

## Overview

This folder contains comprehensive documentation for the **VL-RAG-Graph-RLM** (Vision-Language RAG Graph Recursive Language Model) framework — a unified multimodal document analysis system with **named persistent knowledge collections**, **accuracy-first retrieval**, and **17 LLM provider templates** with automatic fallback.

The system processes documents (PPTX, PDF, TXT, MD) with images through a full 6-pillar pipeline: Qwen3-VL vision-language embeddings → hybrid RAG with RRF fusion → cross-attention reranking → knowledge graph extraction → recursive LLM reasoning → markdown report generation.

## What's New (v0.1.1 — Feb 8, 2026)

### Named Persistent Collections
Build named, location-independent knowledge stores that persist inside the codebase. Add documents from any path, query from anywhere, blend multiple collections, and script everything via CLI — no interaction required.

```bash
vrlmrag -c research --add ./papers/          # add docs to a collection
vrlmrag -c research -q "Key findings?"       # query a collection
vrlmrag -c research -c code -q "How?"        # blend multiple collections
vrlmrag -c research -i                       # interactive w/ collection
vrlmrag --collection-list                    # list all collections
```

### Accuracy-First Query Pipeline
All queries route through a single shared function (`_run_vl_rag_query()`) with widened retrieval parameters: `top_k=50` dense/keyword, `30` reranker candidates, `10` final results. Embedding instruction pairing (`_DOCUMENT_INSTRUCTION` / `_QUERY_INSTRUCTION`) and structured KG extraction with typed entities and explicit relationships.

### Universal Persistent Embeddings
Content-based SHA-256 deduplication across all modes. Re-running on the same folder only embeds new/changed files. Knowledge graph merges across runs and is prepended to every query. Provider-agnostic — same store works with any LLM provider.

## Documentation Files

| File | Purpose |
|------|---------|
| **[README.md](README.md)** | Documentation index — quick navigation, what's new, key capabilities |
| **[PRD.md](PRD.md)** | Product Requirements — six-pillar architecture, 17 providers, CLI, collections, future plans |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture — diagrams, component map, pipeline flow, collection internals, CLI reference |
| **[RULES.md](RULES.md)** | Coding standards — always/never patterns, collection rules, device detection, provider-specific rules |
| **[TODO.md](TODO.md)** | Roadmap — v0.2.0 plans, collection enhancements, completed items |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Contributor guide — adding providers, extending collections, testing |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history — v0.1.0 initial release, v0.1.1 collections + accuracy pipeline |
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
| 1 | **VL** | Qwen3-VL-Embedding-2B — unified text + image embeddings | FREE (local) |
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

17 providers with automatic multi-tier fallback:
1. **Model fallback** — primary model fails → retry with fallback model (same provider)
2. **Provider fallback** — both models fail → try next provider in hierarchy
3. **z.ai three-tier** — Coding Plan endpoint → Normal endpoint → model fallback → hierarchy

Default hierarchy: `sambanova → nebius → groq → cerebras → zai → zenmux → openrouter → ...`

## Future Plans

See **[TODO.md](TODO.md)** for the full roadmap. Key upcoming features:

- **Collection enhancements:** import/export, tagging, collection-to-collection merging, remote sync
- **RAG improvements:** BM25 keyword search, SQLite vector store, multi-query retrieval
- **Knowledge graph:** NetworkX serialization, Mermaid visualization, entity deduplication
- **CLI/UX:** `--format json`, streaming output, progress bars, `--verbose`/`--quiet`
- **Testing:** Unit tests, integration tests, CI pipeline, benchmark suite
- **Providers:** Ollama (local), vLLM (self-hosted), token tracking, rate limiting

## Version

Current release: **v0.1.1** (2026-02-08)

See **[CHANGELOG.md](CHANGELOG.md)** for full release notes.

## Project Links

- Main README: `../README.md`
- Collections module: `../src/vl_rag_graph_rlm/collections.py`
- CLI entry point: `../src/vrlmrag.py`
- Templates: `../templates/`
- Source Code: `../src/vl_rag_graph_rlm/`
