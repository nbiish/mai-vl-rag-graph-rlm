# VL-RAG-Graph-RLM Documentation

Version: 0.1.0

## Overview

This folder contains comprehensive documentation for the VL-RAG-Graph-RLM (Vision-Language RAG Graph Recursive Language Model) framework — a unified multimodal document analysis system.

## Documentation Files

| File | Purpose |
|------|---------|
| **[PRD.md](PRD.md)** | Product Requirements Document — project overview, six-pillar architecture, provider list, CLI usage |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture, component map, template patterns, CLI reference, environment variables |
| **[RULES.md](RULES.md)** | Coding standards, always/never patterns, device detection guidelines |
| **[TODO.md](TODO.md)** | Roadmap, planned features, completed items (v0.1.0) |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Guide for adding providers, six-pillar requirement, testing guidelines |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history, notable changes, release notes |

## Quick Navigation

### For Users
- Start with **[PRD.md](PRD.md)** to understand what the system does
- See **[ARCHITECTURE.md](ARCHITECTURE.md)** for CLI usage and environment setup
- Check **[CHANGELOG.md](CHANGELOG.md)** for the latest features in v0.1.0

### For Contributors
- Read **[CONTRIBUTING.md](CONTRIBUTING.md)** for how to add a new provider
- Follow **[RULES.md](RULES.md)** for coding standards and patterns
- Track progress in **[TODO.md](TODO.md)**

### For Developers
- Study **[ARCHITECTURE.md](ARCHITECTURE.md)** for the complete component map
- Reference **[RULES.md](RULES.md)** for device detection and Qwen3-VL patterns
- See **[PRD.md](PRD.md)** for the six-pillar architecture requirements

## The Six-Pillar Architecture

All documentation refers to the six pillars that form the foundation of this framework:

1. **VL** — Vision-Language Embeddings (Qwen3-VL for text + images)
2. **RAG** — Retrieval-Augmented Generation (hybrid search with RRF)
3. **Reranker** — Multi-stage reranking (Qwen3-VL cross-attention)
4. **Graph** — Knowledge Graph Extraction (via RLM)
5. **RLM** — Recursive Language Model (with REPL)
6. **Pipeline** — Unified API (MultimodalRAGPipeline)

See **[PRD.md](PRD.md)** or **[ARCHITECTURE.md](ARCHITECTURE.md)** for details.

## Version

Current release: **v0.1.0** (2025-02-07)

See **[CHANGELOG.md](CHANGELOG.md)** for full release notes.

## Project Links

- Main README: `../README.md`
- Templates: `../templates/`
- Source Code: `../src/vl_rag_graph_rlm/`
