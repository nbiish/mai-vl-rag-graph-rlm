# Changelog — VL-RAG-Graph-RLM

> **Documentation Index:** See [README.md](README.md) for all documentation files.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-02-08

### Named Persistent Collections
- **`collections.py` module** — CRUD for named collections: `create_collection`, `list_collections`, `delete_collection`, `load_collection_meta`, `save_collection_meta`, `record_source`, `load_kg`, `save_kg`, `merge_kg`
- **Collection storage layout** — `collections/<name>/` with `collection.json` (metadata), `embeddings.json` (Qwen3-VL embeddings), `knowledge_graph.md` (accumulated KG)
- **`-c <name> --add <path>`** — Add documents to a named collection (embed + KG extract + persist)
- **`-c <name> -q "..."`** — Query a collection via full VL-RAG pipeline (scriptable, non-interactive)
- **`-c A -c B -q "..."`** — Blend multiple collections: merge vector stores and knowledge graphs for cross-collection queries
- **`-c <name> -i`** — Interactive session backed by a collection's store directory
- **`--collection-list`** — List all collections with doc/chunk counts and last-updated timestamps
- **`--collection-info`** — Detailed info for a collection (sources, embedding count, KG size)
- **`--collection-delete`** — Delete a collection and all its data
- **`--collection-description`** — Set description when creating a collection via `--add`
- **`collections/.gitignore`** — Collection data excluded from version control

### Accuracy-First Query Pipeline
- **Unified `_run_vl_rag_query()`** — Single source of truth for all query paths (run_analysis, interactive, collections)
- **Retrieval instruction pairing** — `_DOCUMENT_INSTRUCTION` for ingestion, `_QUERY_INSTRUCTION` for search (Qwen3-VL recommended asymmetric pairing)
- **Wider retrieval depth** — `top_k=50` dense/keyword, `30` reranker candidates, `10` final results (accuracy over speed)
- **Structured KG extraction prompt** — Typed entities (Person, Organisation, Concept, Technology) + explicit relationships (`EntityA → rel → EntityB`)
- **KG budget increased** — Up to 8000 chars (⅓ of context budget) prepended to every query
- **Eliminated duplicated query logic** — Both `run_analysis()` and interactive mode delegate to shared function

### Universal Persistent Embeddings & Interactive Mode
- **Content-based deduplication (SHA-256)** — `MultimodalVectorStore` skips re-embedding already-stored content
- **Universal KG persistence** — Knowledge graph saved/merged in both `run_analysis()` and interactive mode
- **KG-augmented queries in all modes** — Knowledge graph context prepended to every query
- **Incremental embedding** — Re-running on same folder only embeds new/changed files
- **Provider-agnostic store** — Same `.vrlmrag_store/` used regardless of provider/model combo
- **`--interactive` / `-i` CLI flag** — Persistent session with VL models loaded once
- **REPL commands** — `/add <path>`, `/kg`, `/stats`, `/save`, `/help`, `/quit`
- **`--store-dir` flag** — Custom persistence directory

### Universal Model Fallback
- **`FALLBACK_MODELS` dict** — Hardcoded fallback models for 11+ providers in base class
- **`{PROVIDER}_FALLBACK_MODEL` env var** — Override fallback per-provider
- **Two-tier resilience** — Model fallback (same provider) → Provider hierarchy fallback (next provider)
- **z.ai three-tier** — Coding Plan endpoint → Normal endpoint → Model fallback → Provider hierarchy

### Provider Model Updates (Feb 7-8, 2026)
- **Groq** → `moonshotai/kimi-k2-instruct-0905` (Kimi K2 on Groq LPU)
- **Cerebras** → `zai-glm-4.7` (GLM 4.7 355B, ~1000 tok/s)
- **SambaNova** → DeepSeek-V3.2 default, also V3.1, gpt-oss-120b, Qwen3-235B
- **Nebius** → MiniMax-M2.1 default, also GLM-4.7-FP8, Nemotron-Ultra-253B

### Security — Local Orchestration (Feb 8, 2026)
- **ainish-coder deployment** — Local security scripts and git hooks
- **`.ainish/scripts/sanitize.py`** — Pre-commit secret sanitizer (50+ patterns, auto-replaces secrets with placeholders)
- **`.ainish/scripts/scan_secrets.sh`** — Pre-push secret scanner (generates SECURITY_REPORT.md)
- **Pre-commit hook** — Auto-sanitizes staged files before every commit (OWASP ASI04 compliance)
- **Pre-push hook** — Scans entire repo and blocks push if secrets detected
- **`SECURITY.md`** — Comprehensive security documentation (secret patterns, best practices, compliance)
- **Secret patterns covered:** 50+ including AI/LLM providers, cloud credentials, database connections, private keys, local paths
- **OWASP Agentic Security 2026 compliance:** ASI04 (Information Disclosure), ASI06 (Legacy Crypto warnings)
- **FIPS 204 awareness:** RSA key detection with ML-DSA migration warnings

### Documentation
- Comprehensive `llms.txt/` update: README, ARCHITECTURE, PRD, RULES, TODO, CHANGELOG, CONTRIBUTING
- Collection system fully documented: storage layout, data flow, blending mechanics, metadata schema, API reference
- Accuracy-first pipeline documented: retrieval parameters, instruction pairing, KG extraction prompt
- Future roadmap: collection enhancements (import/export, tagging, remote sync), RAG improvements, testing

## [0.1.0] - 2025-02-07

### Added
- Initial release of VL-RAG-Graph-RLM (Vision-Language RAG Graph Recursive Language Model)
- Six-pillar multimodal RAG architecture (VL, RAG, Reranker, Graph, RLM, Pipeline)
- Unified `MultimodalRAGPipeline` API for document processing and querying
- CLI with unified `--provider` flag supporting 17 providers
- Support for PPTX, PDF, TXT, and MD document formats
- Automatic image extraction and multimodal embedding

### Vision-Language Embeddings
- Qwen3-VL embedder (`Qwen/Qwen3-VL-Embedding-2B`) for unified text + image embeddings
- Qwen3-VL reranker (`Qwen/Qwen3-VL-Reranker-2B`) for cross-attention relevance scoring
- Device auto-detection: MPS (Apple Silicon), CUDA, or CPU fallback
- Multimodal vector store with JSON persistence

### Retrieval-Augmented Generation
- Dense search via cosine similarity over Qwen3-VL embeddings
- Keyword search via token-overlap scoring for lexical matching
- Hybrid fusion using Reciprocal Rank Fusion (RRF)
- Configurable weights (default: dense=4.0, keyword=1.0)

### Multi-Stage Reranking
- Stage 1: RRF fusion combining dense + keyword results
- Stage 2: Qwen3-VL cross-attention reranking (top-15 → top-5)
- Stage 3: MultiFactorReranker with fuzzy matching, keyword coverage, semantic similarity, length normalization, and proper noun bonus
- Fallback CompositeReranker when Qwen3-VL is unavailable

### Knowledge Graph Extraction
- RLM-based entity, concept, and relationship extraction
- Graph context augmentation for query answering
- Support for larger context windows (Nebius 128K, SambaNova 128K)

### Recursive Language Model
- `VLRAGGraphRLM` class with configurable `max_depth` (default 3) and `max_iterations` (default 10)
- Safe Python execution REPL with RestrictedPython sandbox
- Recursive sub-query spawning with cheaper recursive models
- Environment provides: `context`, `query`, `re`, `recursive_llm`

### Provider Support (17 total)
| Provider | Default Model | Notes |
|----------|--------------|-------|
| SambaNova | DeepSeek-V3.2 | 128K context, 200K TPD free tier |
| Nebius | MiniMax-M2.1 | 128K context, no daily limits |
| OpenRouter | minimax-m2.1 | Per-model rates |
| OpenAI | gpt-4o-mini | 128K context |
| Anthropic | claude-3-5-haiku | 200K context |
| Gemini | gemini-1.5-flash | 1M context |
| Groq | llama-3.1-70b | 128K context |
| DeepSeek | deepseek-chat | 128K context |
| ZenMux | ernie-5.0-thinking | Per-model rates |
| z.ai | glm-4.7 | 128K context |
| Mistral | mistral-large | 128K context |
| Fireworks | llama-3.1-70b | 128K context |
| Together | llama-3.1-70b-turbo | 128K context |
| Azure OpenAI | gpt-4o | 128K context, per-deployment |
| Cerebras | llama-4-scout-17b-16e-instruct | 128K context, ultra-fast wafer-scale |
| Generic OpenAI | (user-configured) | OpenAI-compatible API |
| Generic Anthropic | (user-configured) | Anthropic-compatible API |

### CLI
- Unified `--provider <name>` flag for all 17 providers
- Backward-compatible aliases (`--samba-nova`, `--nebius`)
- `--list-providers` command shows all providers + API key status
- RLM tuning: `--max-depth`, `--max-iterations`
- Model override: `--model <model_name>`
- Query mode: `--query "custom query"` or `-q`
- Output control: `--output report.md` or `-o`

### Core Modules
- `src/vl_rag_graph_rlm/rlm_core.py` — VLRAGGraphRLM class
- `src/vl_rag_graph_rlm/core/parser.py` — FINAL/FINAL_VAR statement extraction
- `src/vl_rag_graph_rlm/core/prompts.py` — System prompt templates
- `src/vl_rag_graph_rlm/core/repl.py` — REPLExecutor with RestrictedPython
- `src/vl_rag_graph_rlm/pipeline.py` — MultimodalRAGPipeline unified API
- `src/vl_rag_graph_rlm/vision.py` — Image encoding and multimodal messages
- `src/vl_rag_graph_rlm/clients/` — Provider client implementations
- `src/vl_rag_graph_rlm/rag/` — RAG components (store, search, rerank, Qwen3-VL)
- `src/vl_rag_graph_rlm/environments/` — Alternative safe execution environments

### Templates
All 17 provider templates implement the full 6-pillar architecture:
- `provider_sambanova.py`
- `provider_nebius.py`
- `provider_openrouter.py`
- `provider_openai.py`
- `provider_anthropic.py`
- `provider_gemini.py`
- `provider_groq.py`
- `provider_deepseek.py`
- `provider_zenmux.py`
- `provider_zai.py`
- `provider_mistral.py`
- `provider_fireworks.py`
- `provider_together.py`
- `provider_azure_openai.py`
- `provider_openai_compatible.py`
- `provider_anthropic_compatible.py`
- `provider_cerebras.py`

### Dependencies
- Core: openai, anthropic, google-generativeai, litellm
- Qwen3-VL: torch, transformers, qwen-vl-utils, pillow
- Document processing: python-pptx, pymupdf, pdf2image
- Utilities: python-dotenv, pydantic, fuzzywuzzy, numpy, jieba, tiktoken, networkx, scikit-learn
- Security: RestrictedPython for safe code execution

### Documentation
- `PRD.md` — Product requirements, architecture overview, provider list
- `ARCHITECTURE.md` — System diagram, component map, template patterns
- `RULES.md` — Coding standards and patterns
- `TODO.md` — Roadmap and planned features
- `CONTRIBUTING.md` — Guide for adding providers and contributions
- `CHANGELOG.md` — This file

### Python Support
- Python 3.9, 3.10, 3.11, 3.12
- MIT License

---

[Unreleased]: https://github.com/nbiish/mai-vl-rag-graph-rlm/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/nbiish/mai-vl-rag-graph-rlm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nbiish/mai-vl-rag-graph-rlm/releases/tag/v0.1.0
