# Changelog — VL-RAG-Graph-RLM

> **Documentation Index:** See [README.md](README.md) for all documentation files.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[Unreleased]: https://github.com/nbiish/mai-vl-rag-graph-rlm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nbiish/mai-vl-rag-graph-rlm/releases/tag/v0.1.0
