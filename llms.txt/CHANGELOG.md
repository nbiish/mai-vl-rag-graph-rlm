# Changelog — VL-RAG-Graph-RLM

> **Documentation Index:** See [README.md](README.md) for all documentation files.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-12

### Modal Research Provider (New!)
**Free GLM-5 745B frontier inference** via Modal Research's experimental OpenAI-compatible endpoint:
- **Model:** `zai-org/GLM-5-FP8` — 745B MoE (44B active), FP8 quantized, MIT license
- **Endpoint:** `https://api.us-west-2.modal.direct/v1` — 8×B200 GPUs via SGLang
- **Performance:** 30-75 tok/s per user, frontier-class reasoning
- **Status:** Experimental (free tier: 1 concurrent request, may have intermittent downtime)
- **Get key:** https://modal.com/glm-5-endpoint
- **Documentation:** https://modal.com/blog/try-glm-5

**Files changed:**
- `src/vl_rag_graph_rlm/clients/openai_compatible.py` — `ModalResearchClient` class
- `src/vl_rag_graph_rlm/clients/hierarchy.py` — `modalresearch` added to `DEFAULT_HIERARCHY` (first position)
- `src/vl_rag_graph_rlm/clients/__init__.py` — Import, `get_client()` routing, `__all__`
- `src/vl_rag_graph_rlm/types.py` — `modalresearch` in `ProviderType` literal
- `src/vl_rag_graph_rlm/rlm_core.py` — `_get_default_model()` and `_get_recursive_model()` defaults
- `src/vrlmrag.py` — `SUPPORTED_PROVIDERS` entry
- `templates/provider_modalresearch.py` — New provider template
- `.env` — `MODAL_RESEARCH_API_KEY`, `MODAL_RESEARCH_API_KEY_FALLBACK`, `MODAL_RESEARCH_MODEL`
- `.env.example` — Modal Research section with documentation

### Fallback API Key System (New!)
**Universal multi-account support** with automatic fallback when primary API keys fail:
- **Pattern:** `{PROVIDER}_API_KEY_FALLBACK` — every provider supports this suffix
- **Retry chain:** Primary key → Fallback key (same provider, different account) → Model fallback → Provider hierarchy
- **Use cases:** Credit distribution across accounts, rate limit mitigation, billing redundancy
- **Key promotion:** After fallback succeeds, it becomes primary for the rest of the session

**Implementation:**
- `OpenAICompatibleClient` (base class) — fallback key retry in `_raw_completion()` / `_raw_acompletion()`
- `AnthropicClient` — fallback key support in `completion()` / `acompletion()`
- `AnthropicCompatibleClient` — inherits fallback from `AnthropicClient`
- `GeminiClient` — fallback key support in `completion()` / `acompletion()`

**Files changed:**
- `src/vl_rag_graph_rlm/clients/openai_compatible.py` — `_fallback_api_key`, `_fallback_key_client`, `_fallback_key_async_client`, `_using_fallback_key`, `_openai_kwargs`
- `src/vl_rag_graph_rlm/clients/anthropic.py` — Logging import, fallback key attributes, `_get_fallback_key_client()`, `_get_fallback_key_async_client()`, retry logic in `completion()`/`acompletion()`
- `src/vl_rag_graph_rlm/clients/gemini.py` — Logging import, fallback key attributes, `_get_fallback_key_client()`, retry logic in `completion()`/`acompletion()`
- `.env.example` — `_FALLBACK` key documentation for ALL providers (15+ entries)

### Omni Model Fallback Chain (New!)
Three-tier resilient multimodal processing for images, audio, and video:
- **Primary Omni:** ZenMux `inclusionai/ming-flash-omni-preview` — handles text, images, audio, video
- **Secondary Omni:** ZenMux `gemini/gemini-3-flash-preview` — fallback when primary fails
- **Tertiary Omni:** OpenRouter `google/gemini-3-flash-preview` — final omni fallback
- **Legacy VLM:** OpenRouter `moonshotai/kimi-k2.5` — images/video only (no audio support)

**Implementation:**
- New environment variable naming: `VRLMRAG_OMNI_*` for primary/secondary/tertiary omni models
- Backward compatibility: Old `VRLMRAG_VLM_*` variables still work
- Audio transcription routes through full omni chain: primary → secondary → tertiary
- Image/video description uses full chain: primary → secondary → tertiary → legacy VLM
- Consistent variable pattern: `VRLMRAG_OMNI_BASE_URL`, `VRLMRAG_OMNI_MODEL`, `VRLMRAG_OMNI_API_KEY` (plus `_FALLBACK` and `_FALLBACK_2` variants)

**Files changed:**
- `.env.example` — New organized "Omni Models" section with clear documentation
- `.env` — Updated with new omni model configuration (preserves existing keys)
- `src/vl_rag_graph_rlm/rag/api_embedding.py` — Updated `APIEmbeddingProvider` with new omni variable names and fallback chain

### Enhanced Document Processing
- **PDF support** — PyMuPDF (`fitz`) extracts text and images from PDF documents
  - Handles multi-page PDFs with page-by-page processing
  - Extracts embedded images and figures
  - Falls back gracefully if PyMuPDF not installed
- **DOCX support** — `python-docx` extracts text and tables from Word documents
  - Paragraph-level text extraction
  - Table cell content as structured text
- **CSV/Excel support** — Tabular data ingestion with natural language chunking
  - CSV via standard library `csv` module
  - Excel (.xlsx) via `openpyxl`
  - Rows converted to natural language sentences for embedding
  - Column headers preserved for context
- **Sliding window chunking** — Configurable text segmentation
  - `--chunk-size N` — Characters per chunk (default: 1000)
  - `--chunk-overlap N` — Overlap between chunks (default: 100)
  - Better than simple header-based chunking for long documents

### RAG Improvements
- **BM25 keyword search** — State-of-the-art keyword retrieval
  - Uses `rank-bm25` library for Okapi BM25 scoring
  - Automatic fallback to simple token-overlap if library not installed
  - Better term frequency and document length normalization
- **SQLite backend** — Alternative to JSON storage for vector store
  - `--use-sqlite` CLI flag enables SQLite persistence
  - Better performance with large collections (10K+ documents)
  - Transaction safety and concurrent read access
  - Automatic table creation with proper indexing
- **Configurable RRF weights** — Tune dense vs keyword search balance
  - `--rrf-dense-weight W` — Weight for dense embedding search (default: 4.0)
  - `--rrf-keyword-weight W` — Weight for BM25 keyword search (default: 1.0)
  - Allows domain-specific tuning (e.g., keyword-heavy for legal docs)
- **Multi-query retrieval** — Generate sub-queries for broader recall
  - `--multi-query` CLI flag activates RLM-powered sub-query generation
  - Generates 2-3 complementary queries covering different aspects
  - Automatically deduplicates generated queries
  - Aggregates results from all sub-queries

### Knowledge Graph Enhancements
- **Graph visualization export** — Multiple diagram formats
  - `--export-graph PATH` exports knowledge graph to file
  - `--graph-format` supports: `mermaid` (default), `graphviz` (DOT), `networkx`
  - Mermaid format for GitHub/GitLab rendering
  - Graphviz DOT for professional diagram tools
  - NetworkX pickle/JSON for programmatic analysis
- **Graph statistics** — `--graph-stats` shows KG metrics
  - Entity count and relationship count
  - Entity type distribution
  - Connected ratio (relationships per entity)
- **Entity deduplication** — Merge similar entities automatically
  - `--deduplicate-kg` applies fuzzy matching merges
  - `--dedup-threshold T` controls sensitivity (default: 0.85, range 0-1)
  - `--dedup-report` previews merges without applying
  - Fuzzy string matching with normalization ("The Company Inc." ≈ "Company")
  - Updates relationships to use canonical entity names
- **NetworkX serialization** — `export_to_networkx()` creates structured graphs
  - Entities as nodes with type/description attributes
  - Relationships as edges with relation type attributes
  - Compatible with NetworkX graph algorithms

### New Files
- `src/vl_rag_graph_rlm/rag/sqlite_store.py` — SQLite backend for MultimodalVectorStore
- `src/vl_rag_graph_rlm/kg_visualization.py` — Graph export/visualization utilities
- `src/vl_rag_graph_rlm/kg_deduplication.py` — Entity deduplication and coreference resolution

## [0.1.5] - 2026-02-12

### Model Upgrade Workflows
- **`--reindex` CLI flag** — Force re-embedding of all documents with current embedding model
  - Works with regular input paths and collections (`-c <name> --reindex`)
  - Clears existing embeddings, re-processes all source documents
  - Useful when upgrading to a new embedding model or repairing corrupted indices
- **`--rebuild-kg` CLI flag** — Regenerate knowledge graph with current RLM model
  - Works with regular input paths and collections (`-c <name> --rebuild-kg`)
  - Clears existing KG, re-extracts entities/relationships from all documents
  - Uses current provider hierarchy for RLM calls
- **`--model-compare` CLI flag** — Compare embeddings between old and new models
  - Runs query with both old model and current model
  - Shows unified diff of responses
  - Reports response length differences and divergence
- **`--check-model` CLI flag** — Check collection compatibility with target embedding model
  - Reports current model, target model, compatibility status
  - Warns if collection has mixed models from different sources
  - Suggests reindex command if needed
- **`collection_reindex` MCP tool** — Reindex a collection via MCP server
  - Clears embeddings, re-adds all recorded sources
  - Returns summary: cleared files, re-added sources, current doc/chunk counts
- **`collection_rebuild_kg` MCP tool** — Rebuild KG for a collection via MCP server
  - Clears knowledge graph, re-processes all sources with current RLM
  - Returns summary: previous KG existed, re-processed sources, new KG size
- **Automatic model version tracking in collection metadata**
  - `embedding_model` field tracks current embedding model for collection
  - `reranker_model` field tracks reranker model
  - `model_history` array tracks all model changes over time
  - Per-source model tracking in `sources[].embedding_model`
- **`check_model_compatibility()` helper** — Check if collection can use target model
  - Returns compatibility info: current_model, target_model, needs_reindex, mixed_models
  - Detects mixed model collections (sources embedded with different models)
- **Collection info shows model details**
  - `vrlmrag -c <name> --collection-info` now displays embedding model
  - Shows model change history (last 5 changes)
  - Warns if collection has mixed embedding models
- **`--quality-check` CLI flag** — RLM-powered embedding quality assessment
  - Analyzes collection metadata and knowledge graph
  - Uses recursive LLM to evaluate retrieval quality
  - Provides quality score (0-100), strengths, weaknesses, recommendations
  - Detects score in RLM response and shows status (Excellent/Good/Fair/Poor)
- **Collection tools now total 7** — `collection_add`, `collection_query`, `collection_list`, `collection_info`, `collection_delete`, `collection_reindex`, `collection_rebuild_kg`

### RLM Architecture Verified
- **True Recursive Language Model implementation confirmed** — VLRAGGraphRLM matches Alex Zhang & Omar Khattab (MIT 2025) specification:
  - **REPL environment** — `REPLExecutor` with RestrictedPython sandbox (`environments/repl.py`)
  - **Context as variable** — Large context stored as Python variable, NOT fed to neural network
  - **`recursive_llm()` function** — Injected into REPL env for spawning sub-RLM calls (`rlm_core.py:376`)
  - **`FINAL()` / `FINAL_VAR()` parsing** — `find_final_answer()` extracts answers from RLM responses (`utils/parsing.py:28-60`)
  - **True recursive calls** — Child `VLRAGGraphRLM` at depth+1 with cheaper model (`rlm_core.py:389-400`)
  - **Infinite context capability** — Root LM only sees query, uses code to explore context programmatically
- **System prompt** — "You cannot see the context directly. You MUST write Python code to search and explore it" (`core/prompts.py:19-20`)

### API-Default Mode & Safety
- **CLI defaults to API mode** — Local Qwen3-VL requires explicit `--local` flag or `VRLMRAG_LOCAL=true`
- **Media safety block** — Video/audio files force API mode regardless of `--local` flag (prevents OOM crashes)
- **`--offline` mode** — Graceful fallback when APIs unavailable (blocks video/audio for safety)
- **MCP server defaults to API mode** — `use_api: bool = True` in MCPSettings

### Files Changed
- `src/vrlmrag.py` — `--reindex`, `--rebuild-kg`, `--model-compare`, `--offline` flags, collection-level operations
- `src/vl_rag_graph_rlm/mcp_server/server.py` — `collection_reindex`, `collection_rebuild_kg` MCP tools
- `llms.txt/TODO.md` — Updated roadmap with completed model upgrade workflows
- `llms.txt/CHANGELOG.md` — This entry

## [0.1.4] - 2026-02-11

### Text-Only Embedding Mode (Lightweight Local RAG)
- **Three mutually exclusive embedding modes:**
  1. **TEXT_ONLY** — `VRLMRAG_TEXT_ONLY=true` / `--text-only` — ~1.2 GB RAM, fully offline, text only
  2. **API** — `VRLMRAG_USE_API=true` / `--use-api` — ~200 MB RAM, requires internet (OpenRouter + ZenMux)
  3. **MULTIMODAL** (default) — both flags false — ~4.6 GB RAM, fully offline, handles images/videos/PowerPoints
- **`TextOnlyEmbeddingProvider`** — New text-only embedding provider using Qwen3-Embedding-0.6B (~1.2 GB)
  - `rag/text_embedding.py` — Implements same `embed_text()` interface as Qwen3VLEmbeddingProvider
  - Uses Qwen3-Embedding pattern: `Instruct: {instruction}\nQuery: {text}`
  - Last-token pooling with L2 normalization (Qwen3-Embedding best practice)
  - Supports MPS/CUDA/CPU auto-detection
- **`--text-only` CLI flag** — Run CLI in text-only mode (env: `VRLMRAG_TEXT_ONLY=true`)
- **`VRLMRAG_TEXT_ONLY_MODEL`** env var — Configure text-only model (default: `Qwen/Qwen3-Embedding-0.6B`)
  - Options: 0.6B (~1.2 GB), 4B (~6 GB), 8B (~8 GB) — 8B has best MTEB score (70.58)
- **All 4 CLI code paths support text-only:** `run_analysis`, `run_interactive_session`, `run_collection_add`, `run_collection_query`
- **Separate persistence:** Text-only stores use `embeddings_text.json` suffix to avoid collision with multimodal stores

### New MCP Tools
- **`query_text_document`** — Text-only RAG pipeline (internal API)
  - Uses TextOnlyEmbeddingProvider + FlashRank reranker
  - Same 6-pillar pipeline: dense search → keyword → RRF → rerank → KG → RLM
  - Skips image/video processing (text content only)
- **`run_text_only_cli`** — Execute CLI command via subprocess
  - Runs actual `vrlmrag --text-only` command and captures output
  - Provides authentic CLI experience (progress bars, timing info) via MCP

### Model Configuration Updates
- **`.env` restructured** with three-mode toggle documentation
- **Model Configuration section** — All model names externalized to env vars:
  - `VRLMRAG_TEXT_ONLY_MODEL` — Text-only embedding (Qwen3-Embedding-0.6B)
  - `VRLMRAG_LOCAL_EMBEDDING_MODEL` — Multimodal embedding (Qwen3-VL-Embedding-2B)
  - `VRLMRAG_RERANKER_MODEL` — FlashRank reranker (ms-marco-MiniLM-L-12-v2)
  - `VRLMRAG_EMBEDDING_MODEL` — API embedding (openai/text-embedding-3-small)
  - `VRLMRAG_VLM_MODEL` — API VLM (inclusionai/ming-flash-omni-preview)

### Files Changed
- `src/vl_rag_graph_rlm/rag/text_embedding.py` — New text-only embedding provider
- `src/vl_rag_graph_rlm/rag/__init__.py` — TextOnlyEmbeddingProvider exports
- `src/vrlmrag.py` — `--text-only` flag, text-only wired through all 4 functions
- `src/vl_rag_graph_rlm/mcp_server/server.py` — `query_text_document` + `run_text_only_cli` tools
- `.env` — Three-mode toggle, text-only model config
- `.env.example` — Matching template updates

## [0.1.3] - 2026-02-11

### FlashRank Lightweight Reranker (RAM Fix)
- **Replaced Qwen3-VL-Reranker-2B (~4.6 GB) with FlashRank ONNX cross-encoder (~34 MB)**
- **`rag/flashrank_reranker.py`** — `FlashRankRerankerProvider` adapter matching existing `rerank()` interface
- **Zero model swapping** — Embedder + reranker coexist in RAM (~4.6 GB + ~34 MB = ~4.63 GB peak)
- **Removed all sequential load-free-load logic** — No more `del model; gc.collect(); torch.mps.empty_cache()` cycles
- **Root cause fix** — Python + PyTorch on macOS does not reliably free model memory after `del`; sequential loading was accumulating RSS
- **8 new FlashRank tests** — `tests/test_flashrank_reranker.py` covers ranking quality, interface compat, edge cases
- **Optional dependency** — `pip install "vl-rag-graph-rlm[reranker]"` for `flashrank>=0.2.0`

### API-Based Embedding Mode (Fully Implemented)
- **`--use-api` CLI flag** — Switch to API-based embeddings (env: `VRLMRAG_USE_API=true`)
- **OpenRouter text embeddings** — `openai/text-embedding-3-large` (3072 dims) via OpenRouter API
- **ZenMux omni VLM** — `inclusionai/ming-flash-omni-preview` for image/video → text descriptions
- **`rag/api_embedding.py`** — `APIEmbeddingProvider` drop-in replacement for `Qwen3VLEmbeddingProvider`
- **Zero local GPU models** — Peak RAM ~200 MB when API mode is enabled (vs ~4.6 GB local)
- **Dual API keys** — `OPENROUTER_API_KEY` for embeddings, `ZENMUX_API_KEY` for VLM descriptions
- **Override env vars** — `VRLMRAG_EMBEDDING_MODEL`, `VRLMRAG_VLM_MODEL`, `VRLMRAG_EMBEDDING_BASE_URL`, etc.
- **MCP settings** — `use_api` field in `mcp_settings.json` + `VRLMRAG_USE_API` env var
- **Pipeline support** — `MultimodalRAGPipeline(use_api=True)` for Python API usage
- **All 4 CLI code paths** — run_analysis, interactive, collection_add, collection_query support `--use-api`

### Files Changed
- `src/vl_rag_graph_rlm/rag/flashrank_reranker.py` — New FlashRank adapter module
- `src/vl_rag_graph_rlm/rag/api_embedding.py` — New API embedding provider (OpenRouter + ZenMux)
- `src/vrlmrag.py` — All 4 code paths use FlashRank + `use_api` flag wired through
- `src/vl_rag_graph_rlm/pipeline.py` — FlashRank reranker, `use_api` in embedding_provider lazy load
- `src/vl_rag_graph_rlm/mcp_server/server.py` — FlashRank + API embedding support
- `src/vl_rag_graph_rlm/mcp_server/settings.py` — `use_api` field + `VRLMRAG_USE_API` env var
- `src/vl_rag_graph_rlm/rag/__init__.py` — FlashRank + API embedding exports
- `pyproject.toml` — `[reranker]` optional dependency group
- `.env.example` — Full ZenMux omni + OpenRouter embedding documentation
- `tests/test_flashrank_reranker.py` — 8 new tests

## [0.1.2] - 2026-02-11

### Audio Transcription Support
- **`rag/parakeet.py`** — `ParakeetTranscriptionProvider` for NVIDIA Parakeet V3 audio transcription via NeMo
- **Lazy-loaded model** — Parakeet model loads only on first `transcribe()` call, zero RAM at init
- **Dual caching** — In-memory + optional disk cache by SHA-256 file hash
- **`add_audio()` on `MultimodalVectorStore`** — Transcribe audio → embed transcript text with Qwen3-VL
- **`add_audio()` on `MultimodalRAGPipeline`** — Pipeline-level audio support with transcription provider
- **`get_audio_transcription()`** — Retrieve cached transcriptions by document ID
- **Graceful fallback** — Placeholder content if transcription fails or provider not configured
- **Optional dependency** — `pip install "vl-rag-graph-rlm[parakeet]"` for `nemo_toolkit[asr]>=2.0.0`

### Memory-Safe Model Loading
- **Pipeline lazy `@property` loading** — `MultimodalRAGPipeline.__init__` stores config only (207 MB); models load on first access
- **Deferred reranker attachment** — `_ensure_reranker_on_store()` attaches reranker only when reranking is triggered
- **Note:** v0.1.3 replaced the sequential load-free-load pattern with FlashRank coexistence (see above)

### RAM-Safe Video Processing
- **ffmpeg frame extraction** — `_extract_frames_ffmpeg()` replaces torchvision video reader
- **Never loads full video into RAM** — ffmpeg seeks and extracts only needed frames as JPEG
- **Frame list embedding** — Extracted frames passed as `List[str]` to `embed_video()` / `embed_multimodal()`
- **Automatic cleanup** — Temp frame directory removed after embedding via `shutil.rmtree()`
- **Configurable** — `fps` and `max_frames` control extraction density

### Tests
- **`tests/test_audio_integration.py`** — 10 tests covering audio transcription, deduplication, fallback, search
- **RAM profile test** — Verified sequential loading on 40-min 480p YouTube video (peak 6,746 MB)
- **All 32 existing tests pass** — Zero regressions

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

[Unreleased]: https://github.com/nbiish/mai-vl-rag-graph-rlm/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/nbiish/mai-vl-rag-graph-rlm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/nbiish/mai-vl-rag-graph-rlm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nbiish/mai-vl-rag-graph-rlm/releases/tag/v0.1.0
