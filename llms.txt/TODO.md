# TODO — VL-RAG-Graph-RLM

> Keep tasks atomic and testable.

## Summary — Feb 12, 2026 Session

**Session Focus**: ZenMux Omni Model Debugging, VLM Fallback Chain, Provider Hierarchy Verification, Video Processing Safeguards

### Key Accomplishments
1. **MODELS.md created** — 342 OpenRouter + 100 ZenMux models documented, sorted by release date
2. **VLM Fallback implemented** — ZenMux Ming omni → OpenRouter Kimi K2.5 fallback chain with circuit breaker
3. **Provider hierarchy tested** — 7/15 providers ready, verified fallback behavior on failure
4. **Video processing safeguards** — Critical try-except wrapper in `_process_media()` prevents system crashes
5. **API-default mode confirmed** — CLI and MCP server both default to API mode, `--local` flag for opt-in

### Files Modified
- `src/vl_rag_graph_rlm/rag/api_embedding.py` — VLM fallback chain, Kimi K2.5 as fallback
- `src/vrlmrag.py` — Critical safety wrapper in `_process_media()` (lines 428-531)
- `.env` — VLM fallback model configuration
- `.env.example` — Documentation updates
- `MODELS.md` — New comprehensive model documentation
- `llms.txt/TODO.md` — This file

### Test Results
- ✅ Video processing: Spectrograms video → 58 embeddings stored, query answered
- ✅ PowerPoint processing: "Overview of International Business" → 15 chunks, 11 images
- ✅ Provider fallback: DeepSeek-V3-0324 → V3.1 working correctly
- ✅ No system crashes with media processing safeguards

## In Progress

- [ ] Verify interactive mode end-to-end with persistent KG + incremental document addition
- [ ] Verify full pipeline end-to-end with Qwen3-VL embedding + reranking + RAG + Graph + RLM

## Issues Found — Feb 12, 2026 (Hierarchy Failure Testing)

### Critical: No Graceful Degradation When All Providers Fail
- **Problem**: When ALL API providers fail (invalid keys, rate limits, no credits), system crashes with unhandled errors
- **Test Results**:
  - PowerPoint with all invalid API keys → Providers exhaust hierarchy → crashes on embedding API failure
  - Video with all invalid API keys → Same pattern, crashes during query phase
- **Error Chain**: SambaNova (fail) → Nebius (fail) → Groq (fail) → ... → OpenRouter (fail) → Embedding API fails → Crash
- **Root Cause**: API embedding (`openai/text-embedding-3-small`) requires valid OpenRouter key even when hierarchy falls through

### Video Processing System Crash Prevention (Feb 12, 2026)
- [x] **Media safety block at CLI level** — Video/audio files force API mode regardless of `--local` flag (lines 2626-2632)
- [x] **Critical safety wrapper in `_process_media()`** — All media processing wrapped in try-except to prevent system crashes
- [x] **Graceful degradation on failure** — Returns empty document with error message instead of crashing
- [x] **Parakeet transcription error handling** — Catches and logs errors without crashing
- [x] **ffmpeg extraction error handling** — Continues without audio/frames if extraction fails

### Needed Fixes
- [ ] Add `--offline` mode that uses local Qwen3-VL embeddings when all API providers fail
- [ ] Add graceful error handling when hierarchy exhausted — return helpful message instead of crash
- [ ] Add local embedding provider fallback for API embedding failures
- [ ] Add circuit breaker for entire provider hierarchy (not just individual providers)
- [ ] Document minimum required providers for video processing (OpenRouter for embeddings + ZenMux/Kimi for VLM)

## Completed (Feb 12, 2026)

### Model Documentation & VLM Fallback Update (Feb 12, 2026)
- [x] **MODELS.md created** — Documented 342 OpenRouter models + 100 ZenMux models sorted by release date
- [x] **VLM fallback updated** — `moonshotai/kimi-k2.5` replaces Kimi K2 (256K context, text+image multimodal)
- [x] **.env updated** — `VRLMRAG_VLM_FALLBACK_MODEL=moonshotai/kimi-k2.5`
- [x] **.env.example updated** — Same Kimi K2.5 fallback documentation
- [x] **api_embedding.py updated** — `DEFAULT_VLM_FALLBACK_MODEL=moonshotai/kimi-k2.5`

### Hierarchy Failure Testing (Feb 12, 2026)
- [x] **Provider hierarchy verified** — 7/15 providers ready (sambanova, nebius, groq, cerebras, zai, zenmux, openrouter)
- [x] **PowerPoint test with invalid keys** — Hierarchy falls through all providers, crashes on embedding failure
- [x] **Video test with invalid keys** — Same pattern, no offline fallback available
- [x] **Local mode test attempted** — Should work for PowerPoint (Qwen3-VL), but video blocked

### API-Default Mode & Video Processing (Feb 12, 2026)
- [x] **CLI defaults to API mode** — `--local` flag required to opt into local models (default: API)
- [x] **Video processing tested** — Spectrograms video processed via ZenMux omni + Kimi K2.5 fallback
- [x] **Media safety block verified** — Video/audio files force API mode regardless of `--local` flag
- [x] **MCP server API-default verified** — `use_api: bool = True` in MCPSettings

## Roadmap — v0.2.0

### Document Processing
- [ ] PDF text extraction via PyMuPDF (`pymupdf`)
- [ ] PDF image extraction (figures, charts, diagrams)
- [ ] DOCX document processing support
- [ ] CSV / Excel tabular data ingestion
- [ ] Chunking strategy: sliding window with overlap (configurable chunk_size, overlap)

### RAG Improvements
- [ ] BM25 keyword search (replace simple token-overlap with rank-bm25)
- [ ] Persistent vector store with SQLite backend (replace JSON)
- [ ] Configurable RRF weights via CLI flags
- [ ] Multi-query retrieval (generate sub-queries for broader recall)

### Knowledge Graph
- [ ] Structured graph output (NetworkX serialization)
- [ ] Graph visualization (Mermaid / Graphviz export)
- [ ] Entity deduplication and coreference resolution
- [ ] Graph-augmented retrieval (traverse graph edges for context expansion)

### Collection Enhancements
- [ ] `--collection-export <name> <path>` — export a collection as a portable archive (tar.gz)
- [ ] `--collection-import <path>` — import a collection archive from another machine
- [ ] `--collection-merge <src> <dst>` — merge one collection into another (embeddings + KG)
- [ ] `--collection-tag <name> <tag>` — tag collections for organization and filtering
- [ ] `--collection-search <query>` — search across all collections without specifying names
- [ ] Collection-level metadata: custom key-value pairs, creation notes, version tracking
- [ ] Remote collection sync (S3/GCS) — push/pull collections to cloud storage
- [ ] Collection snapshots — save/restore point-in-time versions
- [ ] Collection statistics dashboard — embedding distribution, KG entity counts, query history
- [ ] Automatic collection suggestions — recommend relevant collections based on query content

### CLI & UX
- [ ] `--format json` output option (machine-readable results)
- [ ] `--verbose` / `--quiet` log level control
- [ ] `--no-embed` flag to skip VL embedding (text-only fallback)
- [ ] `--cache` flag to reuse existing .vrlmrag_store embeddings
- [ ] Progress bars (tqdm) for embedding and search steps
- [ ] Streaming output for RLM responses
- [ ] `--dry-run` flag for collection operations (show what would be added)
- [ ] Tab completion for collection names in shell

### Testing & CI
- [ ] Unit tests for DocumentProcessor (PPTX, TXT, MD)
- [ ] Unit tests for _keyword_search and RRF fusion
- [ ] Unit tests for collection CRUD operations (create, list, delete, record_source)
- [ ] Unit tests for collection blending (merge stores, merge KGs)
- [ ] Integration test: full pipeline with mock LLM provider
- [ ] Integration test: collection add → query round-trip
- [ ] CI pipeline (GitHub Actions) with lint + test
- [ ] Benchmark suite: embedding speed, search recall, end-to-end latency

### Provider Improvements
- [ ] Migrate `google-generativeai` → `google-genai` (deprecation warning)
- [ ] Add Ollama provider (local LLM inference)
- [ ] Add vLLM provider (self-hosted high-throughput)
- [ ] Token usage tracking and cost estimation per provider
- [ ] Rate limiting / retry logic with exponential backoff

## Completed (v0.1.x — Feb 2026)

### API-Default Mode & Media Safety (Feb 11, 2026)
- [x] **API mode is now the default** — local Qwen3-VL requires explicit `--local` flag or `VRLMRAG_LOCAL=true`
- [x] **`--local` CLI flag**: Opt into local Qwen3-VL models (replaces old `--use-api` flag)
- [x] **Media safety block**: Local models are BLOCKED for video/audio files — always forces API mode to prevent OOM crashes
- [x] **MCP server defaults to API mode** (`use_api: bool = True` in MCPSettings)
- [x] **Audio/video processing via DocumentProcessor**: `_process_media()` extracts audio (ffmpeg), transcribes (Parakeet ASR local), extracts key frames
- [x] **Video frame embedding**: Frames embedded via `add_image()` in all paths (run_analysis, interactive, collections, MCP)
- [x] **Parakeet ASR integration**: `create_parakeet_transcriber()` wired into DocumentProcessor for local audio transcription
- [x] **API embedding circuit breaker**: VLM disabled after 3 consecutive failures — prevents hanging on broken providers
- [x] **API client timeouts**: 30s embedding, 15s VLM — prevents infinite hangs on slow/broken APIs
- [x] **`.env.example` updated**: Audio/video config, embedding mode toggle docs, Parakeet model override

### Persistent Vector Store & Incremental Re-indexing (Feb 11, 2026)
- [x] **Manifest-based change detection**: `manifest.json` tracks indexed files + mtimes in `.vrlmrag_store/`
- [x] **Smart store reuse (CLI)**: Re-running on unchanged files prints "Store up-to-date" and skips all document processing + embedding
- [x] **Incremental updates (CLI)**: Only new/modified files are re-processed; existing embeddings preserved via SHA-256 dedup
- [x] **Smart store reuse (MCP)**: `query_document` and `query_text_document` use manifest to skip re-processing
- [x] **CWD default (MCP)**: `input_path="."` or empty defaults to current working directory
- [x] **Chunk reconstruction from store**: When store is reused (no processing), chunks are reconstructed from stored documents for fallback reranking
- [x] **KG merge on incremental update**: New KG fragments are merged with existing knowledge graph instead of replacing
- [x] **Store status in response**: MCP tools report "store reused" vs "store updated" + embedding count in response footer
- [x] **Manifest helpers**: `_load_manifest()`, `_save_manifest()`, `_scan_supported_files()`, `_detect_file_changes()` shared across CLI and MCP

### SambaNova DeepSeek-V3 Context Fix (Feb 11, 2026)
- [x] **Default model switched**: `DeepSeek-V3.2` (8K tokens) → `DeepSeek-V3-0324` (32K context, production)
- [x] **Fallback model**: `DeepSeek-V3.1` (32K+ context) — safe fallback for any V3-0324 error
- [x] **Context budget increased**: SambaNova `context_budget` 8,000 → 32,000 chars (matching 32K token window)
- [x] **Smart context truncation**: `completion()` detects "maximum context length" errors → truncates input by 50% → retries before model fallback
- [x] **Async truncation**: Same safeguard in `acompletion()` for MCP server async paths
- [x] **DeepSeek-V3.2 marked as legacy**: `legacy_8k` tag in RECOMMENDED_MODELS, warning in docstrings and .env.example
- [x] **All hardcoded defaults updated**: `rlm_core.py`, `openai_compatible.py`, `vrlmrag.py` SUPPORTED_PROVIDERS

### Named Persistent Collections (Feb 8, 2026)
- [x] **`collections.py` module**: CRUD for named collections (`create`, `list`, `delete`, `load_meta`, `record_source`)
- [x] **Collection storage layout**: `collections/<name>/` with `collection.json`, `embeddings.json`, `knowledge_graph.md`
- [x] **`-c <name> --add <path>`**: Add documents to a named collection (embed + KG extract + persist)
- [x] **`-c <name> -q "..."`**: Query a collection via full VL-RAG pipeline (scriptable, non-interactive)
- [x] **`-c A -c B -q "..."`**: Blend multiple collections — merge stores and KGs for cross-collection queries
- [x] **`-c <name> -i`**: Interactive session backed by a collection's store directory
- [x] **`--collection-list`**: List all collections with doc/chunk counts and last-updated timestamps
- [x] **`--collection-info`**: Detailed info for a collection (sources, embedding count, KG size)
- [x] **`--collection-delete`**: Delete a collection and all its data
- [x] **`collections/.gitignore`**: Collection data excluded from version control

### Accuracy-First Query Pipeline (Feb 8, 2026)
- [x] **Unified `_run_vl_rag_query()`**: Single source of truth for all query paths (run_analysis + interactive)
- [x] **Retrieval instruction pairing**: `_DOCUMENT_INSTRUCTION` for ingestion, `_QUERY_INSTRUCTION` for search
- [x] **Wider retrieval depth**: `top_k=50` dense/keyword, `30` reranker candidates, `10` final results
- [x] **Structured KG extraction prompt**: Typed entities + explicit relationships (`EntityA → rel → EntityB`)
- [x] **KG budget increased**: Up to 8000 chars (⅓ of context budget) prepended to every query
- [x] **Eliminated duplicated query logic**: Both run_analysis() and interactive mode delegate to shared function

### Universal Persistent Embeddings & Interactive Mode (Feb 8, 2026)
- [x] **Content-based deduplication (SHA-256)**: `MultimodalVectorStore` skips re-embedding already-stored content
- [x] **Universal KG persistence**: Knowledge graph saved/merged in both `run_analysis()` and interactive mode
- [x] **KG-augmented queries in all modes**: Knowledge graph context prepended to every query (not just interactive)
- [x] **Incremental embedding**: Re-running on same folder only embeds new/changed files
- [x] **Provider-agnostic store**: Same `.vrlmrag_store/` used regardless of provider/model combo
- [x] **`--interactive` / `-i` CLI flag**: Persistent session with VL models loaded once
- [x] **REPL loop**: `/add <path>`, `/kg`, `/stats`, `/save`, `/help`, `/quit` commands
- [x] **Incremental document addition**: `/add` embeds new docs and extends KG without reloading VL models
- [x] **Embedding persistence**: `embeddings.json` reloaded on restart (no re-embedding)
- [x] **`--store-dir` flag**: Custom persistence directory
- [x] **Provider hierarchy order updated**: sambanova → nebius → groq → cerebras → zai → zenmux → openrouter → gemini → deepseek → openai → ...
- [x] **SDK priority**: `openai_compatible` / `anthropic_compatible` auto-prepended if API keys set

### Universal Model Fallback (Feb 8, 2026)
- [x] **`FALLBACK_MODELS` dict**: Hardcoded fallback models for 11+ providers in base class
- [x] **`{PROVIDER}_FALLBACK_MODEL` env var**: Override fallback per-provider
- [x] **Base class `completion()`/`acompletion()`**: Try primary → catch any Exception → retry with fallback
- [x] **`_raw_completion()`/`_raw_acompletion()`**: Low-level methods for providers with custom fallback (z.ai endpoint)
- [x] **SambaNovaClient simplified**: Removed custom overrides, now inherits universal fallback
- [x] **ZaiClient restructured**: Uses `_raw_completion` for endpoint fallback, base class handles model fallback
- [x] **Two-tier resilience**: Model fallback (same provider) → Provider hierarchy fallback (next provider)
- [x] **z.ai three-tier**: Coding Plan endpoint → Normal endpoint → Model fallback → Provider hierarchy

### Provider Hierarchy & Auto Mode
- [x] **`HierarchyClient`**: Automatic fallback through configurable provider order
- [x] **`PROVIDER_HIERARCHY` env var**: Editable comma-separated provider order in `.env`
- [x] **`--provider auto`** (default): CLI no longer requires `--provider` flag
- [x] **`--show-hierarchy`**: CLI command to display fallback order + availability
- [x] **`get_client('auto')`**: Python API returns `HierarchyClient` with fallback
- [x] **`HierarchyClient(start_provider='groq')`**: Start hierarchy from a specific provider
- [x] **Auto fallback on errors**: Rate limits, auth errors, network issues trigger next provider
- [x] **CLI packaging verified**: `pip install -e .` → `vrlmrag` command works
- [x] **Client timeout fix**: Added `timeout=120s` + `max_retries=0` to OpenAI clients (openai lib default retries caused 20–80s delays)
- [x] **Fallback model fix**: `_try_fallback_query` no longer passes provider-specific model names to fallback providers

### Full Pipeline E2E Verification (Feb 8, 2026)
- [x] **International Business PPTX**: All 6 pillars exercised — 15 chunks, 11 images, 26 embeddings, KG via SambaNova DeepSeek-V3.2, query via zai fallback
- [x] **Writing Tutorial PPTX**: All 6 pillars exercised — 20 chunks, 20 embeddings, KG + well-structured 10-point answer via fallback
- [x] **SambaNova defaults verified**: DeepSeek-V3.2 default model, 8K char context budget, recursive model DeepSeek-V3.1
- [x] **Hierarchy fallback verified live**: SambaNova rate-limited → auto fell through to zai → correct answer returned
- [x] **Workflow updated**: `.windsurf/workflows/test-international-business.md` uses CLI auto mode

### Provider Model Updates (Feb 7, 2026 — live API-verified)
- [x] **Groq default → `moonshotai/kimi-k2-instruct-0905`** (Kimi K2 on Groq LPU, verified via API)
- [x] **Cerebras default → `zai-glm-4.7`** (GLM 4.7 355B, ~1000 tok/s — `llama-3.3-70b` deprecated Feb 16)
- [x] **SambaNova models updated**: DeepSeek-V3.2 default, also V3.1, gpt-oss-120b, Qwen3-235B, Llama-4-Maverick
- [x] **Nebius models documented**: MiniMax-M2.1 default, also GLM-4.7-FP8, Nemotron-Ultra-253B
- [x] **RECOMMENDED_MODELS dict** updated with Feb 2026 models for all 8 providers
- [x] **All hardcoded defaults and recursive models** updated in `rlm_core.py`
- [x] **All client docstrings** updated with current model lists from live API queries
- [x] **Comprehensive llms.txt/ update**: PRD, ARCHITECTURE, RULES, TODO reflect Feb 2026 landscape

### Provider Integrations
- [x] **ZenMux integration**: Corrected base URL to `https://zenmux.ai/api/v1`, `provider/model` format
- [x] **z.ai Coding Plan integration**: Dual-endpoint (`api.z.ai` Coding Plan first → `open.bigmodel.cn` fallback)
- [x] **All provider connectivity verified**: Cerebras, Groq, Nebius, ZenMux, z.ai (Coding Plan), OpenRouter, SambaNova

### Core Release (v0.1.0)
- [x] Unified CLI with `--provider` flag supporting 17 providers
- [x] `--list-providers`, `--version`, `--model`, `--max-depth`, `--max-iterations` flags
- [x] Backward-compatible `--samba-nova` and `--nebius` aliases
- [x] All 17 provider templates exercising full 6-pillar pipeline
- [x] Nebius Token Factory support (MiniMax-M2.1 default)
- [x] SambaNova Cloud support (DeepSeek-V3.2 default)
- [x] Generic OpenAI-compatible and Anthropic-compatible provider templates
- [x] Upgrade transformers to 5.1.0 for Qwen3-VL (`qwen3_vl` architecture)
- [x] Qwen3-VL visual embeddings verified (26 embedded docs, 11 images)
- [x] Full pipeline test: PPTX → Qwen3-VL embed → hybrid search → RRF → rerank → RLM → report
- [x] Comprehensive documentation: ARCHITECTURE.md, RULES.md, PRD.md, .env.example
