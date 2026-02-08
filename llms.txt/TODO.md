# TODO — VL-RAG-Graph-RLM

> Keep tasks atomic and testable.

## In Progress

- [ ] Verify interactive mode end-to-end with persistent KG + incremental document addition
- [ ] Verify full pipeline end-to-end with Qwen3-VL embedding + reranking + RAG + Graph + RLM

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
