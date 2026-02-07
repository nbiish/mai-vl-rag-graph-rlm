# TODO — VL-RAG-Graph-RLM

> Keep tasks atomic and testable.

## In Progress

- (none — v0.1.0 initial release ready)

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
- [ ] Incremental embedding (skip already-embedded chunks on re-run)
- [ ] Configurable RRF weights via CLI flags
- [ ] Multi-query retrieval (generate sub-queries for broader recall)

### Knowledge Graph
- [ ] Structured graph output (NetworkX serialization)
- [ ] Graph visualization (Mermaid / Graphviz export)
- [ ] Entity deduplication and coreference resolution
- [ ] Graph-augmented retrieval (traverse graph edges for context expansion)

### CLI & UX
- [ ] `--format json` output option (machine-readable results)
- [ ] `--verbose` / `--quiet` log level control
- [ ] `--no-embed` flag to skip VL embedding (text-only fallback)
- [ ] `--cache` flag to reuse existing .vrlmrag_store embeddings
- [ ] Progress bars (tqdm) for embedding and search steps
- [ ] Streaming output for RLM responses

### Testing & CI
- [ ] Unit tests for DocumentProcessor (PPTX, TXT, MD)
- [ ] Unit tests for _keyword_search and RRF fusion
- [ ] Integration test: full pipeline with mock LLM provider
- [ ] CI pipeline (GitHub Actions) with lint + test
- [ ] Benchmark suite: embedding speed, search recall, end-to-end latency

### Provider Improvements
- [ ] Migrate `google-generativeai` → `google-genai` (deprecation warning)
- [ ] Add Ollama provider (local LLM inference)
- [ ] Add vLLM provider (self-hosted high-throughput)
- [ ] Token usage tracking and cost estimation per provider
- [ ] Rate limiting / retry logic with exponential backoff

## Completed (v0.1.0)

- [x] Unified CLI with `--provider` flag supporting 15 providers (plus 2 generic compatible templates)
- [x] `--list-providers` command showing API key status for all providers
- [x] `--version`, `--model`, `--max-depth`, `--max-iterations` flags
- [x] Backward-compatible `--samba-nova` and `--nebius` aliases
- [x] Upgrade all 17 provider templates to full 6-pillar pipeline
- [x] Implement Nebius Token Factory support (MiniMax-M2.1 default)
- [x] Implement SambaNova Cloud support (DeepSeek-V3.2 default)
- [x] Implement Cerebras support (llama-3.3-70b default)
- [x] Add generic OpenAI-compatible and Anthropic-compatible provider templates
- [x] Upgrade transformers to 5.1.0 for Qwen3-VL (`qwen3_vl` architecture)
- [x] Verify Qwen3-VL visual embeddings working (26 embedded docs, 11 images)
- [x] Full pipeline test: PPTX → Qwen3-VL embed → hybrid search → RRF → rerank → RLM → report
- [x] Document full architecture in `llms.txt/ARCHITECTURE.md` with 6-pillar component map
- [x] Document coding rules in `llms.txt/RULES.md` with device detection patterns
- [x] Document PRD with six-pillar architecture in `llms.txt/PRD.md` with all 17 providers
- [x] Comprehensive `.env.example` with all 17 providers
- [x] `templates/__init__.py` updated with 6-pillar documentation
- [x] **ZenMux integration fixed**: Corrected base URL to `https://zenmux.ai/api/v1`, updated model format to `provider/model`
- [x] **z.ai Coding Plan integration**: Added dual-endpoint support (Coding Plan first, normal fallback)
- [x] **Groq model fixed**: Updated default to `llama-3.3-70b-versatile`
- [x] **All provider connectivity verified**: 7 of 8 providers with API keys working (SambaNova rate-limited)
