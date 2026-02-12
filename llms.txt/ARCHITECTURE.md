# Architecture — VL-RAG-Graph-RLM

> **Documentation Index:** See [README.md](README.md) for all documentation files.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Document Intake                              │
│  PPTX / PDF / TXT / MD / Video / Audio                              │
│  → text chunks + extracted images + video frames + transcripts     │
└──┬──────────────┬──────────────────┬───────────────┬───────────────┘
   │              │                  │               │
   │              │                  │               ▼
   │              │                  │  ┌──────────────────────────┐
   │              │                  │  │  Audio Transcription     │
   │              │                  │  │  (Parakeet V3 / NeMo)    │
   │              │                  │  │  Lazy-loaded, cached     │
   │              │                  │  └──────────┬───────────────┘
   │              │                  │             │ transcript text
   ▼              ▼                  ▼             ▼
┌──────────────────────────┐    ┌──────────────────────────────────┐
│  Qwen3-VL Embedding      │    │  Qwen3-VL Embedding              │
│  (Text + Transcripts)    │    │  (Images + Video Frames)         │
│  Qwen/Qwen3-VL-Embed-2B  │    │  Qwen/Qwen3-VL-Embed-2B         │
└──────────┬───────────────┘    └──────────┬───────────────────────┘
           │                               │
           ▼                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MultimodalVectorStore (Unified Vector Space)            │
│  Text + Image + Video + Audio embeddings stored together            │
│  JSON persistence at .vrlmrag_store/embeddings.json                 │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               │  Query
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Hybrid Search                                    │
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐                         │
│  │  Dense Search    │    │  Keyword Search  │                        │
│  │  (Cosine Sim)    │    │  (Token Overlap) │                        │
│  │  top_k=50        │    │  top_k=50        │                        │
│  └────────┬────────┘    └────────┬─────────┘                        │
│           │                      │                                   │
│           ▼                      ▼                                   │
│  ┌──────────────────────────────────────────┐                       │
│  │  Reciprocal Rank Fusion (RRF)            │                       │
│  │  dense_weight=4.0, keyword_weight=1.0    │                       │
│  │  k=60                                    │                       │
│  └────────────────────┬─────────────────────┘                       │
│                       │                                              │
│                       ▼                                              │
│  ┌──────────────────────────────────────────┐                       │
│  │  Qwen3-VL Reranker (Cross-Attention)     │                       │
│  │  Qwen/Qwen3-VL-Reranker-2B              │                       │
│  │  Top-30 candidates → Top-10 results      │                       │
│  └────────────────────┬─────────────────────┘                       │
│                       │                                              │
│                       ▼                                              │
│  ┌──────────────────────────────────────────┐                       │
│  │  MultiFactorReranker (Fallback/Boost)    │                       │
│  │  Fuzzy + Keyword + Semantic + Length      │                       │
│  └────────────────────┬─────────────────────┘                       │
└───────────────────────┼─────────────────────────────────────────────┘
                        │
                        ▼  Retrieved Context
┌─────────────────────────────────────────────────────────────────────┐
│                  Knowledge Graph Extraction                          │
│  RLM extracts entities, concepts, relationships from full document  │
│  Graph context augments query answering                             │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Recursive Language Model (RLM)                          │
│                                                                      │
│  VLRAGGraphRLM(provider, model, max_depth=3, max_iterations=10)     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  Iteration Loop (max_iterations)                        │        │
│  │                                                         │        │
│  │  1. LLM generates response (may contain Python code)    │        │
│  │  2. REPL executes code blocks in sandboxed env          │        │
│  │     Available: context, query, re, recursive_llm        │        │
│  │  3. Check for FINAL() answer                            │        │
│  │  4. If no answer, feed output back → next iteration     │        │
│  │                                                         │        │
│  │  Recursive: recursive_llm(sub_query, sub_context)       │        │
│  │  → spawns child VLRAGGraphRLM at depth+1                │        │
│  │  → uses cheaper recursive_model                         │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                      │
│  Providers: SambaNova, Nebius, OpenRouter, OpenAI, Anthropic,       │
│             Gemini, Groq, Cerebras, DeepSeek, ZenMux, z.ai,        │
│             Mistral, Fireworks, Together, Azure OpenAI,             │
│             Generic OpenAI, Generic Anthropic, LiteLLM            │
│                                                                      │
│  Provider Notes (Feb 2026):                                          │
│  - ZenMux: Uses provider/model format (e.g., "moonshotai/kimi-k2.5") │
│  - z.ai: Tries Coding Plan endpoint first, falls back to normal     │
│  - Groq: LPU, default moonshotai/kimi-k2-instruct-0905              │
│  - Cerebras: Wafer-scale, default zai-glm-4.7 (355B, ~1000 tok/s)   │
│  - SambaNova: DeepSeek-V3.2, also V3.1, gpt-oss-120b, Qwen3-235B   │
│  - Nebius: MiniMax-M2.1, also GLM-4.7-FP8, Nemotron-Ultra-253B     │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Markdown Report Output                            │
│  - Provider/model metadata                                          │
│  - Embedding/reranker model info                                    │
│  - Knowledge graph summary                                          │
│  - Query responses with retrieved sources and scores                │
│  - Source document inventory (type, path, image count)              │
└─────────────────────────────────────────────────────────────────────┘
```

## Three Embedding Modes

The system supports three mutually exclusive embedding modes, selected via environment variables or CLI flags:

| Mode | Flag / Env Var | RAM | Network | Best For |
|------|---------------|-----|---------|----------|
| **Text-Only** | `--text-only` / `VRLMRAG_TEXT_ONLY=true` | ~1.2 GB | Offline | `.txt`, `.md`, text-heavy PDFs |
| **API** | `--use-api` / `VRLMRAG_USE_API=true` | ~200 MB | Required | Any content, low-RAM machines |
| **Multimodal** (default) | Both false | ~4.6 GB | Offline | PowerPoints, PDFs with figures, images, video |

### Text-Only Mode

Uses `TextOnlyEmbeddingProvider` with Qwen3-Embedding-0.6B (~1.2 GB RAM). Skips image/video processing entirely — ideal for pure text documents.

```python
from vl_rag_graph_rlm.rag.text_embedding import create_text_embedder

embedder = create_text_embedder(model_name="Qwen/Qwen3-Embedding-0.6B")
embedding = embedder.embed_text("Machine learning uses statistical methods...")
```

**Implementation details:**
- Uses standard `transformers.AutoModel` + `AutoTokenizer` (no Qwen3-VL vision dependencies)
- Qwen3-Embedding format: `"Instruct: {instruction}\nQuery: {text}"`
- Last-token pooling with L2 normalization (Qwen3-Embedding best practice)
- Embedding dimension: 1024 (0.6B), 2048 (4B), 4096 (8B)
- Storage: `embeddings_text.json` (separate from multimodal stores)

### API Mode

Uses `APIEmbeddingProvider` — OpenRouter for text embeddings + ZenMux omni VLM for image/video descriptions. Zero local GPU models.

### Multimodal Mode (Default)

Uses `Qwen3VLEmbeddingProvider` — full vision-language embedding with image/video/audio support. Largest RAM footprint but handles all content types.

## Memory Management — Lightweight Coexistence Model

Peak RAM is kept under ~5 GB by using a **lightweight reranker** that
coexists with the embedder — no model swapping needed.

```
Qwen3-VL-Embedding-2B  ~4.6 GB  (local, stays loaded)
FlashRank MiniLM-L-12   ~34 MB  (ONNX cross-encoder, coexists)
─────────────────────────────────
Total peak:             ~4.63 GB
```

> **Why not sequential load-free-load?**  Python + PyTorch on macOS does
> not reliably return model memory to the OS after `del`.  Each
> load/free cycle accumulated ~1-2 GB of unreclaimable RSS.  The
> FlashRank approach eliminates this entirely.

**Key design decisions:**

| Component | Strategy | Why |
|-----------|----------|-----|
| `MultimodalRAGPipeline` | Lazy `@property` loading | `__init__` stores config only; models load on first access |
| `pipeline.store` | Deferred creation | Store created (and embedder loaded) only when first document is added |
| `pipeline.reranker` | FlashRank ONNX (~34 MB) | Coexists with embedder; no model swapping |
| `pipeline.rlm` | Deferred creation | RLM client created only on first `query()` call |
| MCP server `query_document` | Embedder + FlashRank | Both loaded; no `del`/`gc.collect()` needed |
| CLI `run_analysis()` | Embedder + FlashRank | Both coexist; single ~4.63 GB peak |
| Interactive `/add` | Embedder stays loaded | No model swapping on ingestion |
| Video processing | ffmpeg frame extraction | Never loads full video into RAM; extracts only needed frames as JPEG |
| Audio transcription | Lazy Parakeet V3 | Model loads on first `transcribe()` call; cached by file hash |
| API mode (`--use-api`) | Zero local models | Future: API-based embeddings eliminate all local model RAM |

### Expected RAM Profile

| Phase | Peak RSS | Notes |
|-------|----------|-------|
| Init (pipeline configured) | ~207 MB | Zero models loaded |
| Embedder + FlashRank loaded | ~4,635 MB | Single 2B model + 34 MB ONNX |
| Video embedded (8 frames via ffmpeg) | ~4,635 MB | +0 MB — frames are tiny JPEGs |
| Reranking (FlashRank) | ~4,635 MB | ONNX inference, negligible RAM |
| API mode (future) | ~207 MB | Zero local models |

## Accuracy-First Query Pipeline

Every query — in both `run_analysis()` and interactive mode — goes through
a single shared function `_run_vl_rag_query()` that guarantees the full
6-pillar pipeline executes on every call:

| Stage | Component | Parameters |
|-------|-----------|------------|
| 1. Dense search | Qwen3-VL embedding (cosine sim) | `top_k=50`, instruction: `"Find passages that are relevant to and answer the following query."` |
| 2. Keyword search | Token-overlap scoring | `top_k=50` |
| 3. Fusion | Reciprocal Rank Fusion | `k=60`, weights `[4.0, 1.0]` (dense-heavy) |
| 4. Reranking | FlashRank ONNX cross-encoder (ms-marco-MiniLM-L-12-v2) | `30` candidates → `10` final results |
| 5. KG augmentation | Persisted knowledge graph prepended | Up to `8000` chars (⅓ of context budget) |
| 6. RLM completion | Recursive Language Model | `max_depth=3`, `max_iterations=10`, provider hierarchy fallback |

### Embedding Instruction Pairing

Qwen3-VL embedding quality depends on matching the instruction to the task.
The system uses paired instructions:

- **Document ingestion:** `"Represent this document for retrieval."` — used
  in `add_text()` and `add_image()` during embedding.
- **Query retrieval:** `"Find passages that are relevant to and answer the
  following query."` — used in `store.search()` at query time.

This asymmetric instruction pairing (document vs query) is the recommended
approach from the Qwen3-VL embedding model documentation and significantly
improves retrieval precision.

### Structured Knowledge Graph Extraction

KG extraction uses a structured prompt (`_KG_EXTRACTION_PROMPT`) that
produces typed entities (Person, Organisation, Concept, Technology, etc.)
and explicit relationships (`EntityA → relationship → EntityB`). The KG
is persisted to `knowledge_graph.md` and merged across runs.

## Component Map

### Source Files

| Component | File | Description |
|-----------|------|-------------|
| Core RLM | `src/vl_rag_graph_rlm/rlm_core.py` | `VLRAGGraphRLM` class, recursive completion, REPL |
| Core Parser | `src/vl_rag_graph_rlm/core/parser.py` | FINAL/FINAL_VAR statement extraction |
| Core Prompts | `src/vl_rag_graph_rlm/core/prompts.py` | System prompt templates for RLM |
| Core REPL | `src/vl_rag_graph_rlm/core/repl.py` | REPLExecutor with RestrictedPython sandbox |
| Utils Parsing | `src/vl_rag_graph_rlm/utils/parsing.py` | LLM response parsing, code block extraction |
| Utils Prompts | `src/vl_rag_graph_rlm/utils/prompts.py` | Additional prompt utilities |
| Pipeline | `src/vl_rag_graph_rlm/pipeline.py` | `MultimodalRAGPipeline` unified API (lazy `@property` model loading) |
| Vision | `src/vl_rag_graph_rlm/vision.py` | Image encoding, multimodal message formatting |
| Clients | `src/vl_rag_graph_rlm/clients/` | Provider clients (OpenAI-compatible, Anthropic, Gemini, LiteLLM) |
| RAG Init | `src/vl_rag_graph_rlm/rag/__init__.py` | `SearchResult`, `RRF`, `MultiFactorReranker`, `HybridSearcher` |
| RAG Provider | `src/vl_rag_graph_rlm/rag/provider.py` | RAG provider interface |
| RAG Store | `src/vl_rag_graph_rlm/rag/store.py` | Vector store base module |
| RAG Reranker | `src/vl_rag_graph_rlm/rag/reranker.py` | Reranker implementations |
| Qwen3-VL | `src/vl_rag_graph_rlm/rag/qwen3vl.py` | `Qwen3VLEmbeddingProvider`, `Qwen3VLRerankerProvider` |
| Text Embedding | `src/vl_rag_graph_rlm/rag/text_embedding.py` | `TextOnlyEmbeddingProvider` — lightweight text-only embeddings |
| API Embedding | `src/vl_rag_graph_rlm/rag/api_embedding.py` | `APIEmbeddingProvider` — OpenRouter + ZenMux omni |
| FlashRank | `src/vl_rag_graph_rlm/rag/flashrank_reranker.py` | `FlashRankRerankerProvider` — lightweight ONNX reranker |
| ERNIE Client | `src/vl_rag_graph_rlm/rag/ernie_client.py` | Baidu ERNIE / OpenAI-compatible client |
| Vector Store | `src/vl_rag_graph_rlm/rag/multimodal_store.py` | `MultimodalVectorStore` with text + image + video + audio storage |
| Parakeet | `src/vl_rag_graph_rlm/rag/parakeet.py` | `ParakeetTranscriptionProvider` — lazy-loaded audio transcription via NeMo |
| Collections | `src/vl_rag_graph_rlm/collections.py` | Named persistent knowledge stores (CRUD, KG helpers) |
| Environments REPL | `src/vl_rag_graph_rlm/environments/repl.py` | Alternative safe Python execution sandbox |
| CLI | `src/vrlmrag.py` | Unified `--provider <name>` CLI for all 17 providers |
| MCP Server | `src/vl_rag_graph_rlm/mcp_server/server.py` | FastMCP server with 10 tools, stdio transport |
| MCP Settings | `src/vl_rag_graph_rlm/mcp_server/settings.py` | Settings loader, template resolution, hierarchy defaults |
| Types | `src/vl_rag_graph_rlm/types.py` | Dataclasses for completions, usage, results |

### Templates

| Template | Provider | Full Pipeline |
|----------|----------|---------------|
| `provider_sambanova.py` | SambaNova | Yes — all 6 pillars |
| `provider_nebius.py` | Nebius | Yes — all 6 pillars |
| `provider_openrouter.py` | OpenRouter | Yes — all 6 pillars |
| `provider_openai.py` | OpenAI | Yes — all 6 pillars |
| `provider_anthropic.py` | Anthropic | Yes — all 6 pillars |
| `provider_gemini.py` | Gemini | Yes — all 6 pillars |
| `provider_groq.py` | Groq | Yes — all 6 pillars |
| `provider_deepseek.py` | DeepSeek | Yes — all 6 pillars |
| `provider_zenmux.py` | ZenMux | Yes — all 6 pillars |
| `provider_zai.py` | z.ai | Yes — all 6 pillars |
| `provider_mistral.py` | Mistral | Yes — all 6 pillars |
| `provider_fireworks.py` | Fireworks | Yes — all 6 pillars |
| `provider_together.py` | Together | Yes — all 6 pillars |
| `provider_azure_openai.py` | Azure OpenAI | Yes — all 6 pillars |
| `provider_openai_compatible.py` | Generic OpenAI | Yes — all 6 pillars |
| `provider_anthropic_compatible.py` | Generic Anthropic | Yes — all 6 pillars |
| `provider_cerebras.py` | Cerebras | Yes — all 6 pillars |

### Dependencies

```
transformers>=5.1.0          # Qwen3-VL model support
qwen_vl_utils>=0.0.14        # Vision processing for Qwen3-VL
torch                         # Model inference (MPS/CUDA/CPU)
torchvision                   # Image transforms
openai                        # OpenAI-compatible API clients
anthropic                     # Anthropic API client
google-generativeai           # Gemini API client
python-dotenv                 # .env file loading
python-pptx                   # PowerPoint processing
Pillow                        # Image handling
ffmpeg (system)               # Video frame extraction (RAM-safe, replaces torchvision video reader)

# Optional: Audio transcription
nemo_toolkit[asr]>=2.0.0     # pip install "vl-rag-graph-rlm[parakeet]"
```

## Full Pipeline Flow (Template Pattern)

Every provider template follows this pattern. Note the **sequential model
loading** — the embedder is freed before the reranker loads.

```python
import gc, torch

# 1. Document Processing
processor = DocumentProcessor()
documents = processor.process_path(input_path)

# 2. Qwen3-VL Embedding (text + images + video + audio → unified vector space)
embedder = create_qwen3vl_embedder(device="mps")
store = MultimodalVectorStore(embedding_provider=embedder, storage_path=...)
for chunk in chunks:
    store.add_text(content=chunk["content"], metadata={...},
                   instruction=_DOCUMENT_INSTRUCTION)
for image in images:
    store.add_image(image_path=path, description=desc,
                    instruction=_DOCUMENT_INSTRUCTION)
# Video: ffmpeg extracts frames → embeds as image list (never loads full video)
store.add_video(video_path="talk.mp4", description="...", fps=0.1, max_frames=8)
# Audio: transcribe → embed transcript text
store.add_audio(audio_path="talk.wav", transcribe=True)

# 2b. FREE embedder before loading reranker (sequential model loading)
del embedder
store.embedding_provider = None
gc.collect(); torch.mps.empty_cache()

# 3. Hybrid Search + RRF Fusion (accuracy-first: wide retrieval)
dense_results = store.search(query, top_k=50, instruction=_QUERY_INSTRUCTION)
keyword_results = keyword_search(store, query, top_k=50)
fused = ReciprocalRankFusion(k=60).fuse(
    [dense_results, keyword_results], weights=[4.0, 1.0]
)

# 4. Qwen3-VL Reranking (30 candidates → 10 final)
reranker = create_qwen3vl_reranker(device="mps")
reranked = reranker.rerank(query={"text": q}, documents=fused[:30])
final = reranked[:10]

# 5. Knowledge Graph Extraction via RLM
rlm = VLRAGGraphRLM(provider="<provider>", model="<model>")
kg = rlm.completion(_KG_EXTRACTION_PROMPT, doc_context)

# 6. Query Answering via _run_vl_rag_query (single source of truth)
result = _run_vl_rag_query(query, store=store, reranker_vl=reranker, ...)
```

## CLI Reference

```bash
# Core usage
vrlmrag --provider <name> <path>       # Run full pipeline
vrlmrag --provider <name> <path> -q "Query" -o report.md
vrlmrag --provider <name> <path> --model <model> --max-depth 5

# Auto mode (uses provider hierarchy — no --provider needed)
vrlmrag <path>                         # Tries providers in PROVIDER_HIERARCHY order
vrlmrag <path> -q "Summarize"          # Auto mode with custom query

# Hierarchy management
vrlmrag --show-hierarchy               # Show fallback order + availability
vrlmrag --list-providers               # Show providers + API key status

# Utility
vrlmrag --version                      # Print version
vrlmrag --help                         # Full usage

# Interactive mode (load VL models once, query continuously)
vrlmrag --interactive <path>           # Load docs, then REPL
vrlmrag -i ./codebase                  # Load folder interactively
vrlmrag -i                             # Start empty, /add docs later

# Collections (named persistent knowledge stores)
vrlmrag -c research --add ./papers/    # Add docs to collection
vrlmrag -c research -q 'Key findings?' # Query a collection
vrlmrag -c research -c code -q 'How?'  # Blend multiple collections
vrlmrag -c research -i                 # Interactive w/ collection
vrlmrag --collection-list              # List all collections
vrlmrag -c research --collection-info  # Show collection details
vrlmrag -c research --collection-delete # Delete a collection

# Backward-compatible aliases
vrlmrag --samba-nova <path>             # Same as --provider sambanova
vrlmrag --nebius <path>                 # Same as --provider nebius
```

### CLI Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--provider NAME` | `-p` | LLM provider (default: `auto` — uses hierarchy) |
| `PATH` | | File or folder to process |
| `--query QUERY` | `-q` | Custom query (default: auto-generated) |
| `--output PATH` | `-o` | Output markdown report path |
| `--model MODEL` | `-m` | Override default model |
| `--max-depth N` | | RLM recursion depth (default: 3) |
| `--max-iterations N` | | RLM iterations per call (default: 10) |
| `--text-only` | | Use text-only embeddings (~1.2 GB RAM, skips images/videos) |
| `--use-api` | | Use API-based embeddings (~200 MB RAM, requires internet) |
| `--local` | | Use local Qwen3-VL models (requires explicit opt-in) |
| `--offline` | | Force offline mode (blocks video/audio for safety) |
| `--interactive` | `-i` | Interactive session (load VL once, query continuously) |
| `--store-dir DIR` | | Persistence directory for embeddings + knowledge graph |
| `--reindex` | | Force re-embedding of all documents (model upgrade workflow) |
| `--rebuild-kg` | | Regenerate knowledge graph with current RLM (model upgrade workflow) |
| `--collection NAME` | `-c` | Named collection (repeatable: `-c A -c B` to blend) |
| `--add PATH` | | Add documents at PATH to the specified collection(s) |
| `--collection-list` | | List all available collections |
| `--collection-info` | | Show detailed info for the specified collection |
| `--collection-delete` | | Delete the specified collection and all its data |
| `--collection-description TEXT` | | Description for a new collection (used with `--add`) |
| `--show-hierarchy` | | Show provider fallback order + availability |
| `--list-providers` | | Show all providers + key status |
| `--version` | `-V` | Print version |

## Environment Variables

```bash
# Provider hierarchy (auto mode fallback order)
PROVIDER_HIERARCHY=sambanova,nebius,groq,cerebras,zai,zenmux,openrouter,...

# Provider API keys
{PROVIDER}_API_KEY=...
{PROVIDER}_MODEL=...              # Optional: override default model
{PROVIDER}_RECURSIVE_MODEL=...    # Optional: cheaper model for recursive calls
{PROVIDER}_FALLBACK_MODEL=...     # Optional: override fallback model for auto-retry

# Provider-specific behavior
ZAI_CODING_PLAN=true              # Try Coding Plan first (default: true)
NEBIUS_CONTEXT_WINDOW=128000      # Context window in tokens

# Embedding Mode Toggle (mutually exclusive — first match wins)
VRLMRAG_TEXT_ONLY=false           # true = text-only embeddings (~1.2 GB RAM)
VRLMRAG_USE_API=false             # true = API embeddings (~200 MB RAM, requires internet)
                                  # both false = multimodal embeddings (~4.6 GB RAM)

# Model Configuration (all overridable via env vars)
VRLMRAG_TEXT_ONLY_MODEL=Qwen/Qwen3-Embedding-0.6B     # Text-only embedding model
VRLMRAG_LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B  # Multimodal embedding model
VRLMRAG_RERANKER_MODEL=ms-marco-MiniLM-L-12-v2       # FlashRank reranker model
VRLMRAG_EMBEDDING_MODEL=openai/text-embedding-3-small  # API embedding model (OpenRouter)
VRLMRAG_VLM_MODEL=inclusionai/ming-flash-omni-preview # API VLM model (ZenMux omni)

# API Embedding Provider (when VRLMRAG_USE_API=true)
VRLMRAG_EMBEDDING_API_KEY=...     # Optional: override OPENROUTER_API_KEY
VRLMRAG_EMBEDDING_BASE_URL=https://openrouter.ai/api/v1

# API VLM Provider (when VRLMRAG_USE_API=true)
VRLMRAG_VLM_API_KEY=...           # Optional: override ZENMUX_API_KEY
VRLMRAG_VLM_BASE_URL=https://zenmux.ai/api/v1

# HuggingFace token for downloading models (optional)
HF_TOKEN=...
```

### Provider Hierarchy

When `--provider` is omitted or set to `auto`, the system tries providers in
`PROVIDER_HIERARCHY` order, skipping any without API keys. If a provider fails
(rate limit, auth error, network), it automatically falls through to the next.

Default order: `sambanova → nebius → groq → cerebras → zai → zenmux → openrouter → gemini → deepseek → openai → ...`

This enables deploying the same codebase across environments — whichever
providers have keys configured will be used automatically.

### Universal Model Fallback

Every provider has a two-tier resilience strategy:

1. **Model fallback** (within same provider): If the primary model fails for
   any reason (rate limit, token limit, downtime, network error), the client
   automatically retries with a fallback model on the same provider. Defined in
   `OpenAICompatibleClient.FALLBACK_MODELS`, overridable per-provider via
   `{PROVIDER}_FALLBACK_MODEL` env var.

2. **Provider fallback** (hierarchy): If both models on a provider fail, the
   error propagates up and the hierarchy tries the next provider.

This means a single API call has up to `2 × N` chances to succeed (2 models
per provider × N providers with keys). The z.ai provider adds a third tier
(endpoint fallback: Coding Plan → Normal) before model fallback kicks in.

### Interactive Mode

The `--interactive` / `-i` flag starts a persistent session that loads VL
models once and keeps them resident in memory for continuous querying:

```
┌─────────────────────────────────────────────────────┐
│  Startup (once)                                     │
│  ├── Load Qwen3-VL Embedding (2B params)            │
│  ├── Load Qwen3-VL Reranker (2B params)             │
│  ├── Initialize RLM + provider client               │
│  ├── Load persisted embeddings (.vrlmrag_store/)    │
│  └── Load persisted knowledge graph                 │
├─────────────────────────────────────────────────────┤
│  REPL Loop                                          │
│  ├── /add <path> → ingest + embed + extend KG       │
│  ├── <query>     → search + rerank + RLM answer     │
│  ├── /kg         → inspect knowledge graph           │
│  ├── /stats      → session metrics                   │
│  └── /save       → export report                     │
└─────────────────────────────────────────────────────┘
```

**Persistence:** Embeddings are saved to `embeddings.json` and the knowledge
graph to `knowledge_graph.md` inside `.vrlmrag_store/` (or `--store-dir`).
Restarting a session in the same directory reloads previously embedded
documents and the accumulated knowledge graph without re-running the VL model.

**Incremental growth:** Each `/add` extends the vector store and merges new
entities/relationships into the knowledge graph. Queries automatically use
the full accumulated context (KG + retrieved sources).

### Universal Persistent Embeddings

Persistence is **not limited to interactive mode** — the default `run_analysis()`
path also persists and reuses embeddings and the knowledge graph:

```
.vrlmrag_store/
├── embeddings.json       # Qwen3-VL embeddings (text + images)
└── knowledge_graph.md    # Accumulated KG across all runs
```

**Content-based deduplication:** Every `add_text()` and `add_image()` call in
`MultimodalVectorStore` hashes the content (SHA-256) and skips re-embedding if
the content already exists in the store. This means:

- Re-running `vrlmrag ./folder` only embeds new/changed files
- Any provider/model combo reuses the same persistent store
- The knowledge graph merges across runs (append, never overwrite)
- KG context is prepended to every query for richer answers

This works identically whether you use `--provider sambanova`, `--provider auto`,
or `--interactive` — the `.vrlmrag_store/` directory is the single source of
truth for all accumulated embeddings and knowledge.

### Named Persistent Collections

Collections are **named, location-independent knowledge stores** that live
inside the codebase at `collections/<name>/`.  Unlike `.vrlmrag_store/`
(which is tied to a specific input path), collections can be populated from
any path and queried from anywhere.

```
<project_root>/collections/
├── research/
│   ├── collection.json       # metadata (name, description, sources)
│   ├── embeddings.json       # Qwen3-VL embeddings
│   └── knowledge_graph.md    # accumulated KG
├── codebase-docs/
│   ├── collection.json
│   ├── embeddings.json
│   └── knowledge_graph.md
└── .gitignore                # collection data is local-only
```

**Key properties:**

- **Scriptable:** `vrlmrag -c research -q "..."` runs the full 6-pillar
  pipeline and exits — no interaction needed.  Output can be piped or saved
  with `-o report.md`.
- **Blendable:** `-c research -c codebase-docs -q "..."` merges the vector
  stores and knowledge graphs from both collections into a single query.
- **Incremental:** `--add` embeds only new content (SHA-256 dedup) and
  merges the KG.  Sources are tracked in `collection.json`.
- **Interactive-compatible:** `-c research -i` starts an interactive session
  backed by the collection's store directory.
- **Provider-agnostic:** Embeddings are local Qwen3-VL; any LLM provider
  can query any collection.

#### Collection Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  vrlmrag -c research --add ./papers/                                │
│                                                                      │
│  1. create_collection("research")  → collections/research/           │
│  2. DocumentProcessor.process_path("./papers/")                      │
│  3. Qwen3-VL embed chunks  → collections/research/embeddings.json   │
│     (SHA-256 dedup: skip already-embedded content)                   │
│  4. RLM extract KG          → collections/research/knowledge_graph.md│
│     (merge_kg: append, never overwrite)                              │
│  5. record_source("research", "./papers/", doc_count, chunk_count)  │
│     → collections/research/collection.json                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  vrlmrag -c research -c code -q "How does the code implement it?"   │
│                                                                      │
│  1. Load research/embeddings.json  → blended_store                  │
│  2. Load code/embeddings.json      → merge into blended_store       │
│     (doc_id dedup: skip duplicates across collections)              │
│  3. Load research/knowledge_graph.md  → blended_kg                  │
│  4. Load code/knowledge_graph.md      → merge into blended_kg      │
│  5. _run_vl_rag_query(query, store=blended_store, kg=blended_kg)   │
│     → full 6-pillar pipeline on the merged data                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Blending Mechanics

When multiple collections are specified (`-c A -c B`), the system:

1. **Merges vector stores:** Documents from each collection's `embeddings.json`
   are loaded into a single `MultimodalVectorStore`.  Document IDs are unique
   (UUID-based), so duplicates across collections are naturally handled.
2. **Merges knowledge graphs:** Each collection's `knowledge_graph.md` is
   concatenated with `---` separators via `merge_kg()`.  The blended KG is
   prepended to every query's context (up to 8000 chars).
3. **Runs the unified pipeline:** The blended store and KG are passed to
   `_run_vl_rag_query()` — the same function used by `run_analysis()` and
   interactive mode.  No special-casing; collections are just another source
   of embeddings and knowledge.

#### `collection.json` Schema

```json
{
  "name": "research",
  "display_name": "Research",
  "description": "Academic papers on multimodal retrieval",
  "created": "2026-02-08T10:30:00+00:00",
  "updated": "2026-02-08T11:45:00+00:00",
  "sources": [
    {
      "path": "/Users/user/papers/",
      "added": "2026-02-08T10:30:00+00:00",
      "documents": 5,
      "chunks": 42
    },
    {
      "path": "/Users/user/notes/meeting.md",
      "added": "2026-02-08T11:45:00+00:00",
      "documents": 1,
      "chunks": 8
    }
  ],
  "document_count": 6,
  "chunk_count": 50
}
```

#### Collection Manager API (`collections.py`)

| Function | Description |
|----------|-------------|
| `create_collection(name, description)` | Create a new collection (or return existing metadata) |
| `load_collection_meta(name)` | Load `collection.json` metadata |
| `save_collection_meta(name, meta)` | Persist updated metadata (auto-updates `updated` timestamp) |
| `list_collections()` | Return metadata for all collections on disk |
| `delete_collection(name)` | Delete a collection and all its data |
| `collection_exists(name)` | Check if a collection exists |
| `record_source(name, path, doc_count, chunk_count)` | Record a source addition in metadata |
| `load_kg(name)` | Load the knowledge graph for a collection |
| `save_kg(name, kg_text)` | Persist the knowledge graph |
| `merge_kg(existing, new_fragment)` | Merge a new KG fragment into an existing KG |

#### CLI Operation Functions (`vrlmrag.py`)

| Function | CLI Trigger | Description |
|----------|-------------|-------------|
| `run_collection_add()` | `-c <name> --add <path>` | Embed docs, extract KG, persist to collection |
| `run_collection_query()` | `-c <name> -q "..."` | Load/blend collections, run full pipeline, print/save results |
| `show_collection_list()` | `--collection-list` | Print table of all collections with counts |
| `show_collection_info()` | `-c <name> --collection-info` | Print detailed metadata, sources, embedding/KG stats |

**Implementation:** `src/vl_rag_graph_rlm/collections.py` provides the CRUD
layer.  `run_collection_add()` and `run_collection_query()` in `vrlmrag.py`
handle the CLI operations, both routing queries through the shared
`_run_vl_rag_query()` pipeline.

## MCP Server

The codebase can also run as an **MCP (Model Context Protocol) server**,
exposing the full VL-RAG-Graph-RLM pipeline as tools that any MCP-compatible
client (Windsurf, Claude Desktop, etc.) can invoke.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  MCP Client (Windsurf / Claude Desktop / etc.)                      │
│  Connects via stdio transport                                       │
└──────────────┬──────────────────────────────────────────────────────┘
               │  JSON-RPC (MCP protocol)
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VL-RAG-Graph-RLM MCP Server                                        │
│  src/vl_rag_graph_rlm/mcp_server/server.py                         │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Settings Loader (.vrlmrag/mcp_settings.json)            │       │
│  │  → provider, model, template, max_depth, etc.            │       │
│  │  → Default: provider="auto" (hierarchy system)           │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  5-10 MCP Tools (conditional)                            │       │
│  │  ├─ Core tools (always available):                        │       │
│  │  │  ├── query_document      (full VL-RAG pipeline)       │       │
│  │  │  ├── analyze_document    (6-pillar analysis)            │       │
│  │  │  ├── list_providers      (show providers + keys)       │       │
│  │  │  ├── show_hierarchy      (provider fallback order)     │       │
│  │  │  └── show_settings       (current MCP config)         │       │
│  │  │                                                         │       │
│  │  └─ Collection tools (if VRLMRAG_COLLECTIONS=true):       │       │
│  │     ├── collection_add      (ingest docs to collection)  │       │
│  │     ├── collection_query    (query/blend collections)    │       │
│  │     ├── collection_list     (list all collections)         │       │
│  │     ├── collection_info     (collection metadata)          │       │
│  │     └── collection_delete   (remove a collection)          │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
│  All tools default to provider="auto" (hierarchy system)            │
│  Per-call overrides: provider=, model= on each tool                 │
│  Global overrides: .vrlmrag/mcp_settings.json                       │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼  Delegates to existing pipeline
┌─────────────────────────────────────────────────────────────────────┐
│  Existing VL-RAG-Graph-RLM Pipeline                                  │
│  (DocumentProcessor, _run_vl_rag_query, run_analysis, collections)  │
└─────────────────────────────────────────────────────────────────────┘
```

### Source Files

| Component | File | Description |
|-----------|------|-------------|
| Server | `src/vl_rag_graph_rlm/mcp_server/server.py` | FastMCP server, tool definitions, entry point |
| Settings | `src/vl_rag_graph_rlm/mcp_server/settings.py` | Settings loader, template resolution |
| Init | `src/vl_rag_graph_rlm/mcp_server/__init__.py` | Package init, re-exports `mcp` and `main` |
| Main | `src/vl_rag_graph_rlm/mcp_server/__main__.py` | `python -m` entry point |
| Config | `.vrlmrag/mcp_settings.json` | Default settings file (provider=auto) |

### Installation

```bash
# Install via uv (recommended)
uv pip install vl-rag-graph-rlm

# Install via pip
pip install vl-rag-graph-rlm

# Install with Qwen3-VL vision models
uv pip install "vl-rag-graph-rlm[qwen3vl]"
```

### Running the MCP Server

```bash
# Via uvx (no install needed — ephemeral environment)
uvx --from vl-rag-graph-rlm vrlmrag-mcp

# Via installed entry point
vrlmrag-mcp

# Via python -m
python -m vl_rag_graph_rlm.mcp_server
```

### MCP Client Configuration

**Streamlined Server:** 3 core tools, comprehensive defaults, minimal configuration.

### Quick Start

Add to your MCP client configuration (e.g., `~/.config/windsurf/mcp_config.json`):

```json
{
    "mcpServers": {
        "vrlmrag": {
            "command": "/Users/nbiish/.local/bin/uv",
            "args": [
                "run",
                "--project",
                "/Volumes/1tb-sandisk/code-external/mai-vl-rag-graph-rlm",
                "python",
                "-m",
                "vl_rag_graph_rlm.mcp_server"
            ],
            "env": {
                "VRLMRAG_ROOT": "/Volumes/1tb-sandisk/code-external/mai-vl-rag-graph-rlm"
            }
        }
    }
}
```

### Environment Resolution

The server finds the codebase root (and its `.env`) in this order:

1. **`VRLMRAG_ROOT` env var** — set in MCP client config; always works, even via `uv`
2. **`__file__`-based walk** — works for editable / local `pip install -e .` installs
3. **CWD fallback** — if the server is started from the repo directory

This means users configure their API keys in the repo's `.env` file once and
the MCP server picks them up everywhere — no need to pass secrets through
the MCP client's `env` block.

### Settings (`mcp_settings.json`)

The server loads settings from (in priority order):
1. `$VRLMRAG_MCP_SETTINGS` env var → path to a JSON file
2. `<project_root>/.vrlmrag/mcp_settings.json`
3. Built-in defaults

```json
{
    "provider": "auto",
    "model": null,
    "template": null,
    "max_depth": 3,
    "max_iterations": 10,
    "temperature": 0.0,
    "collections_enabled": true,
    "collections_root": null,
    "log_level": "INFO"
}
```

**Key behavior:**
- `provider: "auto"` (default) → uses the hierarchy system, same as CLI
- `template` → shorthand for provider+model combos (e.g., `"fast-free"`, `"nebius-m2"`)
- `collections_enabled` → controls whether collection tools are exposed to the LLM
- Per-call `provider`/`model` overrides on each tool beat settings.json
- Settings.json beats hierarchy defaults

### Environment Variable Configuration

All MCP settings can be overridden per-client via `VRLMRAG_*` env vars in the MCP client's `env` block. This enables different configurations for Windsurf, Claude Desktop, Cursor, etc. without touching the codebase.

| Env Var | Values | Default | Description |
|---------|--------|---------|-------------|
| `VRLMRAG_ROOT` | Path | auto-detect | Path to cloned repo (loads .env from there) |
| `VRLMRAG_PROVIDER` | `auto`, `sambanova`, `nebius`, ... | `auto` | Provider for MCP tools |
| `VRLMRAG_MODEL` | Model name | `null` | Explicit model override |
| `VRLMRAG_TEMPLATE` | Template name | `null` | Shorthand for provider+model |
| `VRLMRAG_MAX_DEPTH` | Integer | `3` | Max RLM recursion depth |
| `VRLMRAG_MAX_ITERATIONS` | Integer | `10` | Max RLM iterations per call |
| `VRLMRAG_TEMPERATURE` | Float | `0.0` | LLM temperature |
| `VRLMRAG_COLLECTIONS` | `true` / `false` | `true` | **Enable/disable collection tools** |
| `VRLMRAG_COLLECTIONS_ROOT` | Path | `null` | Override collections directory |
| `VRLMRAG_LOG_LEVEL` | `DEBUG`, `INFO`, ... | `INFO` | Logging level |

**Priority:** `VRLMRAG_*` env vars > settings file > built-in defaults

### Disabling Collection Tools

To reduce token context for the LLM when collections aren't needed:

```json
{
    "mcpServers": {
        "vrlmrag": {
            "command": "uv",
            "args": ["run", "--project", "/path/to/repo", "python", "-m", "vl_rag_graph_rlm.mcp_server"],
            "env": {
                "VRLMRAG_ROOT": "/path/to/repo",
                "VRLMRAG_COLLECTIONS": "false"
            }
        }
    }
}
```

With `VRLMRAG_COLLECTIONS=false`, the server exposes only 2 core tools instead of 3. The setting takes effect at server startup — restart the MCP server after changing it.

### Available Templates

| Template | Provider | Model |
|----------|----------|-------|
| `fast-free` | sambanova | DeepSeek-V3.2 |
| `fast-groq` | groq | moonshotai/kimi-k2-instruct-0905 |
| `fast-cerebras` | cerebras | zai-glm-4.7 |
| `nebius-m2` | nebius | MiniMaxAI/MiniMax-M2.1 |
| `openrouter-cheap` | openrouter | minimax/minimax-m2.1 |
| `openai-mini` | openai | gpt-4o-mini |
| `anthropic-haiku` | anthropic | claude-3-5-haiku-20241022 |
| `gemini-flash` | gemini | gemini-1.5-flash |
| `deepseek-chat` | deepseek | deepseek-chat |

### MCP Tools Reference (Streamlined — 3 Tools)

The streamlined MCP server consolidates functionality into 3 unified tools:

| Tool | Description | Key Parameters | Availability |
|------|-------------|----------------|--------------|
| `analyze` | **Comprehensive document analysis** — Full VL-RAG-Graph-RLM pipeline | `input_path`, `query?`, `mode?`, `provider?`, `model?`, `output_path?` | Always |
| `query_collection` | **Query knowledge collections** — Persistent stores with blending | `collection`, `query`, `mode?`, `provider?`, `model?` | Always |
| `collection_manage` | **Collection management** — All operations via `action` parameter | `action`, `collection?`, `path?`, `query?`, `tags?`, `target_collection?` | If collections enabled |

**Tool Details:**

- **`analyze`** — Comprehensive document analysis by default. Use `mode` to adjust:
  - `mode="comprehensive"` (default) — Full vision, audio, RAG, graph, recursive LLM
  - `mode="fast"` — Quick search when speed is prioritized over depth

- **`query_collection`** — Query persistent knowledge collections. Collections are created automatically on first query if they don't exist.

- **`collection_manage`** — All collection operations unified:
  - `action="add"` — Add documents at `path` to `collection`
  - `action="list"` — List all collections
  - `action="info"` — Show collection details
  - `action="delete"` — Remove collection
  - `action="export"` — Export to tar.gz
  - `action="import"` — Import from tar.gz
  - `action="merge"` — Merge collections
  - `action="tag"` — Add tags to collection
  - `action="search"` — Search collections by query/tags

All tools default to comprehensive analysis with API provider hierarchy.

## Environment Variables
