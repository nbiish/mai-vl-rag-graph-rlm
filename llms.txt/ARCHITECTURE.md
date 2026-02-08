# Architecture — VL-RAG-Graph-RLM

> **Documentation Index:** See [README.md](README.md) for all documentation files.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Document Intake                              │
│  PPTX / PDF / TXT / MD → text chunks + extracted images             │
└──────────────┬──────────────────────────────────┬───────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────┐    ┌──────────────────────────────────┐
│  Qwen3-VL Embedding      │    │  Qwen3-VL Embedding              │
│  (Text Chunks)            │    │  (Extracted Images)              │
│  Qwen/Qwen3-VL-Embed-2B  │    │  Qwen/Qwen3-VL-Embed-2B         │
└──────────┬───────────────┘    └──────────┬───────────────────────┘
           │                               │
           ▼                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MultimodalVectorStore (Unified Vector Space)            │
│  Text embeddings + Image embeddings stored together                 │
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

## Accuracy-First Query Pipeline

Every query — in both `run_analysis()` and interactive mode — goes through
a single shared function `_run_vl_rag_query()` that guarantees the full
6-pillar pipeline executes on every call:

| Stage | Component | Parameters |
|-------|-----------|------------|
| 1. Dense search | Qwen3-VL embedding (cosine sim) | `top_k=50`, instruction: `"Find passages that are relevant to and answer the following query."` |
| 2. Keyword search | Token-overlap scoring | `top_k=50` |
| 3. Fusion | Reciprocal Rank Fusion | `k=60`, weights `[4.0, 1.0]` (dense-heavy) |
| 4. Reranking | Qwen3-VL cross-attention reranker | `30` candidates → `10` final results |
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
| Pipeline | `src/vl_rag_graph_rlm/pipeline.py` | `MultimodalRAGPipeline` unified API |
| Vision | `src/vl_rag_graph_rlm/vision.py` | Image encoding, multimodal message formatting |
| Clients | `src/vl_rag_graph_rlm/clients/` | Provider clients (OpenAI-compatible, Anthropic, Gemini, LiteLLM) |
| RAG Init | `src/vl_rag_graph_rlm/rag/__init__.py` | `SearchResult`, `RRF`, `MultiFactorReranker`, `HybridSearcher` |
| RAG Provider | `src/vl_rag_graph_rlm/rag/provider.py` | RAG provider interface |
| RAG Store | `src/vl_rag_graph_rlm/rag/store.py` | Vector store base module |
| RAG Reranker | `src/vl_rag_graph_rlm/rag/reranker.py` | Reranker implementations |
| Qwen3-VL | `src/vl_rag_graph_rlm/rag/qwen3vl.py` | `Qwen3VLEmbeddingProvider`, `Qwen3VLRerankerProvider` |
| ERNIE Client | `src/vl_rag_graph_rlm/rag/ernie_client.py` | Baidu ERNIE / OpenAI-compatible client |
| Vector Store | `src/vl_rag_graph_rlm/rag/multimodal_store.py` | `MultimodalVectorStore` with text + image storage |
| Collections | `src/vl_rag_graph_rlm/collections.py` | Named persistent knowledge stores (CRUD, KG helpers) |
| Environments REPL | `src/vl_rag_graph_rlm/environments/repl.py` | Alternative safe Python execution sandbox |
| CLI | `src/vrlmrag.py` | Unified `--provider <name>` CLI for all 17 providers |
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
```

## Full Pipeline Flow (Template Pattern)

Every provider template follows this pattern:

```python
# 1. Document Processing
processor = DocumentProcessor()
documents = processor.process_path(input_path)

# 2. Qwen3-VL Embedding (text + images → unified vector space)
embedder = create_qwen3vl_embedder(device="mps")
store = MultimodalVectorStore(embedding_provider=embedder, storage_path=...)
for chunk in chunks:
    store.add_text(content=chunk["content"], metadata={...},
                   instruction=_DOCUMENT_INSTRUCTION)
for image in images:
    store.add_image(image_path=path, description=desc,
                    instruction=_DOCUMENT_INSTRUCTION)

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
| `--interactive` | `-i` | Interactive session (load VL once, query continuously) |
| `--store-dir DIR` | | Persistence directory for embeddings + knowledge graph |
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

# Embedding models (auto-downloaded from HuggingFace)
HF_TOKEN=...                      # Optional: for gated models
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
