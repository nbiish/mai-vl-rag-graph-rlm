<div align="center">
  <hr width="50%">
  <h3>Support This Project</h3>
  <table style="border: none; border-collapse: collapse;">
    <tr style="border: none;">
      <td align="center" style="border: none; vertical-align: middle; padding: 20px;">
        <h4>Stripe</h4>
        <img src="qr-stripe-donation.png" alt="Scan to donate" width="180"/>
        <p><a href="https://raw.githubusercontent.com/nbiish/license-for-all-works/8e9b73b269add9161dc04bbdd79f818c40fca14e/qr-stripe-donation.png">Donate via Stripe</a></p>
      </td>
      <td align="center" style="border: none; vertical-align: middle; padding: 20px;">
        <a href="https://www.buymeacoffee.com/nbiish">
          <img src="buy-me-a-coffee.svg" alt="Buy me a coffee" />
        </a>
      </td>
    </tr>
  </table>
  <hr width="50%">
</div>

```bibtex
@misc{mai-vl-rag-graph-rlm2026,
  author/creator/steward = {ᓂᐲᔥ ᐙᐸᓂᒥᑮ-ᑭᓇᐙᐸᑭᓯ (Nbiish Waabanimikii-Kinawaabakizi), also known legally as JUSTIN PAUL KENWABIKISE, professionally documented as Nbiish-Justin Paul Kenwabikise, Anishinaabek Dodem (Anishinaabe Clan): Animikii (Thunder), descendant of Chief ᑭᓇᐙᐸᑭᓯ (Kinwaabakizi) of the Beaver Island Band and enrolled member of the sovereign Grand Traverse Band of Ottawa and Chippewa Indians},
  title/description = {mai-vl-rag-graph-rlm},
  type_of_work = {Indigenous digital creation/software incorporating traditional knowledge and cultural expressions},
  year = {2026},
  publisher/source/event = {GitHub repository under tribal sovereignty protections},
  howpublished = {\url{https://github.com/nbiish/mai-vl-rag-graph-rlm}},
  note = {Authored and stewarded by ᓂᐲᔥ ᐙᐸᓂᒥᑮ-ᑭᓇᐙᐸᑭᓯ (Nbiish Waabanimikii-Kinawaabakizi), also known legally as JUSTIN PAUL KENWABIKISE, professionally documented as Nbiish-Justin Paul Kenwabikise, Anishinaabek Dodem (Anishinaabe Clan): Animikii (Thunder), descendant of Chief ᑭᓇᐙᐸᑭᓯ (Kinwaabakizi) of the Beaver Island Band and enrolled member of the sovereign Grand Traverse Band of Ottawa and Chippewa Indians. This work embodies Indigenous intellectual property, traditional knowledge systems (TK), traditional cultural expressions (TCEs), and associated data protected under tribal law, federal Indian law, treaty rights, Indigenous Data Sovereignty principles, and international indigenous rights frameworks including UNDRIP. All usage, benefit-sharing, and data governance are governed by the COMPREHENSIVE RESTRICTED USE LICENSE FOR INDIGENOUS CREATIONS WITH TRIBAL SOVEREIGNTY, DATA SOVEREIGNTY, AND WEALTH RECLAMATION PROTECTIONS.}
}
```

# VL-RAG-Graph-RLM

**Vision-Language RAG Graph Recursive Language Models** — a unified multimodal document analysis framework combining **Qwen3-VL embeddings**, **hybrid RAG with RRF fusion**, **cross-attention reranking**, **knowledge graph extraction**, and **recursive LLM reasoning** across **17 LLM provider templates** with automatic fallback. Supports **text, images, video, and audio** with memory-safe sequential model loading (peak ~6.7 GB). Features **named persistent collections**, **MCP server integration**, **accuracy-first retrieval**, and **universal persistent embeddings** with SHA-256 deduplication.

## The Six Pillars

| # | Pillar | Component | Cost | Modes |
|---|--------|-----------|------|-------|
| 1 | **VL** | Qwen3-VL-Embedding-2B (multimodal) or Qwen3-Embedding-0.6B (text-only) | FREE (local) | 3 modes: text-only (~1.2 GB), API (~200 MB), multimodal (~4.6 GB) |
| 2 | **RAG** | Hybrid search (dense cosine + keyword) with RRF fusion | FREE (local) | All modes |
| 3 | **Reranker** | FlashRank ONNX cross-encoder (~34 MB) | FREE (local) | All modes |
| 4 | **Graph** | Knowledge graph extraction via RLM | LLM cost | All modes |
| 5 | **RLM** | Recursive Language Model with sandboxed REPL | LLM cost | All modes |
| 6 | **Report** | Markdown report with sources, scores, and metadata | FREE | All modes |

## Installation

```bash
# Clone
git clone https://github.com/nbiish/mai-vl-rag-graph-rlm.git
cd mai-vl-rag-graph-rlm

# Install core
uv pip install -e .

# Install with Qwen3-VL multimodal support
uv pip install -e ".[qwen3vl]"

# Install with audio transcription (NVIDIA Parakeet V3)
uv pip install -e ".[parakeet]"

# Install everything
uv pip install -e ".[all]"
```

### MCP Server Installation

Add to your MCP client (Windsurf, Claude Desktop, etc.):

```json
{
    "mcpServers": {
        "vrlmrag": {
            "command": "/Users/YOU/.local/bin/uv",
            "args": [
                "run",
                "--project",
                "/path/to/mai-vl-rag-graph-rlm",
                "python",
                "-m",
                "vl_rag_graph_rlm.mcp_server"
            ],
            "env": {
                "VRLMRAG_ROOT": "/path/to/mai-vl-rag-graph-rlm",
                "VRLMRAG_COLLECTIONS": "true"
            }
        }
    }
}
```

Set `VRLMRAG_COLLECTIONS: "false"` to hide collection tools and reduce token context.

## Quick Start

## Three Embedding Modes

| Mode | Flag | RAM | Network | Best For |
|------|------|-----|---------|----------|
| **Text-Only** | `--text-only` | ~1.2 GB | Offline | `.txt`, `.md`, text-heavy PDFs |
| **API** | `--use-api` | ~200 MB | Required | Any content, low-RAM machines |
| **Multimodal** (default) | (none) | ~4.6 GB | Offline | PowerPoints, PDFs with figures, images, video |

### CLI

```bash
# Set your API key
export SAMBANOVA_API_KEY=your_key_here

# Text-Only Mode (~1.2 GB RAM, fastest, fully offline)
vrlmrag --text-only ./docs -q "Summarize these documents"
vrlmrag --text-only paper.md -q "What are the key findings?"

# API Mode (~200 MB RAM, requires internet)
vrlmrag --use-api ./docs -q "Analyze this content"

# Multimodal Mode (default, ~4.6 GB RAM) — PowerPoints, PDFs with images
vrlmrag --provider sambanova presentation.pptx
vrlmrag --provider nebius document.pptx -q "Summarize key findings" -o report.md

# Override model and tune recursion
vrlmrag --provider openrouter ./docs --model kimi/kimi-k2.5 --max-depth 5

# List all providers and API key status
vrlmrag --list-providers

# Show version
vrlmrag --version

# Security check (before committing)
bash .ainish/scripts/scan_secrets.sh
```

### Interactive Mode

Load VL models once, then query continuously without reloading. Add more documents on the fly. Knowledge graph persists and grows across queries.

```bash
# Start interactive session with a document
vrlmrag --interactive presentation.pptx

# Start with a codebase/folder
vrlmrag -i ./docs

# Start empty (add documents later via /add)
vrlmrag -i

# With explicit provider and custom store directory
vrlmrag -i --provider sambanova ./project --store-dir ./my_store
```

Inside the session:

```
vrlmrag> What are the main topics covered?
  [answer with sources]

vrlmrag> /add ./more_docs/paper.pdf
  [processes and embeds new document, extends knowledge graph]

vrlmrag> How does the new paper relate to the presentation?
  [answer using all loaded documents + accumulated knowledge graph]

vrlmrag> /kg
  [shows the current knowledge graph]

vrlmrag> /stats
  [shows session statistics: documents, queries, timing]

vrlmrag> /save report.md
  [saves session report to file]

vrlmrag> /quit
```

### Persistent Embeddings & Knowledge Graph

**All modes** (default, interactive, any provider/model combo) automatically persist embeddings and the knowledge graph to `.vrlmrag_store/` next to the input path. Re-running on the same folder or file:

- **Skips already-embedded chunks** — content-based SHA-256 deduplication means only new/changed content gets re-embedded
- **Merges the knowledge graph** — new entities and relationships are appended, not overwritten
- **Uses KG context in every query** — the accumulated knowledge graph is prepended to retrieval context for richer answers

```bash
# First run: embeds everything, builds KG
vrlmrag ./my-project -q "Summarize the architecture"
#   New embeddings: 42 | Skipped: 0 | Total in store: 42

# Second run (same folder): skips existing, only embeds new files
vrlmrag ./my-project -q "What changed since last time?"
#   New embeddings: 3 | Skipped: 42 | Total in store: 45

# Works with any provider — same persistent store
vrlmrag --provider nebius ./my-project -q "Explain the auth flow"
vrlmrag --provider sambanova ./my-project -q "Compare auth approaches"
```

The `.vrlmrag_store/` directory contains:
- `embeddings.json` — persisted Qwen3-VL embeddings (text + images)
- `knowledge_graph.md` — accumulated knowledge graph across all runs

### Collections (Named Persistent Knowledge Stores)

Collections let you build named, persistent knowledge bases that can be queried from anywhere — no matter what directory you're in. Documents are embedded with Qwen3-VL, knowledge graphs are extracted and merged, and everything is stored inside this codebase at `collections/<name>/`.

```bash
# Create a collection and add documents
vrlmrag -c research --add ./papers/
vrlmrag -c research --add ./notes/meeting.md

# Query a collection (fully scriptable — no interaction needed)
vrlmrag -c research -q "What are the key findings across all papers?"

# Blend multiple collections in a single query
vrlmrag -c research -c codebase -q "How does the code implement the paper's algorithm?"

# Interactive session backed by a collection
vrlmrag -c research -i

# List all collections
vrlmrag --collection-list

# Show detailed collection info
vrlmrag -c research --collection-info

# Delete a collection
vrlmrag -c research --collection-delete
```

Collections are stored at `collections/<name>/` inside the project root:
- `collection.json` — metadata (name, description, sources, counts)
- `embeddings.json` — Qwen3-VL embeddings (text + images)
- `knowledge_graph.md` — accumulated KG across all additions

Every query against a collection goes through the full 6-pillar pipeline: Qwen3-VL dense search → keyword search → RRF fusion → Qwen3-VL reranking → KG augmentation → RLM recursive completion.

### Python API — High-Level Pipeline

```python
from vl_rag_graph_rlm import create_pipeline

# Pipeline init is instant (~207 MB) — models load lazily on first use
pipeline = create_pipeline(
    llm_provider="sambanova",
    embedding_model="Qwen/Qwen3-VL-Embedding-2B",
    use_reranker=True,
)

# Documents (embedder loads on first add)
pipeline.add_pptx("presentation.pptx", extract_images=True)

# Video: ffmpeg extracts frames (RAM-safe, never loads full video)
pipeline.add_video("talk.mp4", description="Conference talk", fps=0.1, max_frames=8)

# Audio: transcribe with Parakeet V3 → embed transcript text
# Requires: pip install "vl-rag-graph-rlm[parakeet]"
pipeline.add_audio("recording.wav", transcribe=True)

# Query (reranker loads on first query, embedder freed automatically)
result = pipeline.query("What are the main topics covered?")
print(result.answer)
```

### Python API — Manual Pipeline

```python
from vl_rag_graph_rlm import VLRAGGraphRLM
from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder, create_qwen3vl_reranker
from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore
from vl_rag_graph_rlm.rag import ReciprocalRankFusion

# 1. VL — Embed text + images
embedder = create_qwen3vl_embedder(device="mps")  # or "cuda", "cpu"
store = MultimodalVectorStore(embedding_provider=embedder)
store.add_text("Machine learning uses statistical methods...", metadata={"page": 1})
store.add_image("diagram.png", description="Architecture diagram")

# 2. RAG — Hybrid search
dense_results = store.search("How does ML work?", top_k=20)

# 3. Reranker — Cross-attention reranking
reranker = create_qwen3vl_reranker(device="mps")
docs = [{"text": store.get(r.id).content} for r in dense_results if store.get(r.id)]
reranked = reranker.rerank(query={"text": "How does ML work?"}, documents=docs)

# 4. Graph — Knowledge graph extraction
rlm = VLRAGGraphRLM(provider="sambanova", temperature=0.0)
kg = rlm.completion("Extract entities and relationships.", context)

# 5. RLM — Recursive query answering
result = rlm.completion("How does ML work?", context)
print(result.response)
```

## Supported Providers

> Models verified via live API queries on **Feb 7, 2026**.

| Provider | Default Model | Context | Token Limits | Notes |
|----------|--------------|---------|--------------|-------|
| `sambanova` | DeepSeek-V3.2 | 128K | **200K TPD** ⚠️ | Also: V3.1, gpt-oss-120b, Qwen3-235B, Llama-4-Maverick |
| `nebius` | MiniMaxAI/MiniMax-M2.1 | 128K | Unlimited | Also: GLM-4.7-FP8, Nemotron-Ultra-253B, DeepSeek-R1 |
| `openrouter` | minimax/minimax-m2.1 | varies | Per-model | 400+ models incl. GPT-5.3, Claude Opus 4.6, Gemini 3 |
| `openai` | gpt-4o-mini | 128K | Per-tier | Also: gpt-4o, gpt-5.2, gpt-5.3-codex |
| `anthropic` | claude-3-5-haiku | 200K | Per-tier | Also: claude-sonnet-4, claude-opus-4.6 |
| `gemini` | gemini-1.5-flash | 1M | Per-tier | Also: gemini-3-pro, gemini-3-flash |
| `groq` | moonshotai/kimi-k2-instruct-0905 | 128K | Per-tier | Also: gpt-oss-120b, llama-4-maverick, qwen3-32b |
| `deepseek` | deepseek-chat | 128K | Per-tier | Also: deepseek-reasoner (R1) |
| `mistral` | mistral-large-latest | 128K | Per-tier | Also: mistral-large-3 (675B MoE) |
| `fireworks` | llama-v3p1-70b-instruct | 128K | Per-tier | Serverless open-source models |
| `together` | Meta-Llama-3.1-70B-Instruct-Turbo | 128K | Per-tier | Open-source model hosting |
| `zenmux` | moonshotai/kimi-k2.5 | varies | Per-model | 59+ models, provider/model format |
| `zai` | glm-4.7 | 128K | Coding Plan | Tries Coding Plan first, falls back to normal API |
| `azure_openai` | gpt-4o | 128K | Per-deployment | Enterprise GPT deployments |
| `cerebras` | zai-glm-4.7 | 128K | Per-tier | 355B ~1000 tok/s; also gpt-oss-120b (~3000 tok/s) |
| `openai_compatible` | user-configured | varies | Per-provider | Generic OpenAI-compatible endpoint |
| `anthropic_compatible` | user-configured | varies | Per-provider | Generic Anthropic-compatible endpoint |

> ⚠️ **Cerebras**: `llama-3.3-70b` and `qwen-3-32b` deprecated Feb 16, 2026. Use `zai-glm-4.7` or `gpt-oss-120b`.

### Token Budget Architecture

**Important:** All providers except SambaNova have generous or unlimited token limits:

- **Nebius**: No daily limits — can use full 128K context
- **OpenRouter, OpenAI, Anthropic, Gemini, Groq, etc.**: Per-tier limits that are typically generous
- **SambaNova**: **200K TPD free tier** — requires careful context budgeting

The system automatically adjusts context budgets per provider:

| Provider | Context Budget | Rationale |
|----------|---------------|-----------|
| SambaNova | 8,000 chars (~2K tokens) | Fits within 200K TPD for multiple queries |
| Nebius | 100,000 chars (~25K tokens) | No daily limits, use full context |
| Others | 32,000-64,000 chars | Balanced for typical tier limits |

To upgrade SambaNova limits, consider the Developer tier (12K RPD, no TPD limit).

## CLI Reference

```
usage: vrlmrag [-h] [--version] [--list-providers] [--show-hierarchy]
               [--provider NAME] [--query QUERY] [--output OUTPUT]
               [--model MODEL] [--max-depth N] [--max-iterations N]
               [--text-only] [--use-api] [--interactive] [--store-dir DIR]
               [--collection NAME] [--add PATH] [--collection-list]
               [--collection-info] [--collection-delete]
               [--collection-description TEXT] [PATH]
```

| Flag | Short | Description |
|------|-------|-------------|
| `--provider NAME` | `-p` | LLM provider (default: `auto` — uses hierarchy) |
| `PATH` | | File or folder to process (PPTX, PDF, TXT, MD) |
| `--query QUERY` | `-q` | Custom query (default: auto-generated) |
| `--output PATH` | `-o` | Output markdown report path |
| `--model MODEL` | `-m` | Override default model |
| `--max-depth N` | | RLM recursion depth (default: 3) |
| `--max-iterations N` | | RLM iterations per call (default: 10) |
| `--text-only` | | Text-only embeddings (~1.2 GB RAM, skips images) |
| `--use-api` | | API-based embeddings (~200 MB RAM, requires internet) |
| `--interactive` | `-i` | Interactive session (load VL once, query continuously) |
| `--store-dir DIR` | | Persistence directory for embeddings + knowledge graph |
| `--collection NAME` | `-c` | Named collection (repeatable: `-c A -c B` to blend) |
| `--add PATH` | | Add documents at PATH to the specified collection(s) |
| `--collection-list` | | List all available collections |
| `--collection-info` | | Show detailed info for the specified collection |
| `--collection-delete` | | Delete the specified collection and all its data |
| `--collection-description TEXT` | | Description for a new collection (used with `--add`) |
| `--list-providers` | | Show all providers + API key status |
| `--show-hierarchy` | | Show provider fallback order + availability |
| `--version` | `-V` | Print version |

Backward-compatible: `--samba-nova PATH`, `--nebius PATH`

### Auto Mode (Provider Hierarchy)

When `--provider` is omitted (or set to `auto`), the system tries providers in configurable order:

```bash
# Just give it a file — auto picks the best available provider
vrlmrag presentation.pptx

# Auto mode with a custom query
vrlmrag ./docs -q "Summarize key findings"

# See which providers are available and in what order
vrlmrag --show-hierarchy
```

Default hierarchy: **sambanova → nebius → groq → cerebras → zai → zenmux → openrouter → gemini → deepseek → openai → anthropic → ...**

If you configure `OPENAI_COMPATIBLE_API_KEY` or `ANTHROPIC_COMPATIBLE_API_KEY`, those custom SDK endpoints are automatically prepended as the highest-priority providers.

If a provider fails (rate limit, auth error, network issue), the system automatically falls through to the next available provider.

Customize the order in `.env`:

```bash
PROVIDER_HIERARCHY=groq,cerebras,openrouter,zai,zenmux,nebius,sambanova
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Key variables:

```bash
# Provider hierarchy (auto mode fallback order)
PROVIDER_HIERARCHY=sambanova,nebius,groq,cerebras,zai,zenmux,openrouter,...

# Per-provider: API key + optional model override
{PROVIDER}_API_KEY=your_key_here
{PROVIDER}_MODEL=model-name           # optional
{PROVIDER}_RECURSIVE_MODEL=model-name  # optional (for recursive calls)
{PROVIDER}_FALLBACK_MODEL=model-name   # optional (auto-retry on error)

# Embedding Mode Toggle (mutually exclusive — first match wins)
VRLMRAG_TEXT_ONLY=false           # true = text-only (~1.2 GB, offline)
VRLMRAG_USE_API=false             # true = API-based (~200 MB, requires internet)
                                  # both false = multimodal (~4.6 GB, offline)

# Model Configuration (all externalized to env vars)
VRLMRAG_TEXT_ONLY_MODEL=Qwen/Qwen3-Embedding-0.6B      # Text-only embedding
VRLMRAG_LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B  # Multimodal embedding
VRLMRAG_RERANKER_MODEL=ms-marco-MiniLM-L-12-v2      # FlashRank reranker
VRLMRAG_EMBEDDING_MODEL=openai/text-embedding-3-small  # API embedding (OpenRouter)
VRLMRAG_VLM_MODEL=inclusionai/ming-flash-omni-preview # API VLM (ZenMux)

# MCP Server configuration (per-client via mcp_config.json env block)
VRLMRAG_ROOT=/path/to/repo              # Required: finds .env file
VRLMRAG_PROVIDER=auto                   # Provider for MCP tools
VRLMRAG_MODEL=null                      # Model override
VRLMRAG_TEMPLATE=null                   # Template shorthand
VRLMRAG_MAX_DEPTH=3                     # RLM recursion depth
VRLMRAG_MAX_ITERATIONS=10               # RLM iterations
VRLMRAG_TEMPERATURE=0.0                 # LLM temperature
VRLMRAG_COLLECTIONS=true                # Enable/disable collection tools
VRLMRAG_COLLECTIONS_ROOT=null         # Collections directory override
VRLMRAG_LOG_LEVEL=INFO                  # Logging level
```

Special provider notes:
- **z.ai**: Set `ZAI_CODING_PLAN=true` (default) to try Coding Plan endpoint first, falling back to normal endpoint on failure
- **ZenMux**: Uses `provider/model-name` format (e.g., `moonshotai/kimi-k2.5`)
- **Model fallback** (all providers): On any error (rate limit, token limit, downtime), every provider automatically retries with a fallback model before escalating to the next provider in the hierarchy. Override via `{PROVIDER}_FALLBACK_MODEL` env var

See `.env.example` for all options.

## How It Works

### Three Operating Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Default** | `vrlmrag <path>` | Process docs → embed → query → report |
| **Interactive** | `vrlmrag -i <path>` | Load VL models once, query continuously, `/add` docs on the fly |
| **Collection** | `vrlmrag -c <name> -q "..."` | Query named persistent knowledge stores, blend multiple collections |

### Pipeline Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Documents   │ →  │  Qwen3-VL    │ →  │ Hybrid Search│ →  │  RLM         │
│  PPTX/PDF/   │    │  Embedding   │    │ + RRF Fusion │    │  Recursive   │
│  TXT/MD/IMG  │    │  (2B, local) │    │ + Reranking  │    │  Reasoning   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   ↓                    ↓                    ↓
       │            Unified Vector       Retrieved Context    Markdown Report
       │            Space (text+img)     + Knowledge Graph    + Sources
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│  Persistence │    │  Collections │
│  .vrlmrag_   │    │  collections/│
│  store/      │    │  <name>/     │
│  (path-local)│    │  (portable)  │
└──────────────┘    └──────────────┘
```

All queries — default, interactive, and collection — route through `_run_vl_rag_query()`:

| Stage | Component | Parameters |
|-------|-----------|------------|
| 1. Dense search | Qwen3-VL embedding (cosine sim) | `top_k=50`, `_QUERY_INSTRUCTION` |
| 2. Keyword search | Token-overlap scoring | `top_k=50` |
| 3. Fusion | Reciprocal Rank Fusion | `k=60`, weights `[4.0, 1.0]` |
| 4. Reranking | Qwen3-VL cross-attention | `30` candidates → `10` final |
| 5. KG augmentation | Persisted knowledge graph | Up to `8000` chars |
| 6. RLM completion | Recursive Language Model | `max_depth=3`, `max_iterations=10` |

### RLM Workflow

1. **Context Analysis** — RLM analyzes retrieved context + knowledge graph
2. **Code Generation** — Generates Python code to explore the data
3. **Safe Execution** — Runs code in a RestrictedPython sandbox
4. **Recursive Calls** — Breaks complex tasks into sub-tasks via `recursive_llm()`
5. **Answer Synthesis** — Returns final answer via `FINAL()` or `FINAL_VAR()`

## API Reference

### Client Factory

```python
from vl_rag_graph_rlm.clients import get_client, HierarchyClient

# Auto — uses provider hierarchy with automatic fallback
client = get_client('auto')
result = client.completion('Analyze this document...')
print(client.active_provider)  # shows which provider handled the call

# Start hierarchy from a specific provider
client = HierarchyClient(start_provider='groq')

# Explicit provider (no fallback)
client = get_client('zai', model_name='glm-4.7')

# ZenMux — uses provider/model format
client = get_client('zenmux', model_name='moonshotai/kimi-k2.5')

# All other providers
client = get_client('groq')
client = get_client('nebius')
client = get_client('openrouter')
# ... etc
```

### VLRAGGraphRLM

```python
VLRAGGraphRLM(
    provider: str,              # Provider name (sambanova, nebius, openrouter, ...)
    model: str = None,          # Model name (auto-detected from env/defaults)
    recursive_model: str = None,# Cheaper model for recursive sub-queries
    api_key: str = None,        # API key (or use env var)
    max_depth: int = 3,         # Maximum recursion depth
    max_iterations: int = 10,   # Max iterations per call
    temperature: float = 0.0,   # Sampling temperature
)
```

**Methods:**
- `completion(query, context)` → `CompletionResult` — Recursive completion with REPL

### MultimodalVectorStore

```python
MultimodalVectorStore(
    embedding_provider=embedder,  # Qwen3-VL embedding provider
    storage_path="store.json",    # JSON persistence path
)
```

**Methods:**
- `add_text(content, metadata)` — Embed and store text
- `add_image(image_path, description, metadata)` — Embed and store image
- `search(query, top_k)` → `List[SearchResult]` — Dense vector search
- `get(doc_id)` → `Document` — Retrieve document by ID

### Factory Functions

```python
from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder, create_qwen3vl_reranker

embedder = create_qwen3vl_embedder(device="mps")  # or "cuda", "cpu"
reranker = create_qwen3vl_reranker(device="mps")
```

## Provider Templates

Every provider has a ready-to-use template in `templates/`:

```bash
python templates/provider_sambanova.py --input document.pptx
python templates/provider_nebius.py --manual
python templates/provider_openrouter.py --input doc.pdf --query "Summarize"
```

Each template demonstrates all six pillars with both `create_pipeline()` and manual step-by-step examples.

## Troubleshooting

### Troubleshooting Provider Connectivity

| Provider | Common Issues | Solution |
|----------|--------------|----------|
| **SambaNova** | 429 rate limit | You've hit 200K TPD free tier. Wait 24h or upgrade to Developer tier |
| **z.ai** | 429 balance error | Your normal API account needs balance. Coding Plan subscription is separate |
| **ZenMux** | Connection errors | Now fixed — uses correct `https://zenmux.ai/api/v1` endpoint |
| **Groq** | 404 model not found | Now fixed — uses correct `llama-3.3-70b-versatile` model |

### Qwen3-VL won't load
```bash
pip install torch transformers>=5.1.0 qwen-vl-utils>=0.0.14 pillow torchvision
```

### Out of memory
```python
# Use CPU instead of GPU/MPS
embedder = create_qwen3vl_embedder(device="cpu")
```

## Documentation

- **[README.md](README.md)** (this file): Quick start, CLI reference, API docs, MCP server
- **[llms.txt/README.md](llms.txt/README.md)**: Documentation index, what's new, navigation
- **[llms.txt/PRD.md](llms.txt/PRD.md)**: Product requirements, architecture, CLI examples
- **[llms.txt/ARCHITECTURE.md](llms.txt/ARCHITECTURE.md)**: System diagram, component map, pipeline flow, MCP server details
- **[llms.txt/CONTRIBUTING.md](llms.txt/CONTRIBUTING.md)**: Adding providers, extending collections, testing
- **[llms.txt/TODO.md](llms.txt/TODO.md)**: Roadmap, planned features, completed items
- **[SECURITY.md](SECURITY.md)**: Local security orchestration, secret scanning, OWASP compliance
- **[llms.txt/CHANGELOG.md](llms.txt/CHANGELOG.md)**: Version history

## License

MIT License

## Acknowledgments

- **Qwen3-VL-Embedding** — Vision-language embedding and reranking (Qwen team)
- **Paddle-ERNIE-RAG** — Hybrid search and multimodal RAG patterns
- [alexzhang13/rlm](https://github.com/alexzhang13/rlm) — Comprehensive RLM framework
- [ysz/recursive-llm](https://github.com/ysz/recursive-llm) — Minimal recursive LLM implementation
