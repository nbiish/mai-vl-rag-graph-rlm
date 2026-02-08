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
│  │  Top-15 candidates → Top-5 results       │                       │
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
store = MultimodalVectorStore(embedding_provider=embedder)
for chunk in chunks:
    store.add_text(content=chunk["content"], metadata={...})
for image in images:
    store.add_image(image_path=path, description=desc)

# 3. Hybrid Search + RRF Fusion
dense_results = store.search(query, top_k=20)
keyword_results = keyword_search(store, query, top_k=20)
fused = ReciprocalRankFusion(k=60).fuse(
    [dense_results, keyword_results], weights=[4.0, 1.0]
)

# 4. Qwen3-VL Reranking
reranker = create_qwen3vl_reranker(device="mps")
reranked = reranker.rerank(query={"text": q}, documents=docs)

# 5. Knowledge Graph Extraction via RLM
rlm = VLRAGGraphRLM(provider="<provider>", model="<model>")
kg = rlm.completion("Extract entities and relationships.", doc_context)

# 6. Query Answering via RLM with Retrieved Context
result = rlm.completion(query, retrieved_context)
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
