# Rules — VL-RAG-Graph-RLM

## Naming & Comments

- Use descriptive names matching the six-pillar vocabulary: VL, RAG, Reranker, Graph, RLM, Pipeline
- Document every public function with docstrings including Args, Returns, and Example
- Prefix provider-specific code with the provider name (e.g., `run_nebius_analysis`)

## Always

- Exercise all six pillars in every template: VL embeddings, RAG retrieval, reranking, graph extraction, RLM reasoning, report generation
- Use `MultimodalRAGPipeline` or the full manual pipeline pattern from `ARCHITECTURE.md`
- Load `.env` via `python-dotenv` at startup — never hardcode API keys
- Use `create_qwen3vl_embedder()` and `create_qwen3vl_reranker()` factory functions from `vl_rag_graph_rlm.rag`
- Embed both text chunks AND extracted images into the unified vector space
- Apply hybrid search (dense + keyword) with RRF fusion before reranking
- Use `VLRAGGraphRLM` with `max_depth=3`, `max_iterations=10` as defaults
- Generate a markdown report with provider metadata, embedding info, knowledge graph, query responses with sources
- Handle graceful fallback when Qwen3-VL is unavailable — wrap imports in `try/except ImportError` and check `HAS_QWEN3VL`
- Use `transformers>=5.1.0` for Qwen3-VL support (`qwen3_vl` architecture)

## Never

- Create a template that only calls `rlm.completion()` without RAG retrieval — that defeats the architecture
- Skip the reranking stage — always rerank after fusion
- Hardcode API keys or secrets in source files
- Import Qwen3-VL at module level — use lazy imports inside `try/except` blocks
- Assume a specific device — always detect CUDA/CPU dynamically

## Device Detection

Device detection patterns in the codebase:
- **Core library (`qwen3vl.py`)**: Uses `device or ("cuda" if torch.cuda.is_available() else "cpu")` — **no MPS fallback**
- **Templates (`provider_openai.py`)**: Uses `"mps" if torch.backends.mps.is_available() else "cpu"` for Apple Silicon
- **Best practice**: For templates, check MPS first, then CUDA, then CPU:
  ```python
  device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
  ```

## Provider-Specific Rules

### Groq
- Default: `moonshotai/kimi-k2-instruct-0905` (Kimi K2 on Groq LPU)
- Also available: `openai/gpt-oss-120b` (1200 tok/s), `meta-llama/llama-4-maverick-17b-128e-instruct`, `llama-3.3-70b-versatile`, `qwen/qwen3-32b`
- Endpoint: `https://api.groq.com/openai/v1`
- Query live model list: `curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models`

### Cerebras
- Default: `zai-glm-4.7` (GLM 4.7 355B, ~1000 tok/s on wafer-scale)
- Also available: `gpt-oss-120b` (~3000 tok/s), `qwen-3-235b-a22b-instruct-2507` (~1400 tok/s)
- **DEPRECATED Feb 16, 2026**: `llama-3.3-70b`, `qwen-3-32b` — do not use for new integrations
- Endpoint: `https://api.cerebras.ai/v1`

### ZenMux
- ZenMux uses `provider/model-name` format (e.g., `moonshotai/kimi-k2.5`)
- Endpoint: `https://zenmux.ai/api/v1` (OpenAI protocol) or `https://zenmux.ai/api/anthropic` (Anthropic protocol)
- The `ZenMuxClient` uses OpenAI-compatible protocol by default
- 59+ models including Gemini 3 Pro, Claude 3.7 Sonnet, MiMo-V2-Flash, Qwen 3-Max, DeepSeek-V3.2

### z.ai
- z.ai has two endpoints:
  - Coding Plan (flat-rate $3-15/mo): `https://api.z.ai/api/coding/paas/v4`
  - Normal (pay-per-token): `https://open.bigmodel.cn/api/paas/v4`
- `ZaiClient` tries Coding Plan first, automatically falls back to normal endpoint on failure
- Set `ZAI_CODING_PLAN=false` to skip Coding Plan and use normal endpoint directly
- Coding Plan models: `glm-4.7`, `glm-4.5-air`
- GLM-5 expected before Feb 15, 2026

### Universal Model Fallback
- **All providers** have automatic model fallback: if the primary model fails (rate limit, token limit, downtime, network error, etc.), the client retries with a fallback model on the **same provider**
- Fallback models are defined in `OpenAICompatibleClient.FALLBACK_MODELS` and can be overridden per-provider via `{PROVIDER}_FALLBACK_MODEL` env var
- Fallback map:
  - `sambanova`: DeepSeek-V3.2 → DeepSeek-V3.1
  - `groq`: kimi-k2-instruct → llama-3.3-70b-versatile
  - `cerebras`: zai-glm-4.7 → gpt-oss-120b
  - `nebius`: MiniMax-M2.1 → GLM-4.7-FP8
  - `openrouter`: minimax-m2.1 → deepseek-v3.2
  - `zenmux`: kimi-k2.5 → glm-4.7
  - `zai`: glm-4.7 → glm-4.5-air
  - `openai`: gpt-4o-mini → gpt-4o
  - `mistral`: mistral-large → mistral-small
  - `deepseek`: deepseek-chat → deepseek-reasoner
- If already on the fallback model and it fails, the error propagates up to the **provider hierarchy** fallback

### SambaNova
- **Critical**: 200K TPD (tokens per day) free tier limit
- Must budget context aggressively: ~8K chars per call
- Default model: `DeepSeek-V3.2` (128K context, 200+ tok/s)
- Also available: `gpt-oss-120b`, `Qwen3-235B`, `Llama-4-Maverick-17B-128E-Instruct`
- For unlimited usage, upgrade to Developer tier (12K RPD, no TPD limit)

### Nebius
- Default: `MiniMaxAI/MiniMax-M2.1` — no daily token limits
- Also available: `zai-org/GLM-4.7-FP8`, `deepseek-ai/DeepSeek-R1-0528`, `nvidia/Llama-3.1-Nemotron-Ultra-253B-v1`
- Endpoint: `https://api.tokenfactory.nebius.com/v1`
- Best choice when SambaNova rate-limits are a bottleneck

### Provider Hierarchy
- Default order: `sambanova → nebius → groq → cerebras → zai → zenmux → openrouter → gemini → deepseek → openai → anthropic → mistral → fireworks → together → azure_openai`
- If `openai_compatible` or `anthropic_compatible` have API keys set, they are automatically prepended (highest priority — user set up a custom endpoint)
- Configurable via `PROVIDER_HIERARCHY` env var (comma-separated)
- `get_client('auto')` returns a `HierarchyClient` that tries providers in order
- `HierarchyClient(start_provider='groq')` starts from a specific point in the hierarchy
- CLI defaults to `--provider auto` — no explicit provider needed
- On failure (rate limit, auth, network), automatically falls through to next provider
- Only providers with valid API keys (not `your_*_here` placeholders) are attempted

## If

- Qwen3-VL is unavailable → wrap import in `try/except ImportError`, fall back to `MultiFactorReranker` for text-only reranking
- Provider has token limits (e.g., SambaNova 200K TPD) → budget context size accordingly (8K chars per call)
- Provider has no token limits (e.g., Nebius) → use larger context windows (up to 100K chars)
- Running on Apple Silicon → prefer `device="mps"` for Qwen3-VL inference (add MPS check before CUDA)
- A template is for a generic/compatible provider → use the client directly but still wire through the full pipeline
- Cerebras model deprecated → switch to `zai-glm-4.7` or `gpt-oss-120b`
- No specific provider requested → use `get_client('auto')` or `--provider auto` to engage hierarchy
- Multiple provider templates in same codebase → hierarchy ensures any template works with whichever keys are available
- Provider rate-limited mid-query → hierarchy auto-falls through to next provider in `run_analysis`
