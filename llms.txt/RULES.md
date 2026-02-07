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

## If

- Qwen3-VL is unavailable → wrap import in `try/except ImportError`, fall back to `MultiFactorReranker` for text-only reranking
- Provider has token limits (e.g., SambaNova 200K TPD) → budget context size accordingly (8K chars per call)
- Provider has no token limits (e.g., Nebius) → use larger context windows (up to 100K chars)
- Running on Apple Silicon → prefer `device="mps"` for Qwen3-VL inference (add MPS check before CUDA)
- A template is for a generic/compatible provider → use the client directly but still wire through the full pipeline
