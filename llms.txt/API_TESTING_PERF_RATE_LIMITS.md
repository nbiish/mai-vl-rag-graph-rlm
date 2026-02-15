# API-Only Testing, Performance Plan, and Rate-Limit Reference

Date: 2026-02-15

## Scope

This plan enforces **API-only** validation for both CLI and MCP flows.
No local/offline model paths are included in final readiness testing.

## What the latest terminal/testing outputs show

From current benchmark runs (`tests/full_matrix_benchmark.py`):

- Pre-final matrix ended with **100% timeouts** for PPTX and video.
- Video path repeatedly logged provider issues before timeout:
  - OpenRouter primary model fallback events (`z-ai/glm-5` -> `deepseek/deepseek-v3.2`).
  - ZenMux audio/transcription failures including payload-size exhaustion and invalid fallback model.

Result artifacts:
- `tests/full_matrix_benchmark_results.md`
- `tests/full_matrix_benchmark_results.json`

## API-only test strategy (faster + cleaner)

### 1) Split tests into three gates

1. **Gate A: API configuration sanity (2-4 min) — required on every change**
   - CLI help/import sanity.
   - MCP tool schema sanity (`balanced`/`comprehensive` only).
   - Provider key presence and model resolution checks.

2. **Gate B: API smoke execution (5-10 min) — required on every change**
   - PPTX only, `balanced` mode only.
   - Single provider at a time (start with the highest-stability provider in hierarchy).
   - Tight timeout and immediate fail on transport/auth/model-not-found errors.

3. **Gate C: Full multimodal benchmark (scheduled/manual)**
   - PPTX + video.
   - `balanced` and `comprehensive`; `expanded_comprehensive` only for stress windows.
   - Run only after Gate A/B pass.

### 2) Enforce API-only in test harnesses

- Keep `use_api=True` hardcoded in benchmark profiles.
- Reject accidental `--local`/`--offline` paths during benchmark orchestration.
- Add explicit provider selection flags to benchmark runs (avoid hidden variability from `auto` during smoke tests).

### 3) Reduce wasted runtime

- Add a short provider preflight before heavy ingestion:
  - test a tiny chat completion and return early on 401/403/404/429/5xx patterns.
- Separate media preprocessing/transcription timeout from full query timeout.
- Record structured failure categories: `auth`, `model_missing`, `rate_limited`, `payload_too_large`, `provider_unavailable`, `timeout`.

## Rust migration/integration roadmap (performance-first)

Use Rust where orchestration and parsing overhead is highest, while keeping LLM calls in Python clients initially.

### Priority 1 (high ROI, low migration risk)

1. **Benchmark orchestrator runtime**
   - Reimplement process orchestration + timeout supervision in Rust.
   - Expose a thin Python binding/CLI wrapper.
   - Benefits: faster spawn/control, stronger timeout guarantees, lower Python multiprocessing overhead.

2. **Chunking + file manifest scanning**
   - Move hot path text chunking and filesystem manifest diffing to Rust.
   - Benefits: lower latency for large corpora and repeated scans.

3. **Structured log/event pipeline**
   - Rust event collector for per-stage timings and error taxonomy.
   - Benefits: precise bottleneck identification and reduced Python logging overhead.

### Priority 2 (targeted acceleration)

4. **RRF fusion and lightweight rerank prefilters**
   - Rust implementation for dense+keyword fusion and initial candidate pruning.

5. **Knowledge-graph parsing/merge utilities**
   - Rust graph merge + dedup helpers for large graph updates.

### Keep in Python initially

- Provider SDK integrations (fast-changing APIs).
- Prompt composition and high-level business logic.

## Verified rate-limit and quota references

> Note: many providers expose exact numeric limits per account tier and model in dashboard pages. Store those account-specific values internally after retrieval.

### OpenRouter
- Doc: https://openrouter.ai/docs/api/reference/limits
- Pricing/plan FAQ: https://openrouter.ai/pricing
- Verified points:
  - Free `:free` models: 20 RPM.
  - Free-plan/day quotas apply (e.g., 50/day baseline, up to 1000/day after credit threshold).
  - Key introspection endpoint available: `GET /api/v1/key`.

### DeepSeek
- Doc: https://api-docs.deepseek.com/quick_start/rate_limit
- Verified points:
  - States no hard user rate limit constraint in docs.
  - Under load, requests may stay connected with keep-alives.
  - If inference has not started after 10 minutes, server closes connection.

### Groq
- Doc: https://console.groq.com/docs/rate-limits
- Verified points:
  - Limits tracked by RPM/RPD/TPM/TPD (and audio-specific units in docs).
  - 429 on limit exceed with `retry-after` guidance.
  - Exact tier/model limits are account-visible on limits page.

### Cerebras Inference
- Doc: https://inference-docs.cerebras.ai/support/rate-limits
- Verified points:
  - Token-bucket-style limiting with replenishment behavior.
  - Headers include request/day and tokens/minute limits and reset windows.
  - 429 for limit exceed.

### Anthropic
- Doc: https://platform.claude.com/docs/en/api/rate-limits
- Verified points:
  - Limits measured with RPM, input TPM, output TPM per model class.
  - 429 with `retry-after` when exceeded.
  - Rich rate-limit headers for requests/tokens and reset values.

### OpenAI
- Doc: https://developers.openai.com/api/docs/guides/rate-limits
- Verified points:
  - Limits may apply across RPM/RPD/TPM/TPD/IPM.
  - Limits vary by model, org/project tier, and shared limit groups.
  - Exact live limits shown in platform limits dashboard.

### SambaNova
- Doc: https://docs.sambanova.ai/docs/en/models/rate-limits
- Verified points:
  - Free and Developer tiers with production vs preview model distinctions.
  - Rate-limit headers provided for per-minute/per-day requests and reset.

### Nebius Token Factory
- Doc: https://docs.tokenfactory.nebius.com/ai-models-inference/rate-limits
- Verified points:
  - Dynamic scale-up/down behavior over 15-minute windows.
  - Response headers expose limits/remaining/reset and dynamic scaling indicators.
  - `Retry-After` and over-limit signaling documented.

### Modal GLM-5 endpoint
- Blog reference: https://modal.com/blog/try-glm-5
- Verified point:
  - Public note indicates higher limits via sales contact for production workloads.
  - Exact numeric limits are not published in the referenced post.

### ZenMux (critical current bottleneck)
- Public doc source in this audit is insufficient for numeric global limits.
- Observed runtime evidence from current terminal output indicates:
  - audio payload size rejection (`RESOURCE_EXHAUSTED` with 33,554,432-byte ceiling reported in error body)
  - invalid fallback model configuration paths (404 invalid_model)
- Action: capture ZenMux account-level limits and model availability matrix from provider console/docs and pin supported fallback models in `.env`.

## Immediate next implementation actions

1. Add **API preflight** stage to benchmark runner before document processing.
2. Add **error taxonomy fields** to benchmark JSON/markdown output.
3. Add separate **media-transcription timeout** and **overall query timeout**.
4. Add **provider-specific smoke script** (single tiny prompt per provider) to fail fast.
5. Start a Rust sidecar crate for orchestration/timeouts/structured metrics, invoked from Python.
