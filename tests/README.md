# Testing Orchestration (Pre-Final -> Final Confirmation)

This folder is consolidated around a single canonical orchestrator:

- `full_matrix_benchmark.py`

It runs timed validation on the two core multimodal assets:

- PowerPoint: `examples/Overview of International Business.pptx`
- Video: `examples/Real-Time, Low Latency and High Temporal Resolution Spectrograms - Alexandre R.J. Francois - ADC.mp4`

## API-Only Fast Gate Workflow (Production)

### Gate A) Provider smoke checks (fast fail)

```bash
python tests/provider_api_smoke.py \
  --providers openrouter sambanova nebius groq cerebras zai zenmux \
  --timeout 90
```

Output:
- `tests/provider_api_smoke_results.json`

### Gate B) Pre-final benchmark with API preflight

```bash
python tests/full_matrix_benchmark.py \
  --phase pre_final \
  --provider openrouter \
  --api-preflight-timeout 20 \
  --media-preflight-timeout 45 \
  --pptx-timeout 180 \
  --video-timeout 300
```

### Gate C) Final confirmation benchmark

```bash
python tests/full_matrix_benchmark.py --phase final_confirmation --provider openrouter
```

## Phases

### 1) Pre-final gate

Use this before final confirmation to validate standard and expanded depth:

```bash
python tests/full_matrix_benchmark.py --phase pre_final
```

Profiles exercised:
- `balanced`
- `comprehensive`
- `expanded_comprehensive` (stress/depth profile)

### 2) Final confirmation gate

Use this after pre-final passes to confirm user-facing behavior:

```bash
python tests/full_matrix_benchmark.py --phase final_confirmation
```

Profiles exercised:
- `balanced`
- `comprehensive`

## Backward compatibility wrappers

- `benchmark_modes.py` delegates to `--phase pre_final`
- `timing_test.py` delegates to `--phase final_confirmation`

These wrappers are kept to avoid breaking older commands, but `full_matrix_benchmark.py` is the source of truth.

## Outputs

Each run writes:
- `tests/full_matrix_benchmark_results.json`
- `tests/full_matrix_benchmark_results.md`

The markdown report includes per-content timing and expanded/comprehensive overhead where available.

## Failure Categories (full_matrix_benchmark)

Benchmark output includes `stage` (`preflight` or `analysis`) and `failure_category`:

- `auth` — API key missing/invalid/forbidden
- `model_missing` — model route invalid/deprecated/not available
- `rate_limited` — provider throttling/429
- `payload_too_large` — request too large (common in media/audio)
- `provider_unavailable` — transient provider/network/server failures
- `timeout` — hit configured timeout
- `unknown` — uncategorized failure
