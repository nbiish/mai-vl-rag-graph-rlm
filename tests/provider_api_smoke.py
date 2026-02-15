#!/usr/bin/env python3
"""Fast API smoke checks for provider readiness (CLI/MCP shared backend).

Purpose:
- Validate API credentials and provider model routing quickly.
- Fail fast before long benchmark runs.

This script intentionally runs tiny API-only analyses on README.md with
minimal recursion/iterations to minimize cost and runtime.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, "src")


@dataclass
class SmokeResult:
    provider: str
    status: str  # success | error | timeout
    duration_seconds: float
    failure_category: Optional[str] = None
    error: Optional[str] = None


def _categorize_failure(message: Optional[str]) -> Optional[str]:
    if not message:
        return None

    lower = message.lower()
    if "timed out" in lower:
        return "timeout"
    if any(token in lower for token in ["401", "unauthorized", "forbidden", "api key not set", "not set"]):
        return "auth"
    if any(token in lower for token in ["404", "invalid_model", "model not found", "no endpoints for this model"]):
        return "model_missing"
    if any(token in lower for token in ["429", "rate limit", "retry-after", "too many requests"]):
        return "rate_limited"
    if any(token in lower for token in ["resource_exhausted", "message larger than max", "payload", "too large"]):
        return "payload_too_large"
    if any(token in lower for token in ["connection error", "service unavailable", "provider", "5xx", "500"]):
        return "provider_unavailable"
    return "unknown"


def _smoke_worker(queue: mp.Queue, provider: str, input_path: str) -> None:
    from vrlmrag import run_analysis

    start = time.time()
    try:
        _ = run_analysis(
            provider=provider,
            input_path=input_path,
            query="One sentence: what is this project about?",
            max_depth=1,
            max_iterations=1,
            multi_query=False,
            use_graph_augmented=False,
            graph_hops=0,
            use_api=True,
            text_only=False,
            verbose=False,
            _quiet=True,
        )
        queue.put(
            {
                "status": "success",
                "duration_seconds": time.time() - start,
                "error": None,
                "failure_category": None,
            }
        )
    except Exception as exc:  # pragma: no cover
        err = str(exc)
        queue.put(
            {
                "status": "error",
                "duration_seconds": time.time() - start,
                "error": err,
                "failure_category": _categorize_failure(err),
            }
        )


def run_smoke(provider: str, input_path: str, timeout_seconds: int) -> SmokeResult:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_smoke_worker,
        args=(queue, provider, input_path),
        daemon=True,
    )

    start = time.time()
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        err = f"Timed out after {timeout_seconds}s"
        return SmokeResult(
            provider=provider,
            status="timeout",
            duration_seconds=time.time() - start,
            failure_category="timeout",
            error=err,
        )

    if queue.empty():
        return SmokeResult(
            provider=provider,
            status="error",
            duration_seconds=time.time() - start,
            failure_category="unknown",
            error="Worker exited without result",
        )

    payload = queue.get()
    return SmokeResult(
        provider=provider,
        status=payload["status"],
        duration_seconds=payload["duration_seconds"],
        failure_category=payload.get("failure_category"),
        error=payload.get("error"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast provider API smoke checks.")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["openrouter", "sambanova", "nebius", "groq", "cerebras", "zai", "zenmux"],
        help="Providers to smoke test in order",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Per-provider timeout in seconds (default: 90)",
    )
    parser.add_argument(
        "--input-path",
        default="README.md",
        help="Small text input path for smoke run (default: README.md)",
    )
    parser.add_argument(
        "--output-json",
        default="tests/provider_api_smoke_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    print("=" * 88)
    print("PROVIDER API SMOKE")
    print("=" * 88)

    results: list[SmokeResult] = []
    for provider in args.providers:
        print(f"-> {provider:14s} (timeout {args.timeout}s)", end="", flush=True)
        result = run_smoke(provider=provider, input_path=str(input_path), timeout_seconds=args.timeout)
        results.append(result)
        print(f"  [{result.status}] {result.duration_seconds:.1f}s [{result.failure_category or 'n/a'}]")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))

    print("\n" + "=" * 88)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
