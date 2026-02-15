#!/usr/bin/env python3
"""Full matrix benchmark for consolidated MCP-aligned profiles.

Runs timed validation across:
- Content: PowerPoint + Video
- Modes: balanced, comprehensive, expanded_comprehensive

Each test runs in an isolated process with timeout to avoid hangs.
"""

from __future__ import annotations

import json
import argparse
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, "src")


@dataclass
class CaseResult:
    content_type: str
    input_path: str
    mode: str
    duration_seconds: float
    status: str  # success | error | timeout
    stage: str = "analysis"  # preflight | analysis
    failure_category: Optional[str] = None
    document_count: int = 0
    chunk_count: int = 0
    query_count: int = 0
    error: Optional[str] = None


PROFILES: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "max_depth": 3,
        "max_iterations": 8,
        "multi_query": True,
        "graph_augmented": True,
        "graph_hops": 2,
        "use_api": True,
        "text_only": False,
    },
    "comprehensive": {
        "max_depth": 5,
        "max_iterations": 15,
        "multi_query": True,
        "graph_augmented": True,
        "graph_hops": 2,
        "use_api": True,
        "text_only": False,
    },
    "expanded_comprehensive": {
        "max_depth": 7,
        "max_iterations": 22,
        "multi_query": True,
        "graph_augmented": True,
        "graph_hops": 4,
        "use_api": True,
        "text_only": False,
    },
}


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


def _api_preflight_worker(queue: mp.Queue, provider: str, include_omni: bool) -> None:
    """Worker process: verify provider credentials/model availability quickly."""
    from vl_rag_graph_rlm.rag.api_embedding import create_api_embedder

    start = time.time()
    try:
        embedder = create_api_embedder()

        # Fast embedding probe (primary API path)
        embed_resp = embedder._emb_client.embeddings.create(
            model=embedder._emb_model,
            input="healthcheck",
        )
        emb_dim = len(embed_resp.data[0].embedding)

        if include_omni:
            if embedder._omni_client is None:
                raise RuntimeError("ZENMUX_API_KEY not configured for media preflight")

            embedder._omni_client.chat.completions.create(
                model=embedder._omni_model,
                messages=[{"role": "user", "content": "Reply with OK only."}],
                max_tokens=4,
                temperature=0,
            )

        queue.put(
            {
                "status": "success",
                "duration_seconds": time.time() - start,
                "error": None,
                "details": f"provider={provider}; emb_dim={emb_dim}; include_omni={include_omni}",
            }
        )
    except Exception as exc:  # pragma: no cover
        queue.put(
            {
                "status": "error",
                "duration_seconds": time.time() - start,
                "error": str(exc),
                "details": None,
            }
        )


def run_api_preflight(content_type: str, provider: str, timeout_seconds: int) -> dict[str, Any]:
    """Run fast API preflight check for current content type.

    - Always checks embeddings endpoint.
    - For video, also checks omni model availability.
    """
    include_omni = content_type == "video"
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_api_preflight_worker,
        args=(queue, provider, include_omni),
        daemon=True,
    )

    start = time.time()
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "status": "timeout",
            "duration_seconds": time.time() - start,
            "error": f"Preflight timed out after {timeout_seconds}s",
            "details": None,
        }

    if queue.empty():
        return {
            "status": "error",
            "duration_seconds": time.time() - start,
            "error": "Preflight worker exited without result",
            "details": None,
        }

    return queue.get()


PHASE_MODES: Dict[str, list[str]] = {
    # Pre-final gate: validated MCP-exposed modes + expansion stress profile
    "pre_final": ["balanced", "comprehensive", "expanded_comprehensive"],
    # Final confirmation: consolidated user-facing modes only
    "final_confirmation": ["balanced", "comprehensive"],
}


def _run_case_worker(
    queue: mp.Queue,
    input_path: str,
    content_type: str,
    mode: str,
    profile: Dict[str, Any],
    provider: str,
) -> None:
    """Worker process: run a single benchmark case and push result payload to queue."""
    from vrlmrag import run_analysis

    start = time.time()
    try:
        result = run_analysis(
            provider=provider,
            input_path=input_path,
            query="What are the main topics and key concepts presented?",
            max_depth=profile["max_depth"],
            max_iterations=profile["max_iterations"],
            multi_query=profile["multi_query"],
            use_graph_augmented=profile["graph_augmented"],
            graph_hops=profile["graph_hops"],
            use_api=profile["use_api"],
            text_only=profile["text_only"],
            verbose=False,
            _quiet=True,
        )
        duration = time.time() - start
        queue.put(
            {
                "status": "success",
                "duration_seconds": duration,
                "document_count": result.get("document_count", 0),
                "chunk_count": result.get("total_chunks", 0),
                "query_count": len(result.get("queries", [])),
                "stage": "analysis",
                "failure_category": None,
                "error": None,
            }
        )
    except Exception as exc:  # pragma: no cover
        duration = time.time() - start
        queue.put(
            {
                "status": "error",
                "duration_seconds": duration,
                "document_count": 0,
                "chunk_count": 0,
                "query_count": 0,
                "stage": "analysis",
                "failure_category": _categorize_failure(str(exc)),
                "error": str(exc),
            }
        )


def run_case(
    input_path: str,
    content_type: str,
    mode: str,
    timeout_seconds: int,
    provider: str,
) -> CaseResult:
    profile = PROFILES[mode]
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_run_case_worker,
        args=(queue, input_path, content_type, mode, profile, provider),
        daemon=True,
    )

    start = time.time()
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return CaseResult(
            content_type=content_type,
            input_path=input_path,
            mode=mode,
            duration_seconds=time.time() - start,
            status="timeout",
            stage="analysis",
            failure_category="timeout",
            error=f"Timed out after {timeout_seconds}s",
        )

    if queue.empty():
        return CaseResult(
            content_type=content_type,
            input_path=input_path,
            mode=mode,
            duration_seconds=time.time() - start,
            status="error",
            stage="analysis",
            failure_category="unknown",
            error="Worker exited without result",
        )

    payload = queue.get()
    return CaseResult(
        content_type=content_type,
        input_path=input_path,
        mode=mode,
        duration_seconds=payload["duration_seconds"],
        status=payload["status"],
        stage=payload.get("stage", "analysis"),
        failure_category=payload.get("failure_category"),
        document_count=payload["document_count"],
        chunk_count=payload["chunk_count"],
        query_count=payload["query_count"],
        error=payload["error"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run consolidated PPTX/video benchmark matrix.")
    parser.add_argument(
        "--phase",
        choices=sorted(PHASE_MODES.keys()),
        default="pre_final",
        help="pre_final (includes expanded stress profile) or final_confirmation",
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        help="Provider used for benchmark analysis and preflight (default: openrouter)",
    )
    parser.add_argument(
        "--pptx-timeout",
        type=int,
        default=180,
        help="Per-case analysis timeout for PPTX inputs (seconds)",
    )
    parser.add_argument(
        "--video-timeout",
        type=int,
        default=300,
        help="Per-case analysis timeout for video inputs (seconds)",
    )
    parser.add_argument(
        "--api-preflight-timeout",
        type=int,
        default=20,
        help="Preflight timeout for text/document paths (seconds)",
    )
    parser.add_argument(
        "--media-preflight-timeout",
        type=int,
        default=45,
        help="Preflight timeout for media paths (seconds)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip API preflight checks (not recommended for production gate runs)",
    )
    args = parser.parse_args()

    repo_root = Path("/Volumes/1tb-sandisk/code-external/mai-vl-rag-graph-rlm")
    examples = repo_root / "examples"

    pptx_path = examples / "Overview of International Business.pptx"
    video_path = (
        examples
        / "Real-Time, Low Latency and High Temporal Resolution Spectrograms - Alexandre R.J. Francois - ADC.mp4"
    )

    if not pptx_path.exists():
        pptx_path = repo_root / "README.md"

    test_matrix = [
        ("pptx", str(pptx_path), args.pptx_timeout),
        ("video", str(video_path), args.video_timeout),
    ]

    modes = PHASE_MODES[args.phase]
    all_results: list[CaseResult] = []

    print("=" * 88)
    print("FULL MATRIX BENCHMARK")
    print(f"Phase: {args.phase}")
    print(f"Provider: {args.provider}")
    print("=" * 88)

    for content_type, input_path, timeout_s in test_matrix:
        if not Path(input_path).exists():
            print(f"[skip] {content_type}: {input_path} not found")
            continue

        if not args.skip_preflight:
            preflight_timeout = args.media_preflight_timeout if content_type == "video" else args.api_preflight_timeout
            print(
                f"\n[{content_type}] preflight provider={args.provider} "
                f"(timeout {preflight_timeout}s)",
                end="",
                flush=True,
            )
            preflight = run_api_preflight(
                content_type=content_type,
                provider=args.provider,
                timeout_seconds=preflight_timeout,
            )
            print(f"  [{preflight['status']}] {preflight['duration_seconds']:.1f}s")

            if preflight["status"] != "success":
                fail_category = _categorize_failure(preflight.get("error"))
                for mode in modes:
                    all_results.append(
                        CaseResult(
                            content_type=content_type,
                            input_path=input_path,
                            mode=mode,
                            duration_seconds=preflight["duration_seconds"],
                            status="error",
                            stage="preflight",
                            failure_category=fail_category,
                            error=preflight.get("error"),
                        )
                    )
                print(f"  [gate] skipping analysis for {content_type} due to preflight failure")
                continue

        print(f"\n[{content_type}] {input_path}")
        for mode in modes:
            print(f"  -> {mode:24s} (timeout {timeout_s}s)", end="", flush=True)
            result = run_case(
                input_path=input_path,
                content_type=content_type,
                mode=mode,
                timeout_seconds=timeout_s,
                provider=args.provider,
            )
            all_results.append(result)
            print(
                f"  [{result.status}] {result.duration_seconds:.1f}s"
                f" [{result.stage}/{result.failure_category or 'n/a'}]"
            )

    out_json = repo_root / "tests" / "full_matrix_benchmark_results.json"
    out_md = repo_root / "tests" / "full_matrix_benchmark_results.md"

    out_json.write_text(json.dumps([asdict(r) for r in all_results], indent=2))

    lines = [
        "# Full Matrix Benchmark Results",
        "",
        f"**Phase:** {args.phase}",
        f"**Provider:** {args.provider}",
        "",
        "| Content | Mode | Status | Stage | Category | Time (s) | Docs | Chunks | Queries | Error |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for r in all_results:
        lines.append(
            f"| {r.content_type} | {r.mode} | {r.status} | {r.stage} | {r.failure_category or ''} | {r.duration_seconds:.1f} | "
            f"{r.document_count} | {r.chunk_count} | {r.query_count} | {r.error or ''} |"
        )

    # overhead section
    for ctype in {r.content_type for r in all_results}:
        comp = next((r for r in all_results if r.content_type == ctype and r.mode == "comprehensive" and r.status == "success"), None)
        exp = next((r for r in all_results if r.content_type == ctype and r.mode == "expanded_comprehensive" and r.status == "success"), None)
        if comp and exp and comp.duration_seconds > 0:
            overhead = exp.duration_seconds / comp.duration_seconds
            lines.extend([
                "",
                f"- {ctype}: expanded/comprehensive overhead = **{overhead:.2f}x**",
            ])

    out_md.write_text("\n".join(lines) + "\n")

    print("\n" + "=" * 88)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
