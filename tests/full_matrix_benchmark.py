#!/usr/bin/env python3
"""Full matrix benchmark for standard + expanded comprehensive profiles.

Runs timed validation across:
- Content: PowerPoint + Video
- Modes: fast, balanced, thorough, comprehensive, expanded_comprehensive

Each test runs in an isolated process with timeout to avoid hangs.
"""

from __future__ import annotations

import json
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
    document_count: int = 0
    chunk_count: int = 0
    query_count: int = 0
    error: Optional[str] = None


PROFILES: Dict[str, Dict[str, Any]] = {
    "fast": {
        "max_depth": 2,
        "max_iterations": 4,
        "multi_query": False,
        "graph_augmented": False,
        "graph_hops": 0,
        "use_api": True,
        "text_only": False,
    },
    "balanced": {
        "max_depth": 3,
        "max_iterations": 8,
        "multi_query": True,
        "graph_augmented": True,
        "graph_hops": 2,
        "use_api": True,
        "text_only": False,
    },
    "thorough": {
        "max_depth": 4,
        "max_iterations": 12,
        "multi_query": True,
        "graph_augmented": True,
        "graph_hops": 3,
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


def _run_case_worker(
    queue: mp.Queue,
    input_path: str,
    content_type: str,
    mode: str,
    profile: Dict[str, Any],
) -> None:
    """Worker process: run a single benchmark case and push result payload to queue."""
    from vrlmrag import run_analysis

    start = time.time()
    try:
        result = run_analysis(
            provider="openrouter",
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
                "error": str(exc),
            }
        )


def run_case(input_path: str, content_type: str, mode: str, timeout_seconds: int) -> CaseResult:
    profile = PROFILES[mode]
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_run_case_worker,
        args=(queue, input_path, content_type, mode, profile),
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
            error=f"Timed out after {timeout_seconds}s",
        )

    if queue.empty():
        return CaseResult(
            content_type=content_type,
            input_path=input_path,
            mode=mode,
            duration_seconds=time.time() - start,
            status="error",
            error="Worker exited without result",
        )

    payload = queue.get()
    return CaseResult(
        content_type=content_type,
        input_path=input_path,
        mode=mode,
        duration_seconds=payload["duration_seconds"],
        status=payload["status"],
        document_count=payload["document_count"],
        chunk_count=payload["chunk_count"],
        query_count=payload["query_count"],
        error=payload["error"],
    )


def main() -> None:
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
        ("pptx", str(pptx_path), 180),
        ("video", str(video_path), 300),
    ]

    modes = ["fast", "balanced", "thorough", "comprehensive", "expanded_comprehensive"]
    all_results: list[CaseResult] = []

    print("=" * 88)
    print("FULL MATRIX BENCHMARK")
    print("=" * 88)

    for content_type, input_path, timeout_s in test_matrix:
        if not Path(input_path).exists():
            print(f"[skip] {content_type}: {input_path} not found")
            continue

        print(f"\n[{content_type}] {input_path}")
        for mode in modes:
            print(f"  -> {mode:24s} (timeout {timeout_s}s)", end="", flush=True)
            result = run_case(input_path, content_type, mode, timeout_s)
            all_results.append(result)
            print(f"  [{result.status}] {result.duration_seconds:.1f}s")

    out_json = repo_root / "tests" / "full_matrix_benchmark_results.json"
    out_md = repo_root / "tests" / "full_matrix_benchmark_results.md"

    out_json.write_text(json.dumps([asdict(r) for r in all_results], indent=2))

    lines = [
        "# Full Matrix Benchmark Results",
        "",
        "| Content | Mode | Status | Time (s) | Docs | Chunks | Queries | Error |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in all_results:
        lines.append(
            f"| {r.content_type} | {r.mode} | {r.status} | {r.duration_seconds:.1f} | "
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
