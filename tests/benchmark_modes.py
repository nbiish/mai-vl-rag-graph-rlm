#!/usr/bin/env python3
"""Benchmark timing test for fast vs comprehensive modes on PowerPoint and video.

Tests both content types with different profiles to measure optimization impact.
"""
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, 'src')


@dataclass
class BenchmarkResult:
    content_type: str  # 'pptx' or 'video'
    mode: str  # 'fast' | 'balanced' | 'thorough' | 'comprehensive' | 'expanded_comprehensive'
    duration_seconds: float
    success: bool
    document_count: int = 0
    chunk_count: int = 0
    query_count: int = 0
    error: Optional[str] = None


def run_benchmark(content_path: str, mode: str, content_type: str) -> BenchmarkResult:
    """Run a single benchmark test."""
    from vrlmrag import run_analysis
    
    # Profile settings matching MCP server
    profiles = {
        "fast": {
            "max_depth": 2,
            "max_iterations": 4,
            "multi_query": False,
            "graph_augmented": False,
            "graph_hops": 0,
            "text_only": False,
            "use_api": True,
        },
        "balanced": {
            "max_depth": 3,
            "max_iterations": 8,
            "multi_query": True,
            "graph_augmented": True,
            "graph_hops": 2,
            "text_only": False,
            "use_api": True,
        },
        "thorough": {
            "max_depth": 4,
            "max_iterations": 12,
            "multi_query": True,
            "graph_augmented": True,
            "graph_hops": 3,
            "text_only": False,
            "use_api": True,
        },
        "comprehensive": {
            "max_depth": 5,
            "max_iterations": 15,
            "multi_query": True,
            "graph_augmented": True,
            "graph_hops": 2,
            "text_only": False,
            "use_api": True,
        },
        # Expanded comprehensive profile to validate high-expansiveness behavior
        "expanded_comprehensive": {
            "max_depth": 7,
            "max_iterations": 22,
            "multi_query": True,
            "graph_augmented": True,
            "graph_hops": 4,
            "text_only": False,
            "use_api": True,
        },
    }
    
    profile = profiles[mode]
    query = "What are the main topics and key concepts presented?"
    
    print(f"\n{'='*70}")
    print(f"Testing: {content_type.upper()} | Mode: {mode.upper()}")
    print(f"Settings: depth={profile['max_depth']}, iterations={profile['max_iterations']}, "
          f"multi_query={profile['multi_query']}, graph={profile['graph_augmented']}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        results = run_analysis(
            provider="openrouter",  # Use a fast API provider
            input_path=content_path,
            query=query,
            max_depth=profile["max_depth"],
            max_iterations=profile["max_iterations"],
            multi_query=profile["multi_query"],
            use_graph_augmented=profile["graph_augmented"],
            graph_hops=profile["graph_hops"],
            text_only=profile["text_only"],
            use_api=profile.get("use_api", True),
            verbose=False,
            _quiet=True,
        )
        
        duration = time.time() - start_time
        
        return BenchmarkResult(
            content_type=content_type,
            mode=mode,
            duration_seconds=duration,
            success=True,
            document_count=results.get('document_count', 0),
            chunk_count=results.get('total_chunks', 0),
            query_count=len(results.get('queries', [])),
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return BenchmarkResult(
            content_type=content_type,
            mode=mode,
            duration_seconds=duration,
            success=False,
            error=str(e),
        )


def print_summary(results: list[BenchmarkResult]):
    """Print formatted summary of all benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    # Group by content type
    by_content = {}
    for r in results:
        if r.content_type not in by_content:
            by_content[r.content_type] = {}
        by_content[r.content_type][r.mode] = r
    
    mode_order = ["fast", "balanced", "thorough", "comprehensive", "expanded_comprehensive"]

    for content_type, modes in by_content.items():
        print(f"\n📁 {content_type.upper()}")
        print("-" * 50)

        for mode in mode_order:
            result = modes.get(mode)
            if result is None:
                continue
            status = "✅" if result.success else "❌"
            print(f"  {status} {mode:24s} {result.duration_seconds:7.1f}s")
            if result.error:
                print(f"      Error: {result.error}")

        # Key comparisons
        fast = modes.get("fast")
        comp = modes.get("comprehensive")
        expanded = modes.get("expanded_comprehensive")

        if fast and comp and fast.success and comp.success:
            speedup = comp.duration_seconds / fast.duration_seconds
            print(f"  ↳ fast vs comprehensive: {speedup:.2f}x (comprehensive/fast)")

        if comp and expanded and comp.success and expanded.success:
            expansion_overhead = expanded.duration_seconds / comp.duration_seconds
            print(f"  ↳ expanded overhead:      {expansion_overhead:.2f}x (expanded/comprehensive)")
    
    print("\n" + "="*70)


def main():
    """Run full benchmark suite."""
    print("\n" + "="*70)
    print("VL-RAG COMPREHENSIVE MODE BENCHMARK")
    print("="*70)
    print("Testing fast vs comprehensive modes on PowerPoint and video")
    
    # Prefer real examples used in manual validation
    repo_root = Path("/Volumes/1tb-sandisk/code-external/mai-vl-rag-graph-rlm")
    examples_dir = repo_root / "examples"

    pptx_candidates = [
        examples_dir / "Overview of International Business.pptx",
        examples_dir / "Writing Tutorial 2022.pptx",
    ]
    video_candidates = [
        examples_dir / "Real-Time, Low Latency and High Temporal Resolution Spectrograms - Alexandre R.J. Francois - ADC.mp4",
    ]

    pptx_file = next((p for p in pptx_candidates if p.exists()), None)
    video_file = next((v for v in video_candidates if v.exists()), None)

    if pptx_file is None:
        pptx_file = repo_root / "README.md"
    
    results = []
    
    modes_to_test = ["fast", "balanced", "thorough", "comprehensive", "expanded_comprehensive"]

    # Test PowerPoint (or README as text fallback)
    if pptx_file.exists():
        for mode in modes_to_test:
            results.append(run_benchmark(str(pptx_file), mode, "pptx"))
    else:
        print("\n⚠️  No PowerPoint file found, skipping PPTX tests")
    
    # Test Video
    if video_file and video_file.exists():
        # Video can be expensive; focus on key profiles + expanded
        for mode in ["fast", "comprehensive", "expanded_comprehensive"]:
            results.append(run_benchmark(str(video_file), mode, "video"))
    else:
        print("\n⚠️  No video file found, skipping video tests")
    
    # Print summary
    print_summary(results)
    
    # Save results to JSON
    results_dict = [asdict(r) for r in results]
    results_file = Path("benchmark_results.json")
    results_file.write_text(json.dumps(results_dict, indent=2))
    print(f"\n📊 Results saved to: {results_file}")
    
    # Check if comprehensive is usable (< 120s target)
    print("\n" + "="*70)
    print("USABILITY CHECK")
    print("="*70)
    
    comprehensive_results = [r for r in results if r.mode == "comprehensive" and r.success]
    
    if comprehensive_results:
        avg_comp_time = sum(r.duration_seconds for r in comprehensive_results) / len(comprehensive_results)
        print(f"Average comprehensive time: {avg_comp_time:.1f}s")
        
        if avg_comp_time < 60:
            print("✅ EXCELLENT: Comprehensive mode under 60s")
        elif avg_comp_time < 120:
            print("✅ GOOD: Comprehensive mode under 2 minutes")
        else:
            print(f"⚠️  NEEDS WORK: Comprehensive mode over 2 minutes ({avg_comp_time:.1f}s)")
    
    fast_results = [r for r in results if r.mode == "fast" and r.success]
    if fast_results:
        avg_fast_time = sum(r.duration_seconds for r in fast_results) / len(fast_results)
        print(f"Average fast mode time: {avg_fast_time:.1f}s")
        
        if avg_fast_time < 30:
            print("✅ EXCELLENT: Fast mode under 30s")
        elif avg_fast_time < 60:
            print("✅ GOOD: Fast mode under 60s")
        else:
            print(f"⚠️  NEEDS WORK: Fast mode over 60s ({avg_fast_time:.1f}s)")

    expanded_results = [r for r in results if r.mode == "expanded_comprehensive" and r.success]
    if expanded_results:
        avg_exp_time = sum(r.duration_seconds for r in expanded_results) / len(expanded_results)
        print(f"Average expanded comprehensive time: {avg_exp_time:.1f}s")

        if avg_exp_time < 180:
            print("✅ GOOD: Expanded comprehensive under 3 minutes")
        else:
            print(f"⚠️  NEEDS WORK: Expanded comprehensive over 3 minutes ({avg_exp_time:.1f}s)")
    
    print("="*70)


if __name__ == "__main__":
    main()
