#!/usr/bin/env python3
"""Quick timing test for MCP modes using README as test content."""
import sys
import time
sys.path.insert(0, 'src')

from vrlmrag import run_analysis

def test_mode(mode: str, profile: dict):
    """Test a single mode and return timing."""
    print(f"\n{'='*60}")
    print(f"Testing: {mode.upper()}")
    print(f"Settings: {profile}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        results = run_analysis(
            provider="openrouter",
            input_path="README.md",
            query="What is this project about?",
            max_depth=profile['max_depth'],
            max_iterations=profile['max_iterations'],
            multi_query=profile['multi_query'],
            use_graph_augmented=profile['graph_augmented'],
            graph_hops=profile.get('graph_hops', 2),
            text_only=profile['text_only'],
            use_api=profile.get("use_api", not profile["text_only"]),
            verbose=False,
            _quiet=True,
        )
        duration = time.time() - start
        
        print(f"✅ SUCCESS: {duration:.1f}s")
        print(f"   Documents: {results.get('document_count', 0)}")
        print(f"   Chunks: {results.get('total_chunks', 0)}")
        print(f"   Queries: {len(results.get('queries', []))}")
        
        # Check if parallel execution worked (comprehensive mode should show it)
        if mode == 'comprehensive' and profile['multi_query']:
            print(f"   🔄 Parallel multi-query: Enabled")
        
        return duration, True
        
    except Exception as e:
        duration = time.time() - start
        print(f"❌ FAILED: {duration:.1f}s - {e}")
        return duration, False


def main():
    print("\n" + "="*60)
    print("MCP MODE TIMING TEST (README.md)")
    print("="*60)
    
    # Test all four modes - matching MCP server profiles
    profiles = {
        "fast": {
            "max_depth": 2,
            "max_iterations": 4,
            "multi_query": False,
            "graph_augmented": False,
            "graph_hops": 0,
            "text_only": False,
            "use_api": True,  # Use API for speed
        },
        "balanced": {
            "max_depth": 3,
            "max_iterations": 8,
            "multi_query": True,
            "graph_augmented": True,
            "graph_hops": 2,
            "text_only": False,
        },
        "thorough": {
            "max_depth": 4,
            "max_iterations": 12,
            "multi_query": True,
            "graph_augmented": True,
            "graph_hops": 3,
            "text_only": False,
        },
        "comprehensive": {
            "max_depth": 5,
            "max_iterations": 15,
            "multi_query": True,
            "graph_augmented": True,
            "graph_hops": 2,
            "text_only": False,
        },
    }
    
    results = {}
    for mode, profile in profiles.items():
        duration, success = test_mode(mode, profile)
        results[mode] = {'duration': duration, 'success': success}
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for mode, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {mode:15s}: {result['duration']:6.1f}s")
    
    # Compare fast vs comprehensive
    if results['fast']['success'] and results['comprehensive']['success']:
        fast_time = results['fast']['duration']
        comp_time = results['comprehensive']['duration']
        
        print("\n" + "-"*60)
        if fast_time < comp_time:
            speedup = comp_time / fast_time
            print(f"🚀 Fast mode is {speedup:.1f}x faster than comprehensive")
            print(f"   Time saved: {comp_time - fast_time:.1f}s")
        else:
            overhead = fast_time / comp_time
            print(f"⚠️  Fast mode is {overhead:.1f}x SLOWER than comprehensive")
            print(f"   (Local text-only model slower than API)")
    
    print("\n" + "="*60)
    print("TARGET TIMES:")
    print("  Fast:          < 30s (text-only local)")
    print("  Balanced:      < 60s (API, default)")
    print("  Thorough:      < 90s (API, deep)")
    print("  Comprehensive: < 120s (API, parallel multi-query)")
    print("="*60)


if __name__ == "__main__":
    main()
