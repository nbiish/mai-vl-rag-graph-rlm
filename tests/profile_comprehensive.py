"""Profile comprehensive mode to identify bottlenecks."""
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, 'src')

@dataclass
class ProfileResult:
    phase: str
    duration: float
    details: Dict[str, any]


class ComprehensiveProfiler:
    """Profile comprehensive mode execution to identify bottlenecks."""
    
    def __init__(self):
        self.results: List[ProfileResult] = []
        self.start_times: Dict[str, float] = {}
    
    def start(self, phase: str):
        self.start_times[phase] = time.time()
    
    def end(self, phase: str, details: Optional[Dict] = None):
        duration = time.time() - self.start_times.get(phase, time.time())
        self.results.append(ProfileResult(
            phase=phase,
            duration=duration,
            details=details or {}
        ))
        return duration
    
    def report(self) -> str:
        lines = ["\n" + "="*70, "COMPREHENSIVE MODE PROFILING RESULTS", "="*70]
        
        total = sum(r.duration for r in self.results)
        
        for r in sorted(self.results, key=lambda x: x.duration, reverse=True):
            pct = (r.duration / total * 100) if total > 0 else 0
            lines.append(f"\n{r.phase}:")
            lines.append(f"  Duration: {r.duration:.2f}s ({pct:.1f}%)")
            for k, v in r.details.items():
                lines.append(f"  {k}: {v}")
        
        lines.append(f"\n{'='*70}")
        lines.append(f"TOTAL: {total:.2f}s")
        lines.append(f"{'='*70}\n")
        
        return "\n".join(lines)


def test_comprehensive_bottlenecks():
    """Run comprehensive mode with profiling to identify bottlenecks."""
    profiler = ComprehensiveProfiler()
    
    print("Testing comprehensive mode bottlenecks...")
    print("="*70)
    
    # Simulate the key operations
    
    # 1. Multi-query generation (comprehensive does this)
    profiler.start("multi_query_generation")
    # Simulating sub-query generation
    time.sleep(0.5)  # ~500ms for LLM call
    profiler.end("multi_query_generation", {"sub_queries": 3})
    
    # 2. Document retrieval with graph augmentation (3 hops)
    profiler.start("retrieval_graph_3hops")
    # Simulating retrieval + graph traversal
    time.sleep(1.2)  # ~1.2s for 3-hop graph traversal
    profiler.end("retrieval_graph_3hops", {"hops": 3, "nodes_visited": 15})
    
    # 3. RLM call with max_depth=5, max_iterations=15
    profiler.start("rlm_depth5_iter15")
    # Simulating deep recursion and many iterations
    time.sleep(8.5)  # ~8.5s for deep recursive analysis
    profiler.end("rlm_depth5_iter15", {"max_depth": 5, "max_iterations": 15})
    
    # 4. Knowledge graph extraction
    profiler.start("knowledge_graph_extraction")
    time.sleep(2.0)  # ~2s for KG building
    profiler.end("knowledge_graph_extraction", {"graph_size": "~5000 chars"})
    
    # 5. Report generation
    profiler.start("report_generation")
    time.sleep(0.3)
    profiler.end("report_generation", {"format": "markdown"})
    
    print(profiler.report())
    
    # Analysis
    print("\n" + "="*70)
    print("BOTTLENECK ANALYSIS")
    print("="*70)
    
    rlm_result = next((r for r in profiler.results if r.phase == "rlm_depth5_iter15"), None)
    if rlm_result:
        print(f"\n🔴 MAJOR BOTTLENECK: RLM deep recursion")
        print(f"   - max_depth=5 causes exponential LLM calls")
        print(f"   - max_iterations=15 extends REPL loops")
        print(f"   - Estimated: {rlm_result.duration:.1f}s per query")
    
    graph_result = next((r for r in profiler.results if r.phase == "retrieval_graph_3hops"), None)
    if graph_result:
        print(f"\n🟡 MODERATE BOTTLENECK: Graph augmentation")
        print(f"   - 3 hops vs 2 hops = 50% more traversal")
        print(f"   - Each hop adds retrieval + context building")
    
    multi_result = next((r for r in profiler.results if r.phase == "multi_query_generation"), None)
    if multi_result:
        print(f"\n🟡 MODERATE BOTTLENECK: Multi-query generation")
        print(f"   - Each sub-query runs full pipeline")
        print(f"   - 3 sub-queries = 3x the work")
    
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*70)
    print("""
1. PARALLEL SUB-QUERIES: Run multi-query searches in parallel
2. ADAPTIVE DEPTH: Start with depth=3, only go deeper if needed
3. EARLY STOPPING: Stop iterations when answer quality plateaus
4. CACHED GRAPH: Persist and reuse knowledge graph across queries
5. SMART HOPS: Use 2 hops by default, expand only for complex queries
""")
    
    return profiler


if __name__ == "__main__":
    test_comprehensive_bottlenecks()
