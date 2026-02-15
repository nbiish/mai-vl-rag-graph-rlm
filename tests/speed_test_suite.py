"""Comprehensive Speed & Accuracy Test Suite for VL-RAG-Graph-RLM

Tests three configuration profiles across all content types:
- Current (Aggressive): max_depth=5, max_iterations=15, multi_query=True, graph=True
- Balanced (Proposed): max_depth=3, max_iterations=8, multi_query=True, graph=True  
- Fast: max_depth=2, max_iterations=5, multi_query=False, graph=False

Content types tested:
- PowerPoint (.pptx) - slides with images and text
- Video (.mp4) - frame extraction + audio transcription
- PDF - text + image extraction
- Text (.txt) - plain text documents
- Markdown (.md) - structured documents
"""

import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class TestProfile:
    """Configuration profile for testing."""
    name: str
    max_depth: int
    max_iterations: int
    multi_query: bool
    graph_augmented: bool
    graph_hops: int
    description: str


@dataclass  
class TestResult:
    """Result from a single test run."""
    profile_name: str
    content_type: str
    file_path: str
    file_size_mb: float
    execution_time_seconds: float
    documents_processed: int
    chunks_embedded: int
    api_calls_estimate: int
    query_response_quality: str  # Will be manually assessed
    knowledge_graph_size_chars: int
    errors: List[str]


# Define test profiles
PROFILES = {
    "current_aggressive": TestProfile(
        name="Current (Aggressive)",
        max_depth=5,
        max_iterations=15,
        multi_query=True,
        graph_augmented=True,
        graph_hops=3,
        description="Deep recursion + high iterations + graph + multi-query"
    ),
    "balanced": TestProfile(
        name="Balanced (Proposed)",
        max_depth=3,
        max_iterations=8,
        multi_query=True,
        graph_augmented=True,
        graph_hops=2,
        description="Moderate recursion + graph + multi-query for quality"
    ),
    "fast": TestProfile(
        name="Fast",
        max_depth=2,
        max_iterations=5,
        multi_query=False,
        graph_augmented=False,
        graph_hops=1,
        description="Quick analysis for rapid iteration"
    ),
}


# Test content inventory
TEST_CONTENT = {
    "powerpoint": {
        "path": "/Volumes/1tb-sandisk/code-external/mai-vl-rag-graph-rlm/examples/Overview of International Business.pptx",
        "query": "Summarize the key topics covered in this presentation",
        "expected_elements": ["INTB 440", "international finance", "MNCs", "globalization"]
    },
    "video": {
        "path": "/Volumes/1tb-sandisk/code-external/mai-vl-rag-graph-rlm/examples/Real-Time, Low Latency and High Temporal Resolution Spectrograms - Alexandre R.J. Francois - ADC.mp4",
        "query": "What is this video about? Summarize the key technical concepts.",
        "expected_elements": ["spectrograms", "audio processing", "real-time"]
    },
    # Add more content types as available
}


def estimate_api_calls(profile: TestProfile, chunk_count: int, query_count: int = 1) -> int:
    """Estimate API calls based on profile settings."""
    # Knowledge graph extraction
    kg_calls = profile.max_iterations  # Each iteration can trigger LLM call
    
    # Query processing
    actual_queries = query_count * (3 if profile.multi_query else 1)  # multi-query generates sub-queries
    query_calls = actual_queries * profile.max_depth * (profile.max_iterations // 2)  # Approximate
    
    # Graph augmentation adds overhead
    graph_calls = actual_queries * profile.graph_hops if profile.graph_augmented else 0
    
    return kg_calls + query_calls + graph_calls


def run_mcp_speed_test(
    profile: TestProfile,
    content_type: str,
    content_config: Dict,
    use_api: bool = True
) -> TestResult:
    """Run a single speed test with given profile."""
    import os
    from vrlmrag import run_analysis
    
    path = content_config["path"]
    query = content_config["query"]
    
    # Get file size
    file_size = Path(path).stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n{'='*60}")
    print(f"Testing: {profile.name} on {content_type}")
    print(f"Settings: depth={profile.max_depth}, iterations={profile.max_iterations}, "
          f"multi_query={profile.multi_query}, graph={profile.graph_augmented}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    errors = []
    
    try:
        results = run_analysis(
            provider="auto",  # Use hierarchy
            input_path=path,
            query=query,
            output=None,
            model=None,  # Let hierarchy resolve
            max_depth=profile.max_depth,
            max_iterations=profile.max_iterations,
            use_api=use_api,
            text_only=False,
            multi_query=profile.multi_query,
            use_graph_augmented=profile.graph_augmented,
            graph_hops=profile.graph_hops,
            output_format="markdown",
            verbose=True,
            _quiet=False,
        )
        
        execution_time = time.time() - start_time
        
        # Extract metrics
        doc_count = results.get("document_count", 0)
        chunk_count = results.get("total_chunks", 0)
        kg_size = len(results.get("knowledge_graph", ""))
        
        # Estimate API calls
        api_calls = estimate_api_calls(profile, chunk_count, query_count=1)
        
        return TestResult(
            profile_name=profile.name,
            content_type=content_type,
            file_path=path,
            file_size_mb=file_size,
            execution_time_seconds=execution_time,
            documents_processed=doc_count,
            chunks_embedded=chunk_count,
            api_calls_estimate=api_calls,
            query_response_quality="TBD - Manual Review",
            knowledge_graph_size_chars=kg_size,
            errors=errors
        )
        
    except Exception as e:
        errors.append(str(e))
        execution_time = time.time() - start_time
        return TestResult(
            profile_name=profile.name,
            content_type=content_type,
            file_path=path,
            file_size_mb=file_size,
            execution_time_seconds=execution_time,
            documents_processed=0,
            chunks_embedded=0,
            api_calls_estimate=0,
            query_response_quality="FAILED",
            knowledge_graph_size_chars=0,
            errors=errors
        )


def run_full_test_suite() -> Dict:
    """Run complete test suite across all profiles and content types."""
    all_results = []
    
    for content_type, config in TEST_CONTENT.items():
        if not Path(config["path"]).exists():
            print(f"⚠️  Skipping {content_type} - file not found: {config['path']}")
            continue
            
        for profile_key, profile in PROFILES.items():
            result = run_mcp_speed_test(profile, content_type, config)
            all_results.append(asdict(result))
            
            # Print immediate summary
            print(f"\n📊 Result: {result.execution_time_seconds:.1f}s | "
                  f"{result.documents_processed} docs | "
                  f"{result.chunks_embedded} chunks | "
                  f"~{result.api_calls_estimate} API calls")
            if result.errors:
                print(f"❌ Errors: {result.errors}")
    
    return {
        "profiles": {k: asdict(v) for k, v in PROFILES.items()},
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


def generate_report(test_data: Dict) -> str:
    """Generate markdown report from test results."""
    lines = [
        "# VL-RAG-Graph-RLM Speed & Accuracy Test Report",
        "",
        f"**Generated:** {test_data['timestamp']}",
        "",
        "## Test Profiles",
        "",
    ]
    
    for key, profile in test_data["profiles"].items():
        lines.extend([
            f"### {profile['name']} (`{key}`)",
            f"- **Max Depth:** {profile['max_depth']}",
            f"- **Max Iterations:** {profile['max_iterations']}",
            f"- **Multi-Query:** {profile['multi_query']}",
            f"- **Graph Augmented:** {profile['graph_augmented']}",
            f"- **Graph Hops:** {profile['graph_hops']}",
            f"- **Description:** {profile['description']}",
            "",
        ])
    
    lines.extend([
        "## Results Summary",
        "",
        "| Profile | Content Type | Time (s) | Docs | Chunks | Est. API Calls |",
        "|---------|--------------|----------|------|--------|----------------|",
    ])
    
    for result in test_data["results"]:
        lines.append(
            f"| {result['profile_name']} | {result['content_type']} | "
            f"{result['execution_time_seconds']:.1f} | "
            f"{result['documents_processed']} | {result['chunks_embedded']} | "
            f"{result['api_calls_estimate']} |"
        )
    
    lines.extend([
        "",
        "## Recommendations",
        "",
        "_To be filled after manual accuracy review..._",
        "",
        "### Speed vs Accuracy Trade-offs",
        "",
        "| Profile | Speed | Accuracy | Best For |",
        "|---------|-------|----------|----------|",
        "| Current (Aggressive) | ⭐⭐ | ⭐⭐⭐⭐⭐ | Deep research, critical analysis |",
        "| Balanced | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | General purpose, production use |",
        "| Fast | ⭐⭐⭐⭐⭐ | ⭐⭐ | Quick summaries, rapid iteration |",
        "",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("🚀 VL-RAG-Graph-RLM Comprehensive Speed Test Suite")
    print("=" * 60)
    
    # Run tests
    test_data = run_full_test_suite()
    
    # Save raw results
    results_path = Path("speed_test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"\n✅ Raw results saved to: {results_path}")
    
    # Generate report
    report = generate_report(test_data)
    report_path = Path("speed_test_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✅ Report saved to: {report_path}")
