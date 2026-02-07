#!/usr/bin/env python3
"""
Groq Template - Ultra-Fast Inference

Recommended Models:
    - llama-3.1-70b-versatile: Fast, capable
    - llama-3.1-8b-instant: Fastest, cheapest
    - mixtral-8x7b-32768: Large context

Environment:
    export GROQ_API_KEY=your_key_here
    # Optional: export GROQ_MODEL=llama-3.1-70b-versatile

Get API Key: https://console.groq.com
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not set")
        print("Get your API key from: https://console.groq.com")
        return

    rlm = VLRAGGraphRLM(
        provider="groq",
        temperature=0.0,
    )

    query = "Generate a Python script to fetch data from an API."

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")
    print(f"\nExecution time: {result.execution_time:.2f}s (Ultra-fast!)")


if __name__ == "__main__":
    main()
