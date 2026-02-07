#!/usr/bin/env python3
"""
Anthropic Template - Claude 3.5 Series

Recommended Models:
    - claude-3-5-sonnet-20241022: Most capable
    - claude-3-5-haiku-20241022: Fast, cheap

Environment:
    export ANTHROPIC_API_KEY=your_key_here
    # Optional: export ANTHROPIC_BASE_URL=https://api.anthropic.com
    # Optional: export ANTHROPIC_MODEL=claude-3-5-haiku-20241022
    # Optional: export ANTHROPIC_RECURSIVE_MODEL=claude-3-5-haiku-20241022

Get API Key: https://console.anthropic.com
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        print("Get your API key from: https://console.anthropic.com")
        return

    rlm = VLRAGGraphRLM(
        provider="anthropic",
        temperature=0.0,
    )

    query = "Analyze this code for potential bugs and improvements."
    context = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query, context)
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    main()
