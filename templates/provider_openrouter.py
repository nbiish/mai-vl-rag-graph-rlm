#!/usr/bin/env python3
"""
OpenRouter Template - Multi-Model Gateway

Recommended Models:
    - minimax/minimax-m2.1: Minimax 2.1, excellent reasoning
    - kimi/kimi-k2.5: Excellent reasoning, very cheap
    - z-ai/glm-4.7: Great for coding
    - solar-pro/solar-pro-3:free: Free tier
    - google/gemini-3-flash-preview: Fast with 1M context

Environment:
    export OPENROUTER_API_KEY=your_key_here
    # Optional: export OPENROUTER_MODEL=minimax/minimax-m2.1
    # Optional: export OPENROUTER_RECURSIVE_MODEL=solar-pro/solar-pro-3:free

Get API Key: https://openrouter.ai
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set")
        print("Get your API key from: https://openrouter.ai")
        return

    # Initialize with OpenRouter
    rlm = VLRAGGraphRLM(
        provider="openrouter",
        # model="minimax/minimax-m2.1",  # Optional: defaults to minimax/minimax-m2.1
        temperature=0.0,
    )

    query = "Compare the efficiency of different sorting algorithms."

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")
    print(f"\nExecution time: {result.execution_time:.2f}s")


if __name__ == "__main__":
    main()
