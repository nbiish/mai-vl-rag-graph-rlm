#!/usr/bin/env python3
"""
ZenMux Template - Chinese AI Models

Recommended Models:
    - ernie-5.0-thinking-preview: Best for reasoning
    - dubao-seed-1.8: Best for coding
    - glm-4.7-flash: Fast, cheap responses

Environment:
    export ZENMUX_API_KEY=your_key_here
    # Optional: export ZENMUX_MODEL=ernie-5.0-thinking-preview
    # Optional: export ZENMUX_RECURSIVE_MODEL=glm-4.7-flash

Get API Key: https://zenmux.ai
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("ZENMUX_API_KEY"):
        print("Error: ZENMUX_API_KEY not set")
        print("Get your API key from: https://zenmux.ai")
        return

    rlm = VLRAGGraphRLM(
        provider="zenmux",
        temperature=0.0,
    )

    query = "Explain quantum computing in Chinese and English."

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    main()
