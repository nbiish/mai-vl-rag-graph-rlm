#!/usr/bin/env python3
"""
Together AI Template - Open Source Models

Recommended Models:
    - meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo: Fast Llama 3.1
    - mistralai/Mixtral-8x22B-Instruct-v0.1: Mixtral 8x22B

Environment:
    export TOGETHER_API_KEY=your_key_here
    # Optional: export TOGETHER_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo

Get API Key: https://api.together.ai
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("TOGETHER_API_KEY"):
        print("Error: TOGETHER_API_KEY not set")
        print("Get your API key from: https://api.together.ai")
        return

    rlm = VLRAGGraphRLM(
        provider="together",
        temperature=0.0,
    )

    query = "Write a function to reverse a linked list."

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    main()
