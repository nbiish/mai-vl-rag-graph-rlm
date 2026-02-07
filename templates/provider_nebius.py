#!/usr/bin/env python3
"""
Nebius Token Factory Template - GLM-4.7 & More

Recommended Models:
    - z-ai/GLM-4.7: Z.AI's flagship for agentic coding and reasoning
    - deepseek-ai/DeepSeek-R1-0528: DeepSeek R1 reasoning
    - meta-llama/Meta-Llama-3.1-70B-Instruct: Llama 3.1 70B

Environment:
    export NEBIUS_API_KEY=your_key_here
    # Optional: export NEBIUS_MODEL=z-ai/GLM-4.7

Get API Key: https://tokenfactory.nebius.com
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("NEBIUS_API_KEY"):
        print("Error: NEBIUS_API_KEY not set")
        print("Get your API key from: https://tokenfactory.nebius.com")
        return

    # Initialize with Nebius Token Factory
    # Uses z-ai/GLM-4.7 by default
    rlm = VLRAGGraphRLM(
        provider="nebius",
        # model="z-ai/GLM-4.7",  # Optional: defaults to z-ai/GLM-4.7
        temperature=0.0,
    )

    query = "Write a Python function to calculate fibonacci numbers recursively."

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")
    print(f"\nExecution time: {result.execution_time:.2f}s")


if __name__ == "__main__":
    main()
