#!/usr/bin/env python3
"""
DeepSeek Template - DeepSeek-V3 and R1

Recommended Models:
    - deepseek-chat: DeepSeek-V3 general purpose
    - deepseek-reasoner: DeepSeek-R1 reasoning model

Environment:
    export DEEPSEEK_API_KEY=your_key_here
    # Optional: export DEEPSEEK_MODEL=deepseek-chat

Get API Key: https://platform.deepseek.com
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not set")
        print("Get your API key from: https://platform.deepseek.com")
        return

    rlm = VLRAGGraphRLM(
        provider="deepseek",
        temperature=0.0,
    )

    query = "Solve this step by step: What is 15% of 240?"

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    main()
