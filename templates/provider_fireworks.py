#!/usr/bin/env python3
"""
Fireworks AI Template - Open Source Models

Recommended Models:
    - accounts/fireworks/models/llama-v3p1-70b-instruct: Llama 3.1 70B
    - accounts/fireworks/models/mixtral-8x22b-instruct: Mixtral 8x22B

Environment:
    export FIREWORKS_API_KEY=your_key_here
    # Optional: export FIREWORKS_MODEL=accounts/fireworks/models/llama-v3p1-70b-instruct

Get API Key: https://fireworks.ai
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY not set")
        print("Get your API key from: https://fireworks.ai")
        return

    rlm = VLRAGGraphRLM(
        provider="fireworks",
        temperature=0.0,
    )

    query = "Generate a creative story about AI and humanity."

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    main()
