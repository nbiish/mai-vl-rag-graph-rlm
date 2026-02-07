#!/usr/bin/env python3
"""
Mistral AI Template - European LLMs

Recommended Models:
    - mistral-large-latest: Most capable
    - mistral-medium: Balanced
    - mistral-small: Fast, cheap

Environment:
    export MISTRAL_API_KEY=your_key_here
    # Optional: export MISTRAL_MODEL=mistral-large-latest

Get API Key: https://console.mistral.ai
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY not set")
        print("Get your API key from: https://console.mistral.ai")
        return

    rlm = VLRAGGraphRLM(
        provider="mistral",
        temperature=0.0,
    )

    query = "Translate this text to French: Hello, how are you today?"

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    main()
