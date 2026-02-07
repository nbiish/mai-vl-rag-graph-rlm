#!/usr/bin/env python3
"""
OpenAI Template - GPT-4o Series

Recommended Models:
    - gpt-4o-mini: Cheap, fast, capable
    - gpt-4o: Most capable
    - gpt-4o-latest: Latest version

Environment:
    export OPENAI_API_KEY=your_key_here
    # Optional: export OPENAI_BASE_URL=https://api.openai.com/v1
    # Optional: export OPENAI_MODEL=gpt-4o-mini
    # Optional: export OPENAI_RECURSIVE_MODEL=gpt-4o-mini

Get API Key: https://platform.openai.com
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("Get your API key from: https://platform.openai.com")
        return

    rlm = VLRAGGraphRLM(
        provider="openai",
        temperature=0.0,
    )

    query = "Summarize the key principles of clean code."

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    main()
