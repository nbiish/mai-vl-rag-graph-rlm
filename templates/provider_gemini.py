#!/usr/bin/env python3
"""
Google Gemini Template - Multimodal Capable

Recommended Models:
    - gemini-1.5-flash: Fast, cheap
    - gemini-1.5-pro: Most capable
    - gemini-2.0-flash: Latest fast model

Environment:
    export GOOGLE_API_KEY=your_key_here
    # Optional: export GOOGLE_MODEL=gemini-1.5-flash
    # Optional: export GOOGLE_RECURSIVE_MODEL=gemini-1.5-flash

Get API Key: https://makersuite.google.com/app/apikey
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return

    rlm = VLRAGGraphRLM(
        provider="gemini",
        temperature=0.0,
    )

    query = "Explain machine learning concepts with analogies."

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    main()
