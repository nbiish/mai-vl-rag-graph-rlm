#!/usr/bin/env python3
"""
Azure OpenAI Template - Microsoft Azure Deployment

Required Environment Variables:
    export AZURE_OPENAI_API_KEY=your_azure_key_here
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_API_VERSION=2024-02-01  # Optional
    # Optional: export AZURE_OPENAI_MODEL=gpt-4o

Get API Key: Azure Portal > OpenAI Service > Keys and Endpoint
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Error: AZURE_OPENAI_API_KEY not set")
        print("Get your key from Azure Portal > OpenAI Service > Keys and Endpoint")
        return

    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Error: AZURE_OPENAI_ENDPOINT not set")
        print("Format: https://your-resource.openai.azure.com/")
        return

    rlm = VLRAGGraphRLM(
        provider="azure_openai",
        temperature=0.0,
    )

    query = "Summarize the benefits of cloud computing."

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    main()
