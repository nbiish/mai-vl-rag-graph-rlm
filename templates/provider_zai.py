#!/usr/bin/env python3
"""
z.ai (Zhipu AI) Template - GLM Series

Recommended Models:
    - glm-4.7: Flagship model, excellent reasoning
    - glm-4.7-coding: Optimized for code generation
    - glm-4.7-flash: Fast, cost-effective

Note: z.ai also offers flat-rate Coding Plans ($3-15/mo) via their native API.

Environment:
    export ZAI_API_KEY=your_key_here
    # Optional: export ZAI_MODEL=glm-4.7
    # Optional: export ZAI_RECURSIVE_MODEL=glm-4.7-flash

Get API Key: https://open.bigmodel.cn
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm import VLRAGGraphRLM


def main():
    if not os.getenv("ZAI_API_KEY"):
        print("Error: ZAI_API_KEY not set")
        print("Get your API key from: https://open.bigmodel.cn")
        return

    rlm = VLRAGGraphRLM(
        provider="zai",
        temperature=0.0,
    )

    query = "Explain the architecture of transformers in Chinese."

    print(f"Query: {query}")
    print(f"Using model: {rlm.model}")
    print("-" * 50)

    result = rlm.completion(query)
    print(f"\nResponse:\n{result.response}")


if __name__ == "__main__":
    main()
