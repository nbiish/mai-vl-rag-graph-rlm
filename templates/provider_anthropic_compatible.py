#!/usr/bin/env python3
"""
Generic Anthropic-Compatible Template - Custom Endpoints

For any provider with Anthropic-compatible API:
- Custom Claude proxies
- Self-hosted Anthropic-compatible models
- Third-party Claude providers

Required Environment Variables:
    export ANTHROPIC_COMPATIBLE_API_KEY=your_api_key_here
    export ANTHROPIC_COMPATIBLE_BASE_URL=https://api.example.com
    export ANTHROPIC_COMPATIBLE_MODEL=claude-model-name

Or pass directly:
    client = AnthropicCompatibleClient(
        api_key="your-key",
        base_url="https://api.example.com",
        model_name="claude-3-5-sonnet"
    )
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm.clients import AnthropicCompatibleClient


def main():
    # Check required env vars
    api_key = os.getenv("ANTHROPIC_COMPATIBLE_API_KEY")
    base_url = os.getenv("ANTHROPIC_COMPATIBLE_BASE_URL")
    model_name = os.getenv("ANTHROPIC_COMPATIBLE_MODEL")

    if not all([api_key, base_url, model_name]):
        print("Error: Missing required environment variables")
        print("Please set:")
        print("  export ANTHROPIC_COMPATIBLE_API_KEY=your_api_key_here")
        print("  export ANTHROPIC_COMPATIBLE_BASE_URL=https://api.example.com")
        print("  export ANTHROPIC_COMPATIBLE_MODEL=claude-3-5-sonnet")
        return

    # Create client directly
    client = AnthropicCompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name
    )

    query = "Hello, who are you?"

    print(f"Query: {query}")
    print(f"Using base_url: {base_url}")
    print(f"Using model: {model_name}")
    print("-" * 50)

    response = client.completion(query)
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
