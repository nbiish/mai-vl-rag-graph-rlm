#!/usr/bin/env python3
"""
Generic OpenAI-Compatible Template - Custom Endpoints

For any provider with OpenAI-compatible API:
- Groq, Mistral, Fireworks, Together, DeepSeek, etc.
- Self-hosted models (vLLM, TGI)
- Custom proxies

Required Environment Variables:
    export OPENAI_COMPATIBLE_API_KEY=your_api_key_here
    export OPENAI_COMPATIBLE_BASE_URL=https://api.example.com/v1
    export OPENAI_COMPATIBLE_MODEL=your-model-name

Or pass directly:
    client = GenericOpenAIClient(
        api_key="your-key",
        base_url="https://api.example.com/v1",
        model_name="llama-3.1-70b"
    )
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vl_rag_graph_rlm.clients import GenericOpenAIClient


def main():
    # Check required env vars
    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

    if not all([api_key, base_url, model_name]):
        print("Error: Missing required environment variables")
        print("Please set:")
        print("  export OPENAI_COMPATIBLE_API_KEY=your_api_key_here")
        print("  export OPENAI_COMPATIBLE_BASE_URL=https://api.example.com/v1")
        print("  export OPENAI_COMPATIBLE_MODEL=your-model-name")
        return

    # Create client directly
    client = GenericOpenAIClient(
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
