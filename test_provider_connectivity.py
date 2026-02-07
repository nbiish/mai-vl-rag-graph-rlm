#!/usr/bin/env python3
"""Quick connectivity test for providers with real API keys."""

import os
import sys
from pathlib import Path

# Load .env
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)

sys.path.insert(0, str(project_root / "src"))

from vl_rag_graph_rlm.clients import get_client

# Providers with real API keys in .env
PROVIDERS_TO_TEST = [
    "openrouter",
    "zenmux",
    "zai",
    "groq",
    "cerebras",
    "sambanova",
    "nebius",
]

print("=" * 70)
print("Provider Connectivity Test")
print("=" * 70)

results = {}
for provider in PROVIDERS_TO_TEST:
    print(f"\nTesting {provider}...", end=" ")
    try:
        client = get_client(provider)
        # Try a simple completion
        result = client.completion("Say 'OK' and nothing else.")
        if result and len(result) > 0:
            print(f"✓ CONNECTED (response: {result[:50]}...)")
            results[provider] = "✓"
        else:
            print("✗ Empty response")
            results[provider] = "✗"
    except Exception as e:
        print(f"✗ ERROR: {str(e)[:60]}")
        results[provider] = f"✗: {str(e)[:40]}"

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
for provider, status in results.items():
    print(f"  {provider:<15} {status}")

print("\n" + "=" * 70)
print("Token Limit Notes")
print("=" * 70)
print("""
SambaNova: 200K TPD free tier - requires 8K char context budget
Nebius:    No daily limits - can use 100K+ char context
Others:    Per-tier limits, typically generous
""")
