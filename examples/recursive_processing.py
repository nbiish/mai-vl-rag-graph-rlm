"""Example: Custom REPL usage and recursive processing."""

import os
from rlm import RLM


# Example: Recursive document processing
print("=" * 60)
print("Example: Recursive Document Analysis")
print("=" * 60)

# Create a sample document with nested structure
document = """
REPORT: Company Analysis 2024
=============================

EXECUTIVE SUMMARY
-----------------
This report analyzes the top 5 tech companies by market cap.

SECTION 1: Apple Inc.
---------------------
Founded: 1976
CEO: Tim Cook
Market Cap: $3.5T
Revenue: $383B (2023)
Key Products: iPhone, Mac, iPad, Services

SECTION 2: Microsoft Corporation
--------------------------------
Founded: 1975
CEO: Satya Nadella
Market Cap: $3.2T
Revenue: $211B (2023)
Key Products: Windows, Office, Azure, Xbox

SECTION 3: NVIDIA Corporation
-----------------------------
Founded: 1993
CEO: Jensen Huang
Market Cap: $2.8T
Revenue: $60B (2023)
Key Products: GPUs, AI Chips, Data Center Solutions

SECTION 4: Alphabet Inc. (Google)
----------------------------------
Founded: 1998
CEO: Sundar Pichai
Market Cap: $2.1T
Revenue: $307B (2023)
Key Products: Search, Ads, Cloud, YouTube

SECTION 5: Amazon.com Inc.
--------------------------
Founded: 1994
CEO: Andy Jassy
Market Cap: $1.9T
Revenue: $574B (2023)
Key Products: E-commerce, AWS, Prime, Advertising

CONCLUSION
----------
These 5 companies represent the largest tech companies by market capitalization.
"""

rlm = RLM(
    provider="openrouter",
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_depth=3,
    max_iterations=15
)

print(f"Document size: {len(document):,} characters")
print("\nProcessing...")

# The RLM will automatically:
# 1. Parse the document structure
# 2. Use recursive_llm() for each section
# 3. Synthesize a final answer
# result = rlm.completion(
#     query="Analyze each company and identify which has the highest revenue-to-market-cap ratio",
#     context=document
# )
#
# print(f"\nAnswer: {result.response}")
# print(f"\nStats: {rlm.stats}")

print("\n(Uncomment the code above to run the analysis)")


# Example: Using different models for root and recursive calls
print("\n" + "=" * 60)
print("Example: Model Routing (Smart Cost Optimization)")
print("=" * 60)

rlm_routing = RLM(
    provider="openrouter",
    model="anthropic/claude-3.5-sonnet",  # Strong model for root
    recursive_model="openai/gpt-4o-mini",  # Cheaper model for recursive calls
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_depth=4,
    max_iterations=20
)

print("Root model: anthropic/claude-3.5-sonnet (strong reasoning)")
print("Recursive model: openai/gpt-4o-mini (cost-effective)")
print("\nThis setup optimizes cost while maintaining quality at the root level.")


# Example: Direct client usage (bypass RLM for simple calls)
print("\n" + "=" * 60)
print("Example: Direct Client Usage")
print("=" * 60)

from rlm import get_client

# Get any client directly
client = get_client(
    provider="openrouter",
    model_name="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Simple completion
# response = client.completion("What is 2+2?")
# print(f"Response: {response}")

print("Use get_client() for direct API access without RLM recursion")
