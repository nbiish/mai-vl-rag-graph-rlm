"""Example: Basic usage with different providers."""

import os
from rlm import RLM

# Example 1: OpenRouter
print("=" * 50)
print("Example 1: OpenRouter")
print("=" * 50)

rlm = RLM(
    provider="openrouter",
    model="anthropic/claude-3.5-sonnet",  # or openai/gpt-4o, etc.
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_depth=3,
    max_iterations=10
)

context = """
The Recursive Language Model (RLM) is a new architecture for processing unbounded context.
Unlike traditional LLMs that are limited by context window size, RLMs can recursively process
arbitrarily long documents by breaking them into chunks and processing them hierarchically.

Key features:
1. Recursive decomposition of long contexts
2. Python REPL for code execution
3. Automatic summarization at each level
4. Final answer synthesis

The RLM uses a Python REPL environment where the model can write code to:
- Search through documents
- Extract relevant information
- Call itself recursively on sub-contexts
- Return final answers via FINAL() or FINAL_VAR()
"""

result = rlm.completion(
    query="What are the key features of RLM?",
    context=context
)

print(f"Answer: {result.response}")
print(f"Provider: {result.provider}")
print(f"Model: {result.model}")
print(f"Execution time: {result.execution_time:.2f}s")
print(f"Usage: {result.usage_summary.to_dict()}")


# Example 2: ZenMux (if you have an API key)
print("\n" + "=" * 50)
print("Example 2: ZenMux")
print("=" * 50)

# Uncomment when you have ZenMux credentials:
# rlm_zenmux = RLM(
#     provider="zenmux",
#     model="gpt-4o",
#     api_key=os.getenv("ZENMUX_API_KEY"),
# )
# result = rlm_zenmux.completion("Summarize this", context="Your long context here...")
# print(result.response)

print("Uncomment the code above when you have ZenMux credentials")


# Example 3: z.ai (if you have an API key)
print("\n" + "=" * 50)
print("Example 3: z.ai")
print("=" * 50)

# Uncomment when you have z.ai credentials:
# rlm_zai = RLM(
#     provider="zai",
#     model="claude-3-opus",
#     api_key=os.getenv("ZAI_API_KEY"),
# )
# result = rlm_zai.completion("Analyze this", context="Your context here...")
# print(result.response)

print("Uncomment the code above when you have z.ai credentials")


# Example 4: Using with large documents
print("\n" + "=" * 50)
print("Example 4: Large Document Processing")
print("=" * 50)

# Generate a large document
large_doc = "\n".join([f"Section {i}: This is section {i} with important information. " * 20 for i in range(100)])

print(f"Document size: {len(large_doc):,} characters")

rlm_large = RLM(
    provider="openrouter",
    model="openai/gpt-4o-mini",  # Cheaper for large docs
    recursive_model="openai/gpt-4o-mini",
    max_depth=5,
    max_iterations=20
)

# Process the document
# result = rlm_large.completion(
#     query="Find all section numbers mentioned in the document",
#     context=large_doc
# )
# print(f"Found sections: {result.response}")

print("Uncomment to process the large document")
