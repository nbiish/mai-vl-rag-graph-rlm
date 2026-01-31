"""Example: Async usage with multiple providers."""

import asyncio
import os
from rlm import RLM


async def process_with_provider(provider: str, model: str, query: str, context: str):
    """Process a query with a specific provider."""
    api_key_env = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(api_key_env) or os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return f"Skipping {provider}: No API key found"

    rlm = RLM(
        provider=provider,
        model=model,
        api_key=api_key,
        max_depth=2,
        max_iterations=5
    )

    result = await rlm.acompletion(query, context)
    return {
        "provider": provider,
        "model": model,
        "answer": result.response,
        "time": result.execution_time,
        "tokens": result.usage_summary.to_dict()
    }


async def main():
    """Run async examples."""
    context = """
    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. Key types include:

    1. Supervised Learning: Uses labeled training data
    2. Unsupervised Learning: Finds patterns in unlabeled data
    3. Reinforcement Learning: Learns through trial and error with rewards

    Popular algorithms include neural networks, decision trees, and support vector machines.
    """

    query = "What are the main types of machine learning?"

    # Process with multiple providers concurrently
    tasks = [
        process_with_provider("openrouter", "openai/gpt-4o-mini", query, context),
        # Add more providers as needed:
        # process_with_provider("zenmux", "gpt-4o", query, context),
        # process_with_provider("zai", "claude-3-opus", query, context),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, dict):
            print(f"\nProvider: {result['provider']}")
            print(f"Model: {result['model']}")
            print(f"Answer: {result['answer']}")
            print(f"Time: {result['time']:.2f}s")
        else:
            print(f"\nError: {result}")


if __name__ == "__main__":
    asyncio.run(main())
