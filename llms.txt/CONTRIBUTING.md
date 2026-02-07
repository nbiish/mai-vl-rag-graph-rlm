# Contributing — VL-RAG-Graph-RLM

## Overview

VL-RAG-Graph-RLM is a unified multimodal document analysis framework combining vision-language embeddings, retrieval-augmented generation, knowledge graph extraction, and recursive language model reasoning.

Version: 0.1.0
License: See project LICENSE file

## Quick Start for Contributors

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes following the guidelines below
4. Test thoroughly (see Testing Guidelines)
5. Submit a pull request with a clear description

## Adding a New Provider

Providers are LLM services that integrate with the VL-RAG-Graph-RLM pipeline. The project currently supports 17 providers, and adding more follows a consistent pattern.

### Step 1: Create Provider Template

Create a new file in `templates/` following the naming pattern `provider_<name>.py`:

```python
#!/usr/bin/env python3
"""
<Provider Name> Template — Full VL-RAG-Graph-RLM Pipeline

Demonstrates all six pillars:
  1. VL: Qwen3-VL multimodal embeddings (text + images)
  2. RAG: Hybrid search (dense + keyword) with RRF fusion
  3. Reranker: Qwen3-VL cross-attention reranking
  4. Graph: Knowledge graph extraction via RLM
  5. RLM: Recursive Language Model with REPL
  6. Pipeline: Markdown report generation

Recommended Models:
    - model-1: Description
    - model-2: Description

Environment:
    export <PROVIDER>_API_KEY=your_key_here
    # Optional: export <PROVIDER>_MODEL=model-name
    # Optional: export <PROVIDER>_RECURSIVE_MODEL=cheaper-model

Get API Key: https://provider.example.com
Docs: https://docs.provider.example.com
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from vl_rag_graph_rlm import VLRAGGraphRLM, create_pipeline


def example_full_pipeline(input_path: str, query: str = "What are the main topics covered?"):
    """Full 6-pillar pipeline: VL embeddings -> RAG -> reranker -> graph -> RLM -> report."""
    pipeline = create_pipeline(
        llm_provider="<provider-slug>",
        embedding_model="Qwen/Qwen3-VL-Embedding-2B",
        use_reranker=True,
    )

    path = Path(input_path)
    if path.suffix.lower() == ".pptx":
        pipeline.add_pptx(str(path), extract_images=True)
    elif path.suffix.lower() == ".pdf":
        pipeline.add_pdf(str(path), extract_images=True)
    else:
        pipeline.add_text(path.read_text())

    result = pipeline.query(query)
    print(f"Answer: {result.answer[:500]}...")
    print(f"Sources: {len(result.sources)}, Time: {result.execution_time:.2f}s")


def example_manual_pipeline():
    """Manual pipeline showing each pillar explicitly."""
    try:
        from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder, create_qwen3vl_reranker
        from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore
        import torch
        has_vl = True
    except ImportError:
        has_vl = False

    # --- Pillar 1: VL Embeddings ---
    if has_vl:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedder = create_qwen3vl_embedder(device=device)
        store = MultimodalVectorStore(embedding_provider=embedder)
        store.add_text("Your text here.", metadata={"source": "example"})

        # --- Pillar 2: RAG — Dense search ---
        query = "Your query here?"
        dense_results = store.search(query, top_k=10)

        # --- Pillar 3: Reranker ---
        reranker = create_qwen3vl_reranker(device=device)
        docs = [{"text": store.get(r.id).content} for r in dense_results if store.get(r.id)]
        reranked = reranker.rerank(query={"text": query}, documents=docs)
        context = "\n".join([docs[idx]["text"] for idx, _ in reranked[:3]])
    else:
        context = "Fallback text without embeddings."
        query = "Your query here?"

    # --- Pillar 4: Graph ---
    rlm = VLRAGGraphRLM(provider="<provider-slug>", temperature=0.0)
    kg = rlm.completion("Extract key entities and relationships.", context)
    print(f"Knowledge graph: {kg.response[:200]}...")

    # --- Pillar 5: RLM ---
    result = rlm.completion(query, context)
    print(f"Answer: {result.response[:300]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="<Provider Name> — Full VL-RAG-Graph-RLM Pipeline")
    parser.add_argument("--input", "-i", help="Document to process (PPTX, PDF, TXT, MD)")
    parser.add_argument("--query", "-q", default="What are the main topics covered?")
    parser.add_argument("--manual", action="store_true", help="Run manual pipeline example")
    args = parser.parse_args()

    if not os.getenv("<PROVIDER>_API_KEY"):
        print("Error: <PROVIDER>_API_KEY not set")
        print("Get your API key from: https://provider.example.com")
        return

    if args.manual:
        example_manual_pipeline()
    elif args.input:
        example_full_pipeline(args.input, args.query)
    else:
        print("Usage:")
        print("  python provider_<name>.py --input document.pptx")
        print("  python provider_<name>.py --manual")


if __name__ == "__main__":
    main()
```

### Step 2: Register Provider in CLI

Add your provider to `SUPPORTED_PROVIDERS` in `src/vrlmrag.py`:

```python
SUPPORTED_PROVIDERS: Dict[str, Dict[str, Any]] = {
    # ... existing providers ...
    "<provider-slug>": {
        "env_key": "<PROVIDER>_API_KEY",
        "url": "https://provider.example.com",
        "description": "<Provider Name> — Brief description (context size, notes)",
        "context_budget": 32000,  # Adjust based on provider limits
    },
}
```

### Step 3: Implement Client (if needed)

If your provider requires a custom client (not OpenAI-compatible or Anthropic-compatible):

1. Create a client in `src/vl_rag_graph_rlm/clients/`
2. Follow the pattern in `openai_compatible.py` or `anthropic.py`
3. Implement the completion interface matching the project's types

### Step 4: Update Documentation

1. **PRD.md**: Add to the "Supported Providers" table with default model and context window
2. **ARCHITECTURE.md**: Add to the "Templates" table
3. **RULES.md**: Add any provider-specific rules if applicable
4. **TODO.md**: Track completion of the provider integration

### Step 5: Test

```bash
# Test the template directly
python templates/provider_<name>.py --manual

# Test via CLI
vrlmrag --provider <provider-slug> examples/document.pptx

# Verify provider listing
vrlmrag --list-providers
```

## The Six-Pillar Requirement

Every provider template **must** demonstrate all six pillars of the architecture:

| Pillar | Description | Implementation |
|--------|-------------|----------------|
| **1. VL** | Vision-Language Embeddings | Use `create_qwen3vl_embedder()` from `vl_rag_graph_rlm.rag` |
| **2. RAG** | Retrieval-Augmented Generation | Use `MultimodalVectorStore` with dense + keyword search |
| **3. Reranker** | Multi-Stage Reranking | Use `create_qwen3vl_reranker()` with RRF fusion |
| **4. Graph** | Knowledge Graph Extraction | Use `VLRAGGraphRLM` to extract entities/relationships |
| **5. RLM** | Recursive Language Model | Use `VLRAGGraphRLM` with REPL for reasoning |
| **6. Pipeline** | Unified API | Use `create_pipeline()` or implement the full manual flow |

**Never** create a template that only calls `rlm.completion()` without RAG retrieval. The full pipeline is essential.

## Device Detection Patterns

When working with Qwen3-VL models, follow these device detection patterns:

```python
# For templates (Apple Silicon support):
device = "mps" if torch.backends.mps.is_available() else "cpu"

# For core library (CUDA focus):
device = device or ("cuda" if torch.cuda.is_available() else "cpu")

# Best practice (full coverage):
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
```

## Qwen3-VL Fallback Pattern

Always wrap Qwen3-VL imports in try/except for graceful degradation:

```python
try:
    from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder, create_qwen3vl_reranker
    HAS_QWEN3VL = True
except ImportError:
    HAS_QWEN3VL = False

# Later in code:
if HAS_QWEN3VL:
    # Use Qwen3-VL embedder/reranker
else:
    # Fall back to text-only reranking
```

## Testing Guidelines

### Unit Tests

- Test core functions in isolation
- Mock external API calls
- Verify correct error handling

### Integration Tests

- Test full pipeline with sample documents
- Verify all six pillars are exercised
- Test fallback behavior when Qwen3-VL is unavailable

### Provider Tests

- Test API connectivity with actual provider
- Verify model defaults work
- Test environment variable loading

### Test Commands

```bash
# Run existing tests (if test suite exists)
pytest tests/

# Manual smoke test
python templates/provider_<name>.py --manual

# CLI smoke test
vrlmrag --provider <provider-slug> examples/test.txt -q "Test query"
```

## Code Style

- Follow PEP 8 for Python code
- Use descriptive names matching six-pillar vocabulary
- Document every public function with docstrings (Args, Returns, Example)
- Prefix provider-specific code with provider name

## API Key Security

- Never hardcode API keys in source files
- Always load from environment variables via `python-dotenv`
- Use `{PROVIDER}_API_KEY` naming convention
- Document required env vars in template docstring

## Submitting Changes

1. Ensure all tests pass
2. Update relevant documentation (PRD.md, ARCHITECTURE.md, RULES.md, TODO.md)
3. Add clear commit messages referencing issues
4. PR should include:
   - Description of changes
   - Testing performed
   - Documentation updates
   - Screenshots if UI changes

## Documentation References

- **PRD.md**: Product requirements, provider specs, short-term goals
- **RULES.md**: Coding standards, always/never patterns, device detection
- **ARCHITECTURE.md**: System diagram, component map, CLI reference
- **TODO.md**: Roadmap, completed items, version planning
- **CONTRIBUTING.md**: This file

## Getting Help

- Check existing provider templates for reference patterns
- Review `llms.txt/RULES.md` for coding standards
- Examine `src/vrlmrag.py` for CLI integration
- Open an issue for questions or bugs
