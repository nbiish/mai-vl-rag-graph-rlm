"""
Provider Templates — VL-RAG-Graph-RLM (Full 6-Pillar Architecture)

Every template demonstrates the complete pipeline:
  1. VL: Qwen3-VL multimodal embeddings (text + images)
  2. RAG: Hybrid search (dense + keyword) with RRF fusion
  3. Reranker: Qwen3-VL cross-attention reranking
  4. Graph: Knowledge graph extraction via RLM
  5. RLM: Recursive Language Model with REPL
  6. Pipeline: Markdown report generation

Usage:
    python templates/provider_sambanova.py --input document.pptx
    python templates/provider_nebius.py --input document.pptx --output report.md
    python templates/provider_openrouter.py --input doc.pdf --query "Summarize"
    python templates/provider_openai.py --manual

See llms.txt/ARCHITECTURE.md for the full system diagram.
"""
