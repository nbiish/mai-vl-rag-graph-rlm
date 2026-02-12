"""Lightweight FlashRank-based reranker adapter.

Provides a drop-in replacement for Qwen3VLRerankerProvider using
FlashRank's ONNX cross-encoder models (~4-34 MB) instead of a
full 2B-parameter VL model (~4.6 GB).

The adapter matches the same ``rerank()`` interface so it can be
swapped in everywhere the Qwen3-VL reranker was used without
changing any call sites.

Requirements::

    pip install flashrank          # pairwise cross-encoder (default)
    pip install flashrank[listwise] # optional LLM-based listwise

Typical usage::

    >>> from vl_rag_graph_rlm.rag.flashrank_reranker import (
    ...     FlashRankRerankerProvider,
    ...     create_flashrank_reranker,
    ... )
    >>> reranker = create_flashrank_reranker()        # ~34 MB model
    >>> scores = reranker.rerank(
    ...     query={"text": "What causes climate change?"},
    ...     documents=[
    ...         {"text": "Greenhouse gases trap heat in the atmosphere."},
    ...         {"text": "The capital of France is Paris."},
    ...     ],
    ... )
    >>> # scores: [(0, 0.92), (1, 0.03)]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Available model tiers (name → approximate disk/RAM footprint)
FLASHRANK_MODELS = {
    "nano": "ms-marco-TinyBERT-L-2-v2",       # ~4 MB
    "small": "ms-marco-MiniLM-L-12-v2",        # ~34 MB  (best accuracy/size)
    "medium": "rank-T5-flan",                   # ~110 MB
    "multilingual": "ms-marco-MultiBERT-L-12",  # ~150 MB (100+ languages)
}

DEFAULT_MODEL = FLASHRANK_MODELS["small"]
DEFAULT_MAX_LENGTH = 512


class FlashRankRerankerProvider:
    """Lightweight cross-encoder reranker backed by FlashRank/ONNX.

    Drop-in replacement for ``Qwen3VLRerankerProvider``.  The
    ``rerank()`` method accepts the same ``query`` / ``documents``
    dict format and returns ``List[Tuple[int, float]]`` sorted by
    descending relevance score.

    Because FlashRank is text-only, multimodal keys (``image``,
    ``video``) in documents are silently ignored — the ``text`` field
    is used for scoring.  This is acceptable because all multimodal
    content in the vector store already has a text representation
    (captions, transcripts, OCR text, etc.).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
        cache_dir: Optional[str] = None,
    ) -> None:
        from flashrank import Ranker

        logger.info(
            "Loading FlashRank reranker: %s (max_length=%d)", model_name, max_length
        )
        kwargs: Dict[str, Any] = {"model_name": model_name, "max_length": max_length}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        self.ranker = Ranker(**kwargs)
        self.model_name = model_name
        logger.info("FlashRank reranker loaded successfully")

    # ── public API (matches Qwen3VLRerankerProvider) ──────────────

    def rerank(
        self,
        query: Dict[str, Any],
        documents: List[Dict[str, Any]],
        instruction: Optional[str] = None,
        fps: float = 1.0,
        max_frames: int = 64,
    ) -> List[Tuple[int, float]]:
        """Rerank documents by relevance to *query*.

        Args:
            query: Dict with at least a ``"text"`` key.
            documents: List of dicts, each with at least a ``"text"`` key.
            instruction: Ignored (kept for interface compat).
            fps: Ignored (kept for interface compat).
            max_frames: Ignored (kept for interface compat).

        Returns:
            ``[(doc_index, score), ...]`` sorted descending by score.
        """
        from flashrank import RerankRequest

        query_text = query.get("text", "")
        if not query_text:
            logger.warning("Empty query text — returning original order")
            return [(i, 0.0) for i in range(len(documents))]

        passages = []
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            if not text:
                text = "(empty)"
            passages.append({"id": i, "text": text, "meta": doc.get("meta", {})})

        if not passages:
            return []

        request = RerankRequest(query=query_text, passages=passages)
        results = self.ranker.rerank(request)

        scored: List[Tuple[int, float]] = []
        for r in results:
            idx = r["id"] if isinstance(r["id"], int) else int(r["id"])
            scored.append((idx, float(r["score"])))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ── factory function ──────────────────────────────────────────────

def create_flashrank_reranker(
    model_name: str = DEFAULT_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
    cache_dir: Optional[str] = None,
    **_kwargs: Any,
) -> FlashRankRerankerProvider:
    """Create a lightweight FlashRank reranker.

    Extra ``**kwargs`` (e.g. ``device``) are accepted and silently
    ignored so callers written for the Qwen3-VL reranker don't need
    to change their keyword arguments.
    """
    return FlashRankRerankerProvider(
        model_name=model_name,
        max_length=max_length,
        cache_dir=cache_dir,
    )
