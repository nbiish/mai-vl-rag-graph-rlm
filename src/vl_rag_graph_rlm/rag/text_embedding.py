"""Text-only embedding provider using Qwen3-Embedding models.

Lightweight alternative to ``Qwen3VLEmbeddingProvider`` for text-only
content (no images, videos, or multimodal inputs).  Uses the Qwen3-Embedding
model family which is smaller and faster than the VL variants.

Default model: ``Qwen/Qwen3-Embedding-0.6B`` (~1.2 GB RAM).

The provider implements the same ``embed_text()`` interface as
``Qwen3VLEmbeddingProvider`` so it can be used as a drop-in replacement
with ``MultimodalVectorStore`` for text-only workflows.

Required env vars (from ``.env``)::

    # Optional — defaults shown
    VRLMRAG_TEXT_ONLY_MODEL=Qwen/Qwen3-Embedding-0.6B
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TEXT_ONLY_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _last_token_pool(
    last_hidden_states: Any,
    attention_mask: Any,
) -> Any:
    """Extract the last non-padding token's hidden state (Qwen3-Embedding pattern)."""
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


class TextOnlyEmbeddingProvider:
    """Local text-only embedding provider using Qwen3-Embedding.

    Much lighter than the multimodal VL variant (~1.2 GB vs ~4.6 GB for 2B).
    Implements the same ``embed_text()`` interface so it works as a drop-in
    with ``MultimodalVectorStore``.

    Example::

        >>> from vl_rag_graph_rlm.rag.text_embedding import TextOnlyEmbeddingProvider
        >>>
        >>> embedder = TextOnlyEmbeddingProvider()
        >>> emb = embedder.embed_text("Hello world")
        >>> len(emb)  # default dimension
        1024
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 8192,
        torch_dtype: Optional[Any] = None,
        default_instruction: str = "Represent the user's input.",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name_or_path = (
            model_name_or_path
            or os.getenv("VRLMRAG_TEXT_ONLY_MODEL", DEFAULT_TEXT_ONLY_MODEL)
        )
        self.device = device or (
            "mps"
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.max_length = max_length
        self.default_instruction = default_instruction

        resolved_dtype = torch_dtype or torch.float32
        logger.info(
            "Loading text-only embedding model %s on %s (dtype=%s)",
            self.model_name_or_path,
            self.device,
            resolved_dtype,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, padding_side="left"
        )
        self._model = AutoModel.from_pretrained(
            self.model_name_or_path, torch_dtype=resolved_dtype
        ).to(self.device)
        self._model.eval()

        # Determine embedding dimension from the model config
        self.embedding_dim: int = getattr(
            self._model.config, "hidden_size", 1024
        )
        logger.info(
            "Text-only embedding model loaded. dim=%d, device=%s",
            self.embedding_dim,
            self.device,
        )

    # ── Core interface (matches Qwen3VLEmbeddingProvider) ─────────────

    def embed_text(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> List[float]:
        """Generate embedding for text input."""
        import torch
        import torch.nn.functional as F

        instr = instruction or self.default_instruction
        # Qwen3-Embedding expects: "Instruct: {instruction}\nQuery: {text}"
        formatted = f"Instruct: {instr}\nQuery: {text}"

        inputs = self._tokenizer(
            [formatted],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        embeddings = _last_token_pool(
            outputs.last_hidden_state, inputs["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings[0].cpu().tolist()

    def embed_image(self, *args: Any, **kwargs: Any) -> List[float]:
        """Not supported in text-only mode."""
        raise NotImplementedError(
            "TextOnlyEmbeddingProvider does not support image embedding. "
            "Use Qwen3VLEmbeddingProvider for multimodal content."
        )

    def embed_video(self, *args: Any, **kwargs: Any) -> List[float]:
        """Not supported in text-only mode."""
        raise NotImplementedError(
            "TextOnlyEmbeddingProvider does not support video embedding. "
            "Use Qwen3VLEmbeddingProvider for multimodal content."
        )

    def embed_multimodal(self, *args: Any, **kwargs: Any) -> List[float]:
        """Not supported in text-only mode."""
        raise NotImplementedError(
            "TextOnlyEmbeddingProvider does not support multimodal embedding. "
            "Use Qwen3VLEmbeddingProvider for multimodal content."
        )


# ── Factory ───────────────────────────────────────────────────────────

def create_text_embedder(
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> TextOnlyEmbeddingProvider:
    """Create a text-only embedding provider.

    Args:
        model_name: HuggingFace model ID. Defaults to env var
            ``VRLMRAG_TEXT_ONLY_MODEL`` or ``Qwen/Qwen3-Embedding-0.6B``.
        **kwargs: Forwarded to ``TextOnlyEmbeddingProvider``.
    """
    return TextOnlyEmbeddingProvider(model_name_or_path=model_name, **kwargs)
