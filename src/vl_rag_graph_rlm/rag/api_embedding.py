"""API-based embedding provider using OpenRouter + ZenMux.

Provides a drop-in replacement for ``Qwen3VLEmbeddingProvider`` that
uses **zero local GPU models**:

- **Text embeddings** via OpenRouter (``openai/text-embedding-3-small``)
- **Image/video descriptions** via ZenMux omni model (PRIMARY)
  (``inclusionai/ming-flash-omni-preview``)
- **Fallback VLM** via OpenRouter Kimi K2 if ZenMux fails
  (``moonshotai/kimi-k2``)

The VLM uses a **fallback chain**: primary ZenMux → fallback OpenRouter →
circuit-breaker disable.  ZenMux Ming omni is prioritized for video analysis
but may encounter a known server-side ``audioTokens`` bug; the Kimi K2 fallback
handles image descriptions when ZenMux is unavailable.

Peak RAM: ~200 MB (no local model loading).

Required env vars (from ``.env``)::

    OPENROUTER_API_KEY=...
    ZENMUX_API_KEY=...

Optional overrides::

    VRLMRAG_EMBEDDING_MODEL=openai/text-embedding-3-small
    VRLMRAG_EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
    VRLMRAG_VLM_MODEL=inclusionai/ming-flash-omni-preview
    VRLMRAG_VLM_BASE_URL=https://zenmux.ai/api/v1
    VRLMRAG_VLM_FALLBACK_MODEL=moonshotai/kimi-k2
    VRLMRAG_VLM_FALLBACK_BASE_URL=https://openrouter.ai/api/v1
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

# Timeouts (seconds) — prevent hanging on slow/broken providers
_EMBEDDING_TIMEOUT = 30.0
_VLM_TIMEOUT = 60.0  # generous for video frame processing
# Circuit breaker: disable VLM after this many consecutive failures
_VLM_MAX_CONSECUTIVE_FAILURES = 3

logger = logging.getLogger(__name__)

# Defaults — best proven models as of Feb 12, 2026
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_EMBEDDING_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_DIM = 1536  # text-embedding-3-small

# Primary VLM — ZenMux Ming-flash-omni (prioritized for video/multimodal)
DEFAULT_VLM_MODEL = "inclusionai/ming-flash-omni-preview"
DEFAULT_VLM_BASE_URL = "https://zenmux.ai/api/v1"

# Fallback VLM — OpenRouter Kimi K2.5 (handles images when ZenMux fails)
DEFAULT_VLM_FALLBACK_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_VLM_FALLBACK_BASE_URL = "https://openrouter.ai/api/v1"

_IMAGE_DESCRIBE_PROMPT = (
    "Describe this image in detail for a document retrieval system. "
    "Include all visible text, diagrams, charts, labels, and visual elements. "
    "Be thorough and factual."
)

_VIDEO_DESCRIBE_PROMPT = (
    "Describe these video frames in detail for a document retrieval system. "
    "Summarize the visual content, any text shown, and key information."
)


class APIEmbeddingProvider:
    """API-based multimodal embedding provider.

    Uses OpenRouter for text embeddings and ZenMux's omni model for
    converting images/videos to text descriptions before embedding.

    Drop-in replacement for ``Qwen3VLEmbeddingProvider``.
    """

    def __init__(
        self,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        vlm_model: Optional[str] = None,
        default_instruction: str = "Represent the user's input.",
    ) -> None:
        import openai

        # Embedding client (OpenRouter)
        self._emb_api_key = (
            embedding_api_key
            or os.getenv("VRLMRAG_EMBEDDING_API_KEY")
            or os.getenv("OPENROUTER_API_KEY", "")
        )
        self._emb_base_url = (
            embedding_base_url
            or os.getenv("VRLMRAG_EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
        )
        self._emb_model = (
            embedding_model
            or os.getenv("VRLMRAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )

        if not self._emb_api_key:
            raise ValueError(
                "No embedding API key found. Set OPENROUTER_API_KEY or "
                "VRLMRAG_EMBEDDING_API_KEY in your .env file."
            )

        self._emb_client = openai.OpenAI(
            api_key=self._emb_api_key,
            base_url=self._emb_base_url,
            timeout=httpx.Timeout(_EMBEDDING_TIMEOUT, connect=10.0),
            max_retries=1,
        )

        # VLM client (ZenMux omni) — for image/video descriptions
        self._vlm_api_key = (
            vlm_api_key
            or os.getenv("VRLMRAG_VLM_API_KEY")
            or os.getenv("ZENMUX_API_KEY", "")
        )
        self._vlm_base_url = (
            vlm_base_url
            or os.getenv("VRLMRAG_VLM_BASE_URL", DEFAULT_VLM_BASE_URL)
        )
        self._vlm_model = (
            vlm_model
            or os.getenv("VRLMRAG_VLM_MODEL", DEFAULT_VLM_MODEL)
        )

        self._vlm_client: Optional[openai.OpenAI] = None
        if self._vlm_api_key:
            self._vlm_client = openai.OpenAI(
                api_key=self._vlm_api_key,
                base_url=self._vlm_base_url,
                timeout=httpx.Timeout(_VLM_TIMEOUT, connect=5.0),
                max_retries=0,  # fail fast — circuit breaker handles retries
            )
        # VLM fallback client (OpenRouter Kimi K2) — used when ZenMux fails
        self._vlm_fallback_api_key = (
            os.getenv("VRLMRAG_VLM_FALLBACK_API_KEY")
            or os.getenv("OPENROUTER_API_KEY", "")
        )
        self._vlm_fallback_base_url = (
            os.getenv("VRLMRAG_VLM_FALLBACK_BASE_URL", DEFAULT_VLM_FALLBACK_BASE_URL)
        )
        self._vlm_fallback_model = (
            os.getenv("VRLMRAG_VLM_FALLBACK_MODEL", DEFAULT_VLM_FALLBACK_MODEL)
        )

        self._vlm_fallback_client: Optional[openai.OpenAI] = None
        if self._vlm_fallback_api_key:
            self._vlm_fallback_client = openai.OpenAI(
                api_key=self._vlm_fallback_api_key,
                base_url=self._vlm_fallback_base_url,
                timeout=httpx.Timeout(_VLM_TIMEOUT, connect=5.0),
                max_retries=0,
            )

        self._vlm_consecutive_failures = 0
        self._vlm_disabled = False
        self._vlm_fallback_consecutive_failures = 0
        self._vlm_fallback_disabled = False

        self.default_instruction = default_instruction
        self._embedding_dim: Optional[int] = None

        logger.info(
            "API embedding provider configured: embeddings=%s via %s, "
            "vlm=%s via %s, fallback=%s via %s",
            self._emb_model,
            self._emb_base_url,
            self._vlm_model if self._vlm_client else "N/A",
            self._vlm_base_url if self._vlm_client else "N/A",
            self._vlm_fallback_model if self._vlm_fallback_client else "N/A",
            self._vlm_fallback_base_url if self._vlm_fallback_client else "N/A",
        )

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality (discovered on first call)."""
        if self._embedding_dim is None:
            probe = self._embed_texts(["probe"])
            self._embedding_dim = len(probe[0])
        return self._embedding_dim

    # ── Public interface (matches Qwen3VLEmbeddingProvider) ───────

    def embed_text(
        self, text: str, instruction: Optional[str] = None
    ) -> List[float]:
        """Generate embedding for text input."""
        return self._embed_texts([text])[0]

    def embed_image(
        self,
        image: Union[str, Any],
        instruction: Optional[str] = None,
    ) -> List[float]:
        """Describe image via VLM, then embed the description."""
        description = self._describe_image(image)
        return self._embed_texts([description])[0]

    def embed_video(
        self,
        video: Union[str, List[Any]],
        instruction: Optional[str] = None,
        fps: float = 1.0,
        max_frames: int = 64,
    ) -> List[float]:
        """Describe video frames via VLM, then embed the description."""
        description = self._describe_video(video, max_frames)
        return self._embed_texts([description])[0]

    def embed_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Any]] = None,
        video: Optional[Union[str, List[Any]]] = None,
        instruction: Optional[str] = None,
        fps: float = 1.0,
        max_frames: int = 64,
    ) -> List[float]:
        """Generate embedding for multimodal input."""
        parts: List[str] = []
        if text:
            parts.append(text)
        if image:
            parts.append(self._describe_image(image))
        if video:
            parts.append(self._describe_video(video, max_frames))
        combined = "\n\n".join(parts) if parts else ""
        return self._embed_texts([combined])[0]

    # ── Private helpers ───────────────────────────────────────────

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Call OpenRouter embeddings API."""
        resp = self._emb_client.embeddings.create(
            model=self._emb_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def _describe_image(self, image: Union[str, Any]) -> str:
        """Describe image using ZenMux primary, fallback to OpenRouter Kimi K2."""
        image_url = self._to_image_url(image)

        # Try primary VLM (ZenMux omni)
        if self._vlm_client and not self._vlm_disabled:
            try:
                resp = self._vlm_client.chat.completions.create(
                    model=self._vlm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _IMAGE_DESCRIBE_PROMPT},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    max_tokens=512,
                )
                self._vlm_consecutive_failures = 0
                return resp.choices[0].message.content or "(no description)"
            except Exception as e:
                self._vlm_consecutive_failures += 1
                if self._vlm_consecutive_failures >= _VLM_MAX_CONSECUTIVE_FAILURES:
                    self._vlm_disabled = True
                    logger.warning(
                        "Primary VLM (ZenMux) disabled after %d failures: %s",
                        self._vlm_consecutive_failures, e
                    )
                else:
                    logger.debug("Primary VLM failed (%d/%d): %s",
                                 self._vlm_consecutive_failures, _VLM_MAX_CONSECUTIVE_FAILURES, e)

        # Fallback to OpenRouter Kimi K2 if primary failed/disabled
        if self._vlm_fallback_client and not self._vlm_fallback_disabled:
            try:
                resp = self._vlm_fallback_client.chat.completions.create(
                    model=self._vlm_fallback_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _IMAGE_DESCRIBE_PROMPT},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    max_tokens=512,
                )
                self._vlm_fallback_consecutive_failures = 0
                logger.debug("Image described via fallback VLM (OpenRouter Kimi K2)")
                return resp.choices[0].message.content or "(no description)"
            except Exception as e:
                self._vlm_fallback_consecutive_failures += 1
                if self._vlm_fallback_consecutive_failures >= _VLM_MAX_CONSECUTIVE_FAILURES:
                    self._vlm_fallback_disabled = True
                    logger.error(
                        "Fallback VLM (OpenRouter Kimi K2) disabled after %d failures: %s",
                        self._vlm_fallback_consecutive_failures, e
                    )
                else:
                    logger.warning("Fallback VLM failed (%d/%d): %s",
                                   self._vlm_fallback_consecutive_failures, _VLM_MAX_CONSECUTIVE_FAILURES, e)

        return "(image description unavailable)"

    def _describe_video(
        self, video: Union[str, List[Any]], max_frames: int
    ) -> str:
        """Describe video using ZenMux primary, fallback to OpenRouter Kimi K2."""
        frames: List[str] = []
        if isinstance(video, list):
            for f in video[:max_frames]:
                frames.append(self._to_image_url(f))
        elif isinstance(video, str) and Path(video).is_file():
            frames.append(self._to_image_url(video))

        if not frames:
            return "(video — no frames available)"

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": _VIDEO_DESCRIBE_PROMPT}
        ]
        for url in frames[:8]:
            content.append({"type": "image_url", "image_url": {"url": url}})

        # Try primary VLM (ZenMux omni)
        if self._vlm_client and not self._vlm_disabled:
            try:
                resp = self._vlm_client.chat.completions.create(
                    model=self._vlm_model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=512,
                )
                self._vlm_consecutive_failures = 0
                return resp.choices[0].message.content or "(no description)"
            except Exception as e:
                self._vlm_consecutive_failures += 1
                if self._vlm_consecutive_failures >= _VLM_MAX_CONSECUTIVE_FAILURES:
                    self._vlm_disabled = True
                    logger.warning(
                        "Primary VLM (ZenMux) disabled after %d failures: %s",
                        self._vlm_consecutive_failures, e
                    )
                else:
                    logger.debug("Primary VLM failed (%d/%d): %s",
                                 self._vlm_consecutive_failures, _VLM_MAX_CONSECUTIVE_FAILURES, e)

        # Fallback to OpenRouter Kimi K2
        if self._vlm_fallback_client and not self._vlm_fallback_disabled:
            try:
                resp = self._vlm_fallback_client.chat.completions.create(
                    model=self._vlm_fallback_model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=512,
                )
                self._vlm_fallback_consecutive_failures = 0
                logger.debug("Video described via fallback VLM (OpenRouter Kimi K2)")
                return resp.choices[0].message.content or "(no description)"
            except Exception as e:
                self._vlm_fallback_consecutive_failures += 1
                if self._vlm_fallback_consecutive_failures >= _VLM_MAX_CONSECUTIVE_FAILURES:
                    self._vlm_fallback_disabled = True
                    logger.error(
                        "Fallback VLM (OpenRouter Kimi K2) disabled after %d failures: %s",
                        self._vlm_fallback_consecutive_failures, e
                    )
                else:
                    logger.warning("Fallback VLM failed (%d/%d): %s",
                                   self._vlm_fallback_consecutive_failures, _VLM_MAX_CONSECUTIVE_FAILURES, e)

        return "(video description unavailable)"

    @staticmethod
    def _to_image_url(image: Union[str, Any]) -> str:
        """Convert an image path/URL/PIL to a data-URI or URL string."""
        if isinstance(image, str):
            if image.startswith(("http://", "https://", "data:")):
                return image
            path = Path(image)
            if path.is_file():
                data = path.read_bytes()
                suffix = path.suffix.lower().lstrip(".")
                mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                        "gif": "gif", "webp": "webp"}.get(suffix, "jpeg")
                b64 = base64.b64encode(data).decode()
                return f"data:image/{mime};base64,{b64}"
        # PIL Image fallback
        try:
            import io
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return str(image)


def create_api_embedder(
    embedding_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    vlm_api_key: Optional[str] = None,
    vlm_model: Optional[str] = None,
    **_kwargs: Any,
) -> APIEmbeddingProvider:
    """Create an API-based embedding provider.

    Uses OpenRouter for embeddings + ZenMux omni for image descriptions.
    Extra ``**kwargs`` (e.g. ``device``) are silently ignored for
    interface compatibility with ``create_qwen3vl_embedder``.
    """
    return APIEmbeddingProvider(
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        vlm_api_key=vlm_api_key,
        vlm_model=vlm_model,
    )
