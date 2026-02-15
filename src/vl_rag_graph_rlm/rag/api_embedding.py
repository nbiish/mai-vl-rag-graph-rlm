"""API-based embedding provider using OpenRouter + ZenMux.

Provides a drop-in replacement for ``Qwen3VLEmbeddingProvider`` that
uses **zero local GPU models**:

- **Text embeddings** via OpenRouter (``openai/text-embedding-3-small``)
- **Image/video/audio processing** via ZenMux omni model (PRIMARY)
  (``inclusionai/ming-flash-omni-preview``)
- **Omni fallback** via ZenMux Gemini 3 Flash (SECONDARY omni)
  (``gemini/gemini-3-flash-preview``) — supports text, image, audio, video
- **VLM fallback** via OpenRouter Kimi K2 (TERTIARY — images/video only)
  (``moonshotai/kimi-k2``)

The omni model is the primary multimodal processor and is assumed to always
support: text, images, audio files, and video frames. If the primary omni
fails, we fall back to a secondary omni model (Gemini 3 Flash) which also
supports all four modalities. Only if both omnis fail do we fall back to
Kimi K2 (which only supports images/video, not audio).

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
    VRLMRAG_OMNI_FALLBACK_MODEL=gemini/gemini-3-flash-preview
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

_AUDIO_TRANSCRIBE_PROMPT = (
    "Transcribe this audio accurately. Include all spoken words, punctuation, and speaker "
    "identification if multiple speakers are present. Preserve the original language."
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

        # Primary Omni client (ZenMux) — for image/video/audio processing
        # Supports both old VRLMRAG_VLM_* and new VRLMRAG_OMNI_* variable names
        self._omni_api_key = (
            vlm_api_key
            or os.getenv("VRLMRAG_OMNI_API_KEY")
            or os.getenv("VRLMRAG_VLM_API_KEY")
            or os.getenv("ZENMUX_API_KEY", "")
        )
        self._omni_base_url = (
            vlm_base_url
            or os.getenv("VRLMRAG_OMNI_BASE_URL", DEFAULT_VLM_BASE_URL)
            or os.getenv("VRLMRAG_VLM_BASE_URL", DEFAULT_VLM_BASE_URL)
        )
        self._omni_model = (
            vlm_model
            or os.getenv("VRLMRAG_OMNI_MODEL", DEFAULT_VLM_MODEL)
            or os.getenv("VRLMRAG_VLM_MODEL", DEFAULT_VLM_MODEL)
        )

        self._omni_client: Optional[openai.OpenAI] = None
        if self._omni_api_key:
            self._omni_client = openai.OpenAI(
                api_key=self._omni_api_key,
                base_url=self._omni_base_url,
                timeout=httpx.Timeout(_VLM_TIMEOUT, connect=5.0),
                max_retries=0,  # fail fast — circuit breaker handles retries
            )

        # Secondary Omni fallback client (ZenMux with Gemini 3 Flash)
        # Uses same ZenMux API key but different model
        self._omni_fallback_api_key = (
            os.getenv("VRLMRAG_OMNI_FALLBACK_API_KEY")
            or self._omni_api_key
        )
        self._omni_fallback_base_url = (
            os.getenv("VRLMRAG_OMNI_FALLBACK_BASE_URL", self._omni_base_url)
        )
        self._omni_fallback_model = os.getenv(
            "VRLMRAG_OMNI_FALLBACK_MODEL", "gemini/gemini-3-flash-preview"
        )

        self._omni_fallback_client: Optional[openai.OpenAI] = None
        if self._omni_fallback_api_key and self._omni_fallback_model:
            self._omni_fallback_client = openai.OpenAI(
                api_key=self._omni_fallback_api_key,
                base_url=self._omni_fallback_base_url,
                timeout=httpx.Timeout(_VLM_TIMEOUT, connect=5.0),
                max_retries=0,
            )

        # Tertiary Omni fallback client (OpenRouter with Gemini 3 Flash)
        self._omni_fallback_api_key_2 = (
            os.getenv("VRLMRAG_OMNI_FALLBACK_API_KEY_2")
            or os.getenv("OPENROUTER_API_KEY", "")
        )
        self._omni_fallback_base_url_2 = (
            os.getenv("VRLMRAG_OMNI_FALLBACK_BASE_URL_2", DEFAULT_VLM_FALLBACK_BASE_URL)
        )
        self._omni_fallback_model_2 = os.getenv(
            "VRLMRAG_OMNI_FALLBACK_MODEL_2", "google/gemini-3-flash-preview"
        )

        self._omni_fallback_client_2: Optional[openai.OpenAI] = None
        if self._omni_fallback_api_key_2 and self._omni_fallback_model_2:
            self._omni_fallback_client_2 = openai.OpenAI(
                api_key=self._omni_fallback_api_key_2,
                base_url=self._omni_fallback_base_url_2,
                timeout=httpx.Timeout(_VLM_TIMEOUT, connect=5.0),
                max_retries=0,
            )

        # Legacy VLM fallback client (OpenRouter Kimi K2) — images/video only
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

        # Legacy aliases for backward compatibility in method code
        self._vlm_api_key = self._omni_api_key
        self._vlm_base_url = self._omni_base_url
        self._vlm_model = self._omni_model
        self._vlm_client = self._omni_client

        self._vlm_consecutive_failures = 0
        self._vlm_disabled = False
        self._vlm_fallback_consecutive_failures = 0
        self._vlm_fallback_disabled = False

        self.default_instruction = default_instruction
        self._embedding_dim: Optional[int] = None

        logger.info(
            "API embedding provider configured: embeddings=%s via %s, "
            "omni=%s via %s, omni_fallback=%s, omni_fallback_2=%s, vlm_fallback=%s via %s",
            self._emb_model,
            self._emb_base_url,
            self._omni_model if self._omni_client else "N/A",
            self._omni_base_url if self._omni_client else "N/A",
            self._omni_fallback_model if self._omni_fallback_client else "N/A",
            self._omni_fallback_model_2 if self._omni_fallback_client_2 else "N/A",
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

    def batch_embed_images(
        self,
        image_paths: List[str],
        batch_size: int = 4,
        instruction: Optional[str] = None,
    ) -> List[List[float]]:
        """Embed multiple images in batches for improved throughput.
        
        This method processes images in parallel batches, significantly reducing
        total processing time for video frames or multiple images.
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process in parallel (default 4)
            instruction: Optional custom instruction for image description
            
        Returns:
            List of embeddings, one per input image
        """
        import concurrent.futures
        from functools import partial
        
        if not image_paths:
            return []
        
        # Process images in parallel using thread pool
        embed_fn = partial(self.embed_image, instruction=instruction)
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(embed_fn, path): idx 
                for idx, path in enumerate(image_paths)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embedding = future.result()
                    results.append((idx, embedding))
                except Exception as e:
                    logger.warning(f"Failed to embed image {image_paths[idx]}: {e}")
                    # Return zero vector as fallback
                    results.append((idx, [0.0] * self.embedding_dim))
        
        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

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

    # ── Omni Model Methods (Primary + Fallbacks) ──────────────────
    # These methods implement the three-tier omni fallback chain:
    #   1. Primary: ZenMux omni model (text/image/audio/video)
    #   2. Secondary: ZenMux Gemini 3 Flash (text/image/audio/video)
    #   3. Tertiary: OpenRouter Gemini 3 Flash (text/image/audio/video)
    #   4. Legacy VLM: OpenRouter Kimi K2 (text/image/video only, no audio)

    def transcribe_audio(
        self,
        audio: Union[str, Any],
        instruction: Optional[str] = None,
    ) -> str:
        """Transcribe audio using omni model chain: primary → secondary → tertiary.

        The omni models support audio transcription directly via the chat.completions
        API with audio file inputs. Falls back through all three omni tiers.

        Args:
            audio: Path to audio file (.wav, .mp3, etc.) or audio data
            instruction: Optional custom transcription instruction

        Returns:
            Transcription text or placeholder if all omni models fail
        """
        # Convert audio to data URI if it's a file path
        audio_url = self._to_audio_url(audio)
        if not audio_url:
            return "(audio transcription unavailable — file conversion failed)"

        prompt = instruction or _AUDIO_TRANSCRIBE_PROMPT

        # Try primary VLM (ZenMux omni) with audio support
        if self._vlm_client and not self._vlm_disabled:
            try:
                resp = self._vlm_client.chat.completions.create(
                    model=self._vlm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "input_audio", "input_audio": {"data": audio_url, "format": "wav"}},
                            ],
                        }
                    ],
                    max_tokens=1024,  # Longer for transcripts
                )
                self._vlm_consecutive_failures = 0
                transcript = resp.choices[0].message.content or "(no transcription)"
                logger.debug(f"Audio transcribed via ZenMux omni: {len(transcript)} chars")
                return transcript
            except Exception as e:
                self._vlm_consecutive_failures += 1
                logger.warning(f"ZenMux omni audio transcription failed: {e}")
                if self._vlm_consecutive_failures >= _VLM_MAX_CONSECUTIVE_FAILURES:
                    self._vlm_disabled = True
                    logger.warning(f"Primary VLM disabled after {self._vlm_consecutive_failures} failures")

        # Fallback: try secondary omni model (ZenMux Gemini 3 Flash) if configured
        if self._omni_fallback_client and not self._vlm_disabled:
            try:
                logger.info(f"Trying omni fallback model: {self._omni_fallback_model}")
                resp = self._omni_fallback_client.chat.completions.create(
                    model=self._omni_fallback_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "input_audio", "input_audio": {"data": audio_url, "format": "wav"}},
                            ],
                        }
                    ],
                    max_tokens=1024,
                )
                transcript = resp.choices[0].message.content or "(no transcription)"
                logger.info(f"Audio transcribed via omni fallback: {len(transcript)} chars")
                return transcript
            except Exception as e:
                logger.warning(f"Omni fallback audio transcription failed: {e}")

        # Fallback: try tertiary omni model (OpenRouter Gemini 3 Flash) if configured
        if self._omni_fallback_client_2 and not self._vlm_fallback_disabled:
            try:
                logger.info(f"Trying tertiary omni fallback model: {self._omni_fallback_model_2}")
                resp = self._omni_fallback_client_2.chat.completions.create(
                    model=self._omni_fallback_model_2,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "input_audio", "input_audio": {"data": audio_url, "format": "wav"}},
                            ],
                        }
                    ],
                    max_tokens=1024,
                )
                transcript = resp.choices[0].message.content or "(no transcription)"
                logger.info(f"Audio transcribed via tertiary omni fallback: {len(transcript)} chars")
                return transcript
            except Exception as e:
                logger.warning(f"Tertiary omni fallback audio transcription failed: {e}")

        # Fallback to legacy VLM (Kimi K2) - NOTE: Does NOT support audio!
        # This fallback is only reached if all omni models fail
        logger.debug("Audio transcription: All omni models failed, Kimi K2 fallback unavailable for audio")

        return "(audio transcription unavailable — all omni models failed)"

    # ── Omni Model Helpers ──────────────────────────────────────

    @staticmethod
    def _to_audio_url(audio: Union[str, Any]) -> Optional[str]:
        """Convert an audio path to a base64 data-URI for API upload."""
        if isinstance(audio, str):
            if audio.startswith(("http://", "https://", "data:")):
                return audio
            path = Path(audio)
            if path.is_file():
                # Read and encode audio file
                data = path.read_bytes()
                suffix = path.suffix.lower().lstrip(".")
                # Map common audio formats
                mime_map = {
                    "wav": "wav",
                    "mp3": "mp3",
                    "flac": "flac",
                    "ogg": "ogg",
                    "m4a": "m4a",
                    "webm": "webm",
                }
                mime = mime_map.get(suffix, "wav")  # Default to wav
                b64 = base64.b64encode(data).decode()
                return f"data:audio/{mime};base64,{b64}"
        return None

    # ── Private helpers ───────────────────────────────────────────

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Call OpenRouter embeddings API."""
        resp = self._emb_client.embeddings.create(
            model=self._emb_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    # ── Omni Model Methods ──────────────────────────────────────

    def _describe_image(self, image: Union[str, Any]) -> str:
        """Describe image using omni model chain: primary → secondary → tertiary → legacy VLM."""
        image_url = self._to_image_url(image)

        # Try primary omni (ZenMux)
        if self._omni_client and not self._vlm_disabled:
            try:
                resp = self._omni_client.chat.completions.create(
                    model=self._omni_model,
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
                        "Primary omni (ZenMux) disabled after %d failures: %s",
                        self._vlm_consecutive_failures, e
                    )
                else:
                    logger.debug("Primary omni failed (%d/%d): %s",
                                 self._vlm_consecutive_failures, _VLM_MAX_CONSECUTIVE_FAILURES, e)

        # Fallback: try secondary omni model (ZenMux Gemini 3 Flash) if configured
        if self._omni_fallback_client and not self._vlm_disabled:
            try:
                logger.info(f"Trying omni fallback for image: {self._omni_fallback_model}")
                resp = self._omni_fallback_client.chat.completions.create(
                    model=self._omni_fallback_model,
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
                logger.info(f"Image described via secondary omni: {self._omni_fallback_model}")
                return resp.choices[0].message.content or "(no description)"
            except Exception as e:
                logger.warning(f"Secondary omni image description failed: {e}")

        # Fallback: try tertiary omni (OpenRouter Gemini 3 Flash)
        if self._omni_fallback_client_2 and not self._vlm_fallback_disabled:
            try:
                logger.info(f"Trying tertiary omni fallback for image: {self._omni_fallback_model_2}")
                resp = self._omni_fallback_client_2.chat.completions.create(
                    model=self._omni_fallback_model_2,
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
                logger.info(f"Image described via tertiary omni: {self._omni_fallback_model_2}")
                return resp.choices[0].message.content or "(no description)"
            except Exception as e:
                logger.warning(f"Tertiary omni image description failed: {e}")

        # Fallback: try legacy VLM (OpenRouter Kimi K2) - images/video only
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
        """Describe video using omni model chain: primary → secondary → tertiary → legacy VLM."""
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

        # Try primary omni (ZenMux)
        if self._omni_client and not self._vlm_disabled:
            try:
                resp = self._omni_client.chat.completions.create(
                    model=self._omni_model,
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
                        "Primary omni (ZenMux) disabled after %d failures: %s",
                        self._vlm_consecutive_failures, e
                    )
                else:
                    logger.debug("Primary omni failed (%d/%d): %s",
                                 self._vlm_consecutive_failures, _VLM_MAX_CONSECUTIVE_FAILURES, e)

        # Fallback: try secondary omni model (ZenMux Gemini 3 Flash) if configured
        if self._omni_fallback_client and not self._vlm_disabled:
            try:
                logger.info(f"Trying omni fallback for video: {self._omni_fallback_model}")
                resp = self._omni_fallback_client.chat.completions.create(
                    model=self._omni_fallback_model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=512,
                )
                logger.info(f"Video described via secondary omni: {self._omni_fallback_model}")
                return resp.choices[0].message.content or "(no description)"
            except Exception as e:
                logger.warning(f"Secondary omni video description failed: {e}")

        # Fallback: try tertiary omni (OpenRouter Gemini 3 Flash)
        if self._omni_fallback_client_2 and not self._vlm_fallback_disabled:
            try:
                logger.info(f"Trying tertiary omni fallback for video: {self._omni_fallback_model_2}")
                resp = self._omni_fallback_client_2.chat.completions.create(
                    model=self._omni_fallback_model_2,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=512,
                )
                logger.info(f"Video described via tertiary omni: {self._omni_fallback_model_2}")
                return resp.choices[0].message.content or "(no description)"
            except Exception as e:
                logger.warning(f"Tertiary omni video description failed: {e}")

        # Fallback: try legacy VLM (OpenRouter Kimi K2) - images/video only
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
