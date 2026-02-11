"""Lightweight Parakeet V3 audio transcription provider.

Transcribes audio to text using NVIDIA Parakeet V3, then uses existing
Qwen3-VL text embedder for vector representations. Minimal RAM impact
through lazy loading and aggressive caching.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("rlm.rag.parakeet")


class ParakeetTranscriptionProvider:
    """Lazy-loaded Parakeet V3 transcription provider.
    
    Transcribes audio to text using the smallest Parakeet model (0.6B),
    then text embeddings are handled by Qwen3-VL. This avoids loading
    a separate audio embedding model, minimizing RAM usage.
    
    The model is only loaded when first transcription is requested.
    Transcriptions are cached by file hash to avoid re-processing.
    
    Example:
        >>> from vl_rag_graph_rlm.rag.parakeet import ParakeetTranscriptionProvider
        >>> 
        >>> # Initialize (model not loaded yet)
        >>> provider = ParakeetTranscriptionProvider()
        >>> 
        >>> # First transcription triggers model load
        >>> transcript = provider.transcribe("audio.mp3")
        >>> print(transcript)
        "The quick brown fox jumps over the lazy dog..."
        >>> 
        >>> # Subsequent calls use cache
        >>> transcript2 = provider.transcribe("audio.mp3")  # Instant (cached)
    """
    
    # Smallest, fastest model for transcription
    DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """Initialize transcription provider (model NOT loaded yet).
        
        Args:
            model_name: Parakeet model name. Default: nvidia/parakeet-tdt-0.6b-v3
            device: Device for inference (cuda/cpu). Auto-detected if None.
            cache_dir: Directory for transcription cache. Disabled if None.
            use_cache: Whether to cache transcriptions in memory.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Lazy-loaded components (None until first use)
        self._model = None
        self._transcription_cache: Dict[str, str] = {}
        
        logger.info(f"Parakeet provider initialized (model='{self.model_name}', lazy-load)")
    
    def _cuda_available(self) -> bool:
        """Check CUDA availability without importing torch at init time."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_model(self):
        """Lazy-load the Parakeet model only when needed."""
        if self._model is not None:
            return
        
        logger.info(f"Loading Parakeet model: {self.model_name}")
        
        try:
            import nemo.collections.asr as nemo_asr
            from nemo.utils import logging as nemo_logging
            
            # Suppress NeMo verbosity
            nemo_logging.setLevel(logging.WARNING)
            
            # Load model - this is the RAM-heavy operation
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name,
                map_location=self.device
            )
            
            # Optimize for inference
            self._model.eval()
            if self.device == "cuda":
                self._model = self._model.to(self.device)
            
            logger.info(f"Parakeet model loaded on {self.device}")
            
        except ImportError as e:
            raise ImportError(
                "nemo_toolkit[asr] not installed. "
                "Install with: pip install nemo_toolkit[asr]"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Parakeet model: {e}")
            raise
    
    def _compute_file_hash(self, audio_path: str) -> str:
        """Compute hash of audio file for caching."""
        # Use file metadata for speed (mtime + size + path)
        stat = os.stat(audio_path)
        hash_input = f"{audio_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _get_cached_transcription(self, audio_path: str) -> Optional[str]:
        """Check memory and disk cache for transcription."""
        file_hash = self._compute_file_hash(audio_path)
        
        # Check memory cache
        if self.use_cache and file_hash in self._transcription_cache:
            logger.debug(f"Memory cache hit for {audio_path}")
            return self._transcription_cache[file_hash]
        
        # Check disk cache
        if self.cache_dir:
            cache_file = Path(self.cache_dir) / f"{file_hash}.txt"
            if cache_file.exists():
                transcript = cache_file.read_text(encoding="utf-8")
                # Populate memory cache
                if self.use_cache:
                    self._transcription_cache[file_hash] = transcript
                logger.debug(f"Disk cache hit for {audio_path}")
                return transcript
        
        return None
    
    def _cache_transcription(self, audio_path: str, transcript: str):
        """Cache transcription to memory and disk."""
        file_hash = self._compute_file_hash(audio_path)
        
        # Memory cache
        if self.use_cache:
            self._transcription_cache[file_hash] = transcript
        
        # Disk cache
        if self.cache_dir:
            cache_dir = Path(self.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{file_hash}.txt"
            cache_file.write_text(transcript, encoding="utf-8")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        batch_size: int = 1,
        return_timestamps: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Transcribe audio file to text.
        
        Model is loaded on first call. Subsequent calls for the same
        file are served from cache instantly.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, .flac, etc.)
            batch_size: Batch size for inference (1 for single file)
            return_timestamps: If True, return dict with text and timestamps
            
        Returns:
            Transcription text (str) or dict with timestamps if requested
        """
        audio_path = str(audio_path)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check cache first (fast path)
        cached = self._get_cached_transcription(audio_path)
        if cached is not None:
            if return_timestamps:
                # Cached transcriptions don't have timestamps
                # Re-transcribe if timestamps needed
                pass
            else:
                return cached
        
        # Load model if needed
        self._load_model()
        
        # Transcribe
        logger.info(f"Transcribing: {audio_path}")
        
        try:
            # NeMo transcribe returns list for batch
            results = self._model.transcribe([audio_path], batch_size=batch_size)
            
            # Extract text (handle different return formats)
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
                if isinstance(result, dict):
                    transcript = result.get("text", "")
                    timestamps = result.get("timestamps", None)
                else:
                    transcript = str(result)
                    timestamps = None
            elif isinstance(results, dict):
                transcript = results.get("text", "")
                timestamps = results.get("timestamps", None)
            else:
                transcript = str(results)
                timestamps = None
            
            # Clean up transcript
            transcript = transcript.strip()
            
            # Cache result
            self._cache_transcription(audio_path, transcript)
            
            logger.info(f"Transcription complete: {len(transcript)} chars")
            
            if return_timestamps and timestamps:
                return {"text": transcript, "timestamps": timestamps}
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        batch_size: int = 4
    ) -> List[str]:
        """Transcribe multiple audio files efficiently.
        
        Uses batch processing for faster throughput. Cached files
        are returned instantly, only uncached files go through the model.
        
        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for inference (higher = more RAM)
            
        Returns:
            List of transcriptions in same order as input
        """
        if not audio_paths:
            return []
        
        results = []
        to_transcribe = []
        to_transcribe_indices = []
        
        # Check cache for each file
        for i, path in enumerate(audio_paths):
            path_str = str(path)
            cached = self._get_cached_transcription(path_str)
            if cached is not None:
                results.append((i, cached))
            else:
                to_transcribe.append(path_str)
                to_transcribe_indices.append(i)
                results.append((i, None))  # Placeholder
        
        # Transcribe uncached files in batch
        if to_transcribe:
            self._load_model()
            logger.info(f"Batch transcribing {len(to_transcribe)} files")
            
            try:
                batch_results = self._model.transcribe(
                    to_transcribe, 
                    batch_size=batch_size
                )
                
                # Process results
                for idx, path, result in zip(to_transcribe_indices, to_transcribe, batch_results):
                    if isinstance(result, dict):
                        transcript = result.get("text", "").strip()
                    else:
                        transcript = str(result).strip()
                    
                    # Cache and store
                    self._cache_transcription(path, transcript)
                    results[idx] = (idx, transcript)
                    
            except Exception as e:
                logger.error(f"Batch transcription failed: {e}")
                # Fill in errors
                for idx in to_transcribe_indices:
                    results[idx] = (idx, f"[Transcription error: {e}]")
        
        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._model is not None,
            "cache_size": len(self._transcription_cache),
            "cache_dir": self.cache_dir,
        }
    
    def clear_cache(self):
        """Clear in-memory transcription cache."""
        self._transcription_cache.clear()
        logger.info("Transcription cache cleared")


def create_parakeet_transcriber(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> ParakeetTranscriptionProvider:
    """Create a Parakeet transcription provider with sensible defaults.
    
    Args:
        model_name: Model name (default: nvidia/parakeet-tdt-0.6b-v3)
        device: cuda or cpu (auto-detected if None)
        cache_dir: Directory for disk cache (disabled if None)
        
    Returns:
        Configured ParakeetTranscriptionProvider
    """
    return ParakeetTranscriptionProvider(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        use_cache=True
    )
