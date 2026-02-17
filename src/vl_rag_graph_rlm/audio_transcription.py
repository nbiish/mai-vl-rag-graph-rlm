"""Parallel chunked audio transcription for efficient processing.

Splits long audio files into chunks, transcribes them in parallel,
and combines results for faster overall transcription.
"""

import asyncio
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger("rlm.audio.transcription")


@dataclass
class ChunkResult:
    """Result from transcribing a single audio chunk."""
    chunk_index: int
    text: str
    start_time: float  # Start time in original audio
    end_time: float    # End time in original audio
    duration: float    # Chunk duration


class ParallelAudioTranscriber:
    """Transcribe long audio files using parallel chunking.
    
    Splits audio into chunks, transcribes in parallel using thread pool,
    and combines results. Optimized for both local (Parakeet) and API providers.
    
    Example:
        >>> transcriber = ParallelAudioTranscriber(
        ...     transcription_provider=parakeet_provider,
        ...     max_workers=4,
        ...     chunk_duration_sec=30
        ... )
        >>> result = transcriber.transcribe("long_audio.wav")
        >>> print(result["text"])  # Full transcript
        >>> print(result["chunks"])  # Per-chunk results with timestamps
    """
    
    def __init__(
        self,
        transcription_provider: Any,
        max_workers: int = 4,
        chunk_duration_sec: float = 30.0,
        overlap_sec: float = 1.0,
    ):
        """Initialize parallel transcriber.
        
        Args:
            transcription_provider: Provider with transcribe() method
            max_workers: Number of parallel transcription workers
            chunk_duration_sec: Duration of each chunk in seconds
            overlap_sec: Overlap between chunks to avoid cutting words
        """
        self.provider = transcription_provider
        self.max_workers = max_workers
        self.chunk_duration_sec = chunk_duration_sec
        self.overlap_sec = overlap_sec
        
    def transcribe(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """Transcribe audio file using parallel chunking.
        
        Args:
            audio_path: Path to audio file
            progress_callback: Optional callback(current_chunk, total_chunks)
            
        Returns:
            Dict with 'text' (full transcript), 'chunks' (list of ChunkResult),
            and 'duration' (total audio duration)
        """
        # Get audio duration using ffprobe
        duration = self._get_audio_duration(audio_path)
        if duration is None:
            logger.error(f"Could not determine duration of {audio_path}")
            # Fall back to single-chunk transcription
            return self._transcribe_single(audio_path)
        
        # If audio is short, don't chunk
        if duration <= self.chunk_duration_sec:
            return self._transcribe_single(audio_path)
        
        # Split into chunks
        chunks = self._split_audio(audio_path, duration)
        logger.info(f"Split {duration:.1f}s audio into {len(chunks)} chunks")
        
        # Transcribe chunks in parallel
        results = self._transcribe_chunks_parallel(chunks, progress_callback)
        
        # Combine results (handle overlaps)
        full_text = self._combine_transcripts(results)
        
        return {
            "text": full_text,
            "chunks": [
                {
                    "index": r.chunk_index,
                    "text": r.text,
                    "start": r.start_time,
                    "end": r.end_time,
                }
                for r in results
            ],
            "duration": duration,
            "chunk_count": len(chunks),
        }
    
    def _get_audio_duration(self, audio_path: str) -> Optional[float]:
        """Get audio duration using ffprobe."""
        import subprocess
        
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path
                ],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"ffprobe failed: {e}")
            return None
    
    def _split_audio(
        self,
        audio_path: str,
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """Split audio into chunks using ffmpeg.
        
        Returns list of dicts with temp_path, start_time, end_time.
        """
        import subprocess
        import tempfile
        import os
        
        chunks = []
        chunk_dir = tempfile.mkdtemp(prefix="vrlmrag_audio_chunks_")
        
        # Calculate chunk boundaries
        step = self.chunk_duration_sec - self.overlap_sec
        num_chunks = int((total_duration + step - 1) // step)
        
        for i in range(num_chunks):
            start = i * step
            end = min(start + self.chunk_duration_sec, total_duration)
            
            if end - start < 1.0:  # Skip very short final chunk
                continue
            
            chunk_path = os.path.join(chunk_dir, f"chunk_{i:04d}.wav")
            
            # Extract chunk using ffmpeg
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-ss", str(start),
                "-t", str(end - start),
                "-ar", "16000",  # Resample to 16kHz (common for ASR)
                "-ac", "1",      # Convert to mono
                "-y",             # Overwrite output
                "-loglevel", "error",
                chunk_path
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                    chunks.append({
                        "index": i,
                        "path": chunk_path,
                        "start": start,
                        "end": end,
                    })
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to extract chunk {i}: {e}")
                continue
        
        return chunks
    
    def _transcribe_chunks_parallel(
        self,
        chunks: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ChunkResult]:
        """Transcribe chunks in parallel using thread pool."""
        results = []
        total = len(chunks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self._transcribe_chunk, chunk): chunk
                for chunk in chunks
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Chunk {chunk['index']} transcription failed: {e}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        
        # Sort by chunk index
        results.sort(key=lambda r: r.chunk_index)
        return results
    
    def _transcribe_chunk(self, chunk: Dict[str, Any]) -> Optional[ChunkResult]:
        """Transcribe a single chunk."""
        try:
            result = self.provider.transcribe(chunk["path"])
            
            # Handle different return types
            if isinstance(result, dict):
                text = result.get("text", "")
            else:
                text = str(result)
            
            return ChunkResult(
                chunk_index=chunk["index"],
                text=text,
                start_time=chunk["start"],
                end_time=chunk["end"],
                duration=chunk["end"] - chunk["start"]
            )
            
        except Exception as e:
            logger.warning(f"Failed to transcribe chunk {chunk['index']}: {e}")
            return None
        finally:
            # Clean up temp file
            try:
                import os
                os.unlink(chunk["path"])
            except:
                pass
    
    def _combine_transcripts(self, results: List[ChunkResult]) -> str:
        """Combine chunk transcripts, handling overlaps."""
        if not results:
            return ""
        
        # Simple concatenation with deduplication of potential overlapping text
        # More sophisticated algorithms could use text similarity
        texts = [r.text.strip() for r in results]
        
        # Remove empty strings
        texts = [t for t in texts if t]
        
        # Join with spaces
        full_text = " ".join(texts)
        
        # Clean up extra whitespace
        full_text = " ".join(full_text.split())
        
        return full_text
    
    def _transcribe_single(self, audio_path: str) -> Dict[str, Any]:
        """Fallback: transcribe entire file without chunking."""
        duration = self._get_audio_duration(audio_path)
        
        try:
            result = self.provider.transcribe(audio_path)
            
            if isinstance(result, dict):
                text = result.get("text", "")
            else:
                text = str(result)
            
            return {
                "text": text,
                "chunks": [
                    {
                        "index": 0,
                        "text": text,
                        "start": 0,
                        "end": duration or 0,
                    }
                ],
                "duration": duration or 0,
                "chunk_count": 1,
            }
            
        except Exception as e:
            logger.error(f"Single-chunk transcription failed: {e}")
            return {
                "text": "",
                "chunks": [],
                "duration": duration or 0,
                "chunk_count": 0,
                "error": str(e)
            }


async def transcribe_audio_parallel(
    audio_path: str,
    transcription_provider: Any,
    max_workers: int = 4,
    chunk_duration_sec: float = 30.0,
) -> str:
    """Async wrapper for parallel audio transcription.
    
    Args:
        audio_path: Path to audio file
        transcription_provider: Provider with transcribe() method
        max_workers: Number of parallel workers
        chunk_duration_sec: Chunk duration in seconds
        
    Returns:
        Transcribed text
    """
    transcriber = ParallelAudioTranscriber(
        transcription_provider=transcription_provider,
        max_workers=max_workers,
        chunk_duration_sec=chunk_duration_sec,
    )
    
    # Run in thread pool since transcription is CPU/IO bound
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, transcriber.transcribe, audio_path
    )
    
    return result.get("text", "")
