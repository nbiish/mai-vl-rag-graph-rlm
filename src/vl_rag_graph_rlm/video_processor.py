"""Async video processing with progress callbacks for VL-RAG-Graph-RLM

Provides non-blocking video processing with real-time progress updates
to support long-running video analysis in production environments.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class VideoProcessingProgress:
    """Progress state for video processing."""
    stage: str  # 'extracting_audio', 'extracting_frames', 'transcribing', 'embedding'
    current: int
    total: int
    percent: float
    message: str
    timestamp: datetime


@dataclass
class VideoProcessingResult:
    """Result from async video processing."""
    success: bool
    document: Dict[str, Any]
    processing_time_seconds: float
    frames_extracted: int
    transcript_length: int
    errors: List[str]


class AsyncVideoProcessor:
    """Async video processor with progress callbacks for production use."""
    
    def __init__(
        self,
        max_frames: int = 16,
        fps: float = 0.5,
        use_api: bool = True,
        progress_callback: Optional[Callable[[VideoProcessingProgress], None]] = None
    ):
        self.max_frames = max_frames
        self.fps = fps
        self.use_api = use_api
        self.progress_callback = progress_callback
        self._cancelled = False
    
    def cancel(self):
        """Signal cancellation of ongoing processing."""
        self._cancelled = True
    
    def _report_progress(
        self, stage: str, current: int, total: int, message: str
    ):
        """Report progress via callback if registered."""
        if self.progress_callback and not self._cancelled:
            progress = VideoProcessingProgress(
                stage=stage,
                current=current,
                total=total,
                percent=(current / total * 100) if total > 0 else 0,
                message=message,
                timestamp=datetime.now()
            )
            try:
                self.progress_callback(progress)
            except Exception:
                # Don't let callback errors break processing
                pass
    
    async def process_video(
        self,
        video_path: Path,
        api_embedder: Optional[Any] = None,
        transcription_provider: Optional[Any] = None
    ) -> VideoProcessingResult:
        """Process a video file asynchronously with progress updates.
        
        Args:
            video_path: Path to video file
            api_embedder: API embedder for omni model processing
            transcription_provider: Local transcription provider (Parakeet)
            
        Returns:
            VideoProcessingResult with document data and metadata
        """
        import tempfile
        import time
        
        start_time = time.time()
        errors = []
        
        result = VideoProcessingResult(
            success=False,
            document={},
            processing_time_seconds=0,
            frames_extracted=0,
            transcript_length=0,
            errors=[]
        )
        
        try:
            # Initialize document structure
            doc = {
                "type": "video",
                "path": str(video_path),
                "content": "",
                "chunks": [],
                "frame_paths": [],
                "audio_path": None,
            }
            
            # Stage 1: Extract audio (async via thread pool)
            self._report_progress("extracting_audio", 0, 1, "Starting audio extraction...")
            audio_path = await self._extract_audio_async(video_path)
            doc["audio_path"] = audio_path
            self._report_progress("extracting_audio", 1, 1, "Audio extracted")
            
            if self._cancelled:
                raise asyncio.CancelledError("Processing cancelled")
            
            # Stage 2: Extract frames (async via thread pool)
            self._report_progress("extracting_frames", 0, self.max_frames, 
                                f"Extracting up to {self.max_frames} frames...")
            
            frame_paths = await self._extract_frames_async(
                video_path,
                fps=self.fps,
                max_frames=self.max_frames
            )
            doc["frame_paths"] = frame_paths
            result.frames_extracted = len(frame_paths)
            
            self._report_progress(
                "extracting_frames", 
                len(frame_paths), 
                self.max_frames,
                f"Extracted {len(frame_paths)} frames"
            )
            
            if self._cancelled:
                raise asyncio.CancelledError("Processing cancelled")
            
            # Stage 3: Transcribe audio
            transcript = ""
            if audio_path:
                self._report_progress("transcribing", 0, 1, "Starting transcription...")
                
                if self.use_api and api_embedder:
                    transcript = await self._transcribe_audio_api_async(
                        audio_path, api_embedder
                    )
                elif transcription_provider:
                    transcript = await self._transcribe_audio_local_async(
                        audio_path, transcription_provider
                    )
                
                result.transcript_length = len(transcript)
                self._report_progress("transcribing", 1, 1, 
                                    f"Transcribed {len(transcript)} chars")
            
            if self._cancelled:
                raise asyncio.CancelledError("Processing cancelled")
            
            # Stage 4: Build chunks
            self._report_progress("embedding", 0, 1, "Building document chunks...")
            
            if transcript:
                doc["content"] = transcript
                # Chunk transcript into ~500-char segments
                words = transcript.split()
                chunk_words = []
                chunks = []
                for word in words:
                    chunk_words.append(word)
                    if len(" ".join(chunk_words)) >= 500:
                        chunk_text = " ".join(chunk_words)
                        chunks.append({"content": chunk_text, "type": "transcript"})
                        chunk_words = []
                if chunk_words:
                    chunk_text = " ".join(chunk_words)
                    if chunk_text.strip():
                        chunks.append({"content": chunk_text, "type": "transcript"})
                doc["chunks"] = chunks
            else:
                doc["content"] = f"[Video: {video_path.name}]"
            
            self._report_progress("embedding", 1, 1, "Document ready for embedding")
            
            result.success = True
            result.document = doc
            result.processing_time_seconds = time.time() - start_time
            
        except asyncio.CancelledError:
            errors.append("Processing was cancelled by user")
            result.errors = errors
        except Exception as e:
            errors.append(f"Video processing failed: {e}")
            result.errors = errors
            result.processing_time_seconds = time.time() - start_time
        
        return result
    
    async def _extract_audio_async(self, video_path: Path) -> Optional[str]:
        """Extract audio track from video asynchronously."""
        import tempfile
        
        loop = asyncio.get_event_loop()
        audio_path = tempfile.mktemp(suffix=".wav", prefix="vrlmrag_audio_")
        
        def _extract():
            from vrlmrag import DocumentProcessor
            processor = DocumentProcessor()
            ok = processor._extract_audio_ffmpeg(str(video_path), audio_path)
            return audio_path if ok else None
        
        return await loop.run_in_executor(None, _extract)
    
    async def _extract_frames_async(
        self, video_path: Path, fps: float, max_frames: int
    ) -> List[str]:
        """Extract frames from video asynchronously."""
        loop = asyncio.get_event_loop()
        
        def _extract():
            from vrlmrag import DocumentProcessor
            processor = DocumentProcessor()
            return processor._extract_frames_ffmpeg(
                str(video_path), fps=fps, max_frames=max_frames
            )
        
        return await loop.run_in_executor(None, _extract)
    
    async def _transcribe_audio_api_async(
        self, audio_path: str, api_embedder: Any
    ) -> str:
        """Transcribe audio using API omni model asynchronously."""
        loop = asyncio.get_event_loop()
        
        def _transcribe():
            try:
                return api_embedder.transcribe_audio(audio_path)
            except Exception as e:
                return f""
        
        transcript = await loop.run_in_executor(None, _transcribe)
        
        # Filter out placeholder responses
        if transcript and not transcript.startswith("(audio"):
            return transcript
        return ""
    
    async def _transcribe_audio_local_async(
        self, audio_path: str, transcription_provider: Any
    ) -> str:
        """Transcribe audio using local provider (Parakeet) asynchronously."""
        loop = asyncio.get_event_loop()
        
        def _transcribe():
            try:
                result = transcription_provider.transcribe(audio_path)
                if isinstance(result, dict):
                    return result.get("text", "")
                return result
            except Exception as e:
                return ""
        
        return await loop.run_in_executor(None, _transcribe)


def create_video_processing_task(
    video_path: str,
    on_progress: Optional[Callable[[VideoProcessingProgress], None]] = None,
    on_complete: Optional[Callable[[VideoProcessingResult], None]] = None,
    max_frames: int = 16,
    use_api: bool = True
) -> asyncio.Task:
    """Create an async task for video processing with callbacks.
    
    Args:
        video_path: Path to video file
        on_progress: Callback for progress updates
        on_complete: Callback for completion
        max_frames: Maximum frames to extract
        use_api: Use API-based processing
        
    Returns:
        asyncio.Task that can be awaited or cancelled
    """
    processor = AsyncVideoProcessor(
        max_frames=max_frames,
        use_api=use_api,
        progress_callback=on_progress
    )
    
    async def _process():
        result = await processor.process_video(Path(video_path))
        if on_complete:
            on_complete(result)
        return result
    
    return asyncio.create_task(_process())
