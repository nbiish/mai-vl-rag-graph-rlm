"""Test audio integration with Parakeet transcription.

Tests the audio workflow: add_audio() -> transcribe -> embed -> search.
Uses mocked transcription provider to avoid heavy dependencies.
"""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

import numpy as np


# Mock Parakeet transcription provider for testing
class MockParakeetProvider:
    """Mock transcription provider for testing."""
    
    def __init__(self):
        self.transcriptions = {}
        self._loaded = False
    
    def transcribe(self, audio_path: str, **kwargs) -> str:
        """Return mock transcription."""
        self._loaded = True
        # Return predefined transcription or generate based on filename
        if audio_path in self.transcriptions:
            return self.transcriptions[audio_path]
        
        # Generate mock transcription from filename
        filename = Path(audio_path).stem
        return f"This is a mock transcription of {filename}. It discusses important topics."
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "mock-parakeet",
            "loaded": self._loaded,
            "cache_size": 0
        }


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""
    def __init__(self, dim: int = 4):
        self.embedding_dim = dim
        self.default_instruction = "You are a helpful assistant."
        self._rng = np.random.RandomState(42)
    
    def embed_text(self, text: str, instruction: Optional[str] = None) -> List[float]:
        # Deterministic embedding based on text hash
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(self.embedding_dim).tolist()
    
    def embed_batch(self, texts: List[Dict[str, Any]], batch_size: int = 16) -> List[List[float]]:
        return [self.embed_text(t.get("text", "")) for t in texts]


class TestAudioIntegration(unittest.TestCase):
    """Test audio integration with MultimodalVectorStore."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Need to mock Qwen3VL imports
        import sys
        
        # Create mock qwen3vl module
        mock_qwen3vl = MagicMock()
        mock_qwen3vl.Qwen3VLEmbeddingProvider = MockEmbeddingProvider
        
        @dataclass
        class MockMultimodalDocument:
            id: str
            content: str
            metadata: Dict[str, Any]
            embedding: Optional[List[float]] = None
            image_path: Optional[str] = None
            video_path: Optional[str] = None
        
        mock_qwen3vl.MultimodalDocument = MockMultimodalDocument
        mock_qwen3vl.HAS_QWEN3VL = True
        
        sys.modules['vl_rag_graph_rlm.rag.qwen3vl'] = mock_qwen3vl
        
        # Now import our modules
        from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore
        from vl_rag_graph_rlm.rag.parakeet import ParakeetTranscriptionProvider
        
        self.store_class = MultimodalVectorStore
        self.parakeet_class = ParakeetTranscriptionProvider
        
        self.mock_embedder = MockEmbeddingProvider(dim=128)
        self.mock_transcriber = MockParakeetProvider()
    
    def test_add_audio_with_transcription(self):
        """Test adding audio with transcription."""
        store = self.store_class(
            embedding_provider=self.mock_embedder,
            transcription_provider=self.mock_transcriber
        )
        
        # Add audio file
        doc_id = store.add_audio(
            audio_path="/path/to/meeting.mp3",
            metadata={"meeting": "team_sync"}
        )
        
        # Verify document added
        self.assertIn(doc_id, store.documents)
        doc = store.documents[doc_id]
        
        # Verify transcription used
        self.assertIn("mock transcription", doc.content)
        self.assertIn("meeting", doc.content)
        
        # Verify metadata
        self.assertEqual(doc.metadata["type"], "audio")
        self.assertEqual(doc.metadata["audio_path"], "/path/to/meeting.mp3")
        self.assertEqual(doc.metadata["meeting"], "team_sync")
        self.assertTrue(doc.metadata["transcribed"])
        
        # Verify embedding exists
        self.assertIsNotNone(doc.embedding)
        self.assertEqual(len(doc.embedding), 128)
    
    def test_add_audio_without_transcription(self):
        """Test adding audio without transcription."""
        store = self.store_class(
            embedding_provider=self.mock_embedder,
            transcription_provider=self.mock_transcriber
        )
        
        # Add audio with transcribe=False
        doc_id = store.add_audio(
            audio_path="/path/to/audio.wav",
            transcribe=False
        )
        
        doc = store.documents[doc_id]
        
        # Should use placeholder
        self.assertEqual(doc.content, "[Audio: /path/to/audio.wav]")
        self.assertFalse(doc.metadata["transcribed"])
    
    def test_add_audio_without_provider(self):
        """Test adding audio when no transcription provider available."""
        store = self.store_class(
            embedding_provider=self.mock_embedder,
            transcription_provider=None
        )
        
        doc_id = store.add_audio(audio_path="/path/to/audio.mp3")
        
        doc = store.documents[doc_id]
        
        # Should use placeholder
        self.assertEqual(doc.content, "[Audio: /path/to/audio.mp3]")
        self.assertFalse(doc.metadata["transcribed"])
    
    def test_get_audio_transcription(self):
        """Test retrieving transcription from audio document."""
        store = self.store_class(
            embedding_provider=self.mock_embedder,
            transcription_provider=self.mock_transcriber
        )
        
        # Add audio
        doc_id = store.add_audio(audio_path="/path/to/audio.mp3")
        
        # Get transcription
        transcript = store.get_audio_transcription(doc_id)
        
        # Should return the content (since it was transcribed)
        self.assertIsNotNone(transcript)
        self.assertIn("mock transcription", transcript)
    
    def test_get_audio_transcription_not_audio(self):
        """Test getting transcription for non-audio document."""
        store = self.store_class(
            embedding_provider=self.mock_embedder,
            transcription_provider=self.mock_transcriber
        )
        
        # Add text document
        doc_id = store.add_text("Some text content")
        
        # Should return None for non-audio
        transcript = store.get_audio_transcription(doc_id)
        self.assertIsNone(transcript)
    
    def test_audio_deduplication(self):
        """Test that duplicate audio files are deduplicated."""
        store = self.store_class(
            embedding_provider=self.mock_embedder,
            transcription_provider=self.mock_transcriber
        )
        
        # Add same audio twice
        doc_id1 = store.add_audio(audio_path="/path/to/audio.mp3")
        doc_id2 = store.add_audio(audio_path="/path/to/audio.mp3")
        
        # Should return same doc_id (deduplication)
        self.assertEqual(doc_id1, doc_id2)
        
        # Should only have one document
        self.assertEqual(len(store.documents), 1)
    
    def test_audio_search_integration(self):
        """Test that audio content is searchable."""
        store = self.store_class(
            embedding_provider=self.mock_embedder,
            transcription_provider=self.mock_transcriber
        )
        
        # Add audio
        store.add_audio(
            audio_path="/path/to/tech_meeting.mp3",
            metadata={"topic": "machine learning"}
        )
        
        # Add text
        store.add_text("Another document about AI")
        
        # Search should work
        self.mock_embedder.embed_text = lambda text, instruction=None: np.random.randn(128).tolist()
        results = store.search("machine learning discussion")
        
        # Should return results (even if mocked)
        self.assertIsInstance(results, list)
    
    def test_transcription_fallback_on_error(self):
        """Test that audio adds with placeholder if transcription fails."""
        # Create provider that raises error
        failing_provider = MockParakeetProvider()
        failing_provider.transcribe = lambda audio_path, **kwargs: (_ for _ in ()).throw(Exception("Transcription failed"))
        
        store = self.store_class(
            embedding_provider=self.mock_embedder,
            transcription_provider=failing_provider
        )
        
        # Add audio - should not raise, should fallback
        doc_id = store.add_audio(audio_path="/path/to/corrupt.mp3")
        
        doc = store.documents[doc_id]
        
        # Should use placeholder
        self.assertEqual(doc.content, "[Audio: /path/to/corrupt.mp3]")
        self.assertFalse(doc.metadata["transcribed"])
        self.assertIn("transcription_error", doc.metadata)


class TestParakeetProvider(unittest.TestCase):
    """Test ParakeetTranscriptionProvider directly."""
    
    def test_lazy_loading(self):
        """Test that model is not loaded on initialization."""
        from vl_rag_graph_rlm.rag.parakeet import ParakeetTranscriptionProvider
        
        provider = ParakeetTranscriptionProvider()
        
        # Model should not be loaded yet
        self.assertIsNone(provider._model)
        self.assertFalse(provider.get_model_info()["loaded"])
    
    def test_caching_disabled(self):
        """Test provider without caching."""
        from vl_rag_graph_rlm.rag.parakeet import ParakeetTranscriptionProvider
        
        provider = ParakeetTranscriptionProvider(
            use_cache=False,
            cache_dir=None
        )
        
        # Should not cache
        self.assertFalse(provider.use_cache)
        self.assertIsNone(provider.cache_dir)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAudioIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestParakeetProvider))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
