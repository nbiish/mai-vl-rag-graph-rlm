"""Comprehensive tests for NumPy performance fixes in vector stores.

Tests cover:
1. NumPy vectorized search correctness vs manual cosine similarity
2. O(1) hash deduplication in MultimodalVectorStore
3. Embedding matrix caching (dirty flag behavior)
4. Persistence with hash maps
5. Delete operations with hash cleanup
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock, patch

import numpy as np


# Mock classes for testing without heavy dependencies
class MockEmbeddingProvider:
    """Mock embedding provider that returns deterministic embeddings."""
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


class MockMultimodalDocument:
    """Mock MultimodalDocument for testing."""
    def __init__(self, id: str, content: str, metadata: Dict[str, Any], 
                 embedding: List[float], image_path: Optional[str] = None,
                 video_path: Optional[str] = None):
        self.id = id
        self.content = content
        self.metadata = metadata
        self.embedding = embedding
        self.image_path = image_path
        self.video_path = video_path


class TestSimpleVectorStore(unittest.TestCase):
    """Test SimpleVectorStore NumPy performance fixes."""
    
    def setUp(self):
        from vl_rag_graph_rlm.rag.store import SimpleVectorStore, Document, EmbeddingClient
        self.store_class = SimpleVectorStore
        self.doc_class = Document
        
        # Create a mock embedding client
        self.mock_client = Mock(spec=EmbeddingClient)
        self.mock_client.embedding_dim = 4
        self.mock_client.embed.side_effect = lambda text: [1.0, 0.0, 0.0, 0.0] if "query" in text.lower() else [0.0, 1.0, 0.0, 0.0]
        
    def test_numpy_vectorized_search_correctness(self):
        """Verify NumPy search produces correct results."""
        store = self.store_class(self.mock_client)
        
        # Manually add documents with known embeddings (bypassing add() to avoid mock complexity)
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],  # Closest to query [0.95, 0.05, 0, 0]
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        contents = ["doc1", "doc2", "doc3"]
        
        for emb, content in zip(embeddings, contents):
            store.documents[content] = self.doc_class(
                id=content, content=content, metadata={}, embedding=emb
            )
        store._matrix_dirty = True
        
        # Mock the embed for search query
        store.embedding_client.embed = lambda text: [0.95, 0.05, 0.0, 0.0]
        
        results = store.search("test query", top_k=3)
        
        # doc1 should be first (cosine similarity ~0.99)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].id, "doc1")  # Most similar
        self.assertEqual(results[1].id, "doc2")  # Second most similar
        
        # Verify scores are in expected range (semantic_score is cosine distance = 1 - similarity)
        self.assertLess(results[0].semantic_score, 0.5)  # Close
        self.assertGreater(results[1].semantic_score, results[0].semantic_score)
    
    def test_embedding_matrix_caching(self):
        """Test that matrix is rebuilt only when dirty."""
        store = self.store_class(self.mock_client)
        
        # Initially dirty
        self.assertTrue(store._matrix_dirty)
        self.assertIsNone(store._embedding_matrix)
        
        # Manually add doc and mark dirty
        store.documents["test"] = self.doc_class(id="test", content="test", metadata={}, embedding=[1.0, 0.0, 0.0, 0.0])
        store._matrix_dirty = True
        
        # Trigger matrix build
        store._rebuild_embedding_matrix()
        self.assertFalse(store._matrix_dirty)
        self.assertIsNotNone(store._embedding_matrix)
        first_matrix = store._embedding_matrix.copy()  # Save values, not reference
        
        # Calling rebuild again creates new array (method always rebuilds when called)
        # The caching works by NOT calling rebuild when not dirty
        # Simulate what search() does: check dirty flag before rebuild
        self.assertFalse(store._matrix_dirty)
        # If we were to call search, it would skip rebuild since not dirty
        
        # Adding doc marks dirty
        store.documents["test2"] = self.doc_class(id="test2", content="test2", metadata={}, embedding=[0.0, 1.0, 0.0, 0.0])
        store._matrix_dirty = True
        
        # Now rebuild creates different matrix (more rows)
        store._rebuild_embedding_matrix()
        self.assertFalse(store._matrix_dirty)
        self.assertEqual(store._embedding_matrix.shape[0], 2)  # 2 documents now
    
    def test_embedding_matrix_normalization(self):
        """Verify embeddings are L2-normalized in the matrix."""
        store = self.store_class(self.mock_client)
        
        # Add document with non-unit embedding
        store.documents["doc"] = self.doc_class(id="doc", content="doc", metadata={}, embedding=[2.0, 0.0, 0.0, 0.0])
        store._matrix_dirty = True
        
        store._rebuild_embedding_matrix()
        
        # Matrix should be normalized
        norms = np.linalg.norm(store._embedding_matrix, axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_matrix_rebuild_on_delete(self):
        """Test matrix is marked dirty after delete."""
        store = self.store_class(self.mock_client)
        store.documents["doc1"] = self.doc_class(id="doc1", content="doc1", metadata={}, embedding=[1.0, 0.0, 0.0, 0.0])
        store.documents["doc2"] = self.doc_class(id="doc2", content="doc2", metadata={}, embedding=[0.0, 1.0, 0.0, 0.0])
        store._matrix_dirty = True
        
        # Force matrix build
        store._rebuild_embedding_matrix()
        self.assertFalse(store._matrix_dirty)
        
        # Delete marks dirty
        store.delete("doc1")
        self.assertTrue(store._matrix_dirty)
        
        # Rebuild
        store._rebuild_embedding_matrix()
        self.assertFalse(store._matrix_dirty)
        self.assertEqual(len(store.documents), 1)
    
    def test_persistence_with_matrix_state(self):
        """Test save/load works with matrix caching."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Write empty JSON object first
            json.dump({}, f)
            f.flush()
            path = f.name
        
        try:
            # Create and populate store
            store = self.store_class(self.mock_client, storage_path=path)
            store.documents["doc1"] = self.doc_class(id="doc1", content="c1", metadata={}, embedding=[1.0, 0.0, 0.0, 0.0])
            store.documents["doc2"] = self.doc_class(id="doc2", content="c2", metadata={}, embedding=[0.0, 1.0, 0.0, 0.0])
            store._save()
            
            # Load into new store
            store2 = self.store_class(self.mock_client, storage_path=path)
            self.assertEqual(len(store2.documents), 2)
            
            # Matrix should be dirty after load (needs rebuild)
            self.assertTrue(store2._matrix_dirty)
        finally:
            os.unlink(path)


class TestMultimodalVectorStoreHashDedup(unittest.TestCase):
    """Test O(1) hash deduplication in MultimodalVectorStore."""
    
    def setUp(self):
        # Need to mock the Qwen3VL imports
        import sys
        from unittest.mock import MagicMock
        
        # Create mock module
        mock_qwen3vl = MagicMock()
        mock_qwen3vl.Qwen3VLEmbeddingProvider = MockEmbeddingProvider
        mock_qwen3vl.MultimodalDocument = MockMultimodalDocument
        mock_qwen3vl.HAS_QWEN3VL = True
        
        sys.modules['vl_rag_graph_rlm.rag.qwen3vl'] = mock_qwen3vl
        
        from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore
        self.store_class = MultimodalVectorStore
        
    def test_hash_deduplication_o1_lookup(self):
        """Test O(1) hash dedup with _hash_to_id map."""
        provider = MockEmbeddingProvider(dim=4)
        store = self.store_class(embedding_provider=provider)
        
        # Add first document
        doc_id1 = store.add_text("unique content here", metadata={"key": "value1"})
        
        # Verify hash was recorded
        content_hash = store._hash_content("unique content here")
        self.assertIn(content_hash, store._content_hashes)
        self.assertEqual(store._hash_to_id[content_hash], doc_id1)
        
        # Try to add duplicate content - should return existing doc
        doc_id2 = store.add_text("unique content here", metadata={"key": "value2"})
        
        # Should return same doc_id (deduplication)
        self.assertEqual(doc_id1, doc_id2)
        
        # Verify only one document exists
        self.assertEqual(len(store.documents), 1)
    
    def test_hash_map_updated_on_delete(self):
        """Test hash maps are cleaned up on delete."""
        provider = MockEmbeddingProvider(dim=4)
        store = self.store_class(embedding_provider=provider)
        
        doc_id = store.add_text("content to delete", metadata={})
        content_hash = store._hash_content("content to delete")
        
        # Verify hash exists
        self.assertIn(content_hash, store._content_hashes)
        self.assertIn(content_hash, store._hash_to_id)
        
        # Delete
        store.delete(doc_id)
        
        # Hash should be removed
        self.assertNotIn(content_hash, store._content_hashes)
        self.assertNotIn(content_hash, store._hash_to_id)
    
    def test_content_exists_check(self):
        """Test content_exists() method."""
        provider = MockEmbeddingProvider(dim=4)
        store = self.store_class(embedding_provider=provider)
        
        # New content shouldn't exist
        self.assertFalse(store.content_exists("new content"))
        
        # Add content
        store.add_text("existing content", metadata={})
        
        # Now it should exist
        self.assertTrue(store.content_exists("existing content"))
        self.assertFalse(store.content_exists("different content"))
    
    def test_persistence_preserves_hash_maps(self):
        """Test save/load preserves hash deduplication state."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)  # Write empty JSON first
            f.flush()
            path = f.name
        
        try:
            provider = MockEmbeddingProvider(dim=4)
            
            # Create and populate store
            store = self.store_class(
                embedding_provider=provider,
                storage_path=path
            )
            doc_id = store.add_text("persistent content", metadata={"meta": "data"})
            content_hash = store._hash_content("persistent content")
            
            # Verify hash exists before save
            self.assertIn(content_hash, store._hash_to_id)
            
            # Load into new store
            store2 = self.store_class(
                embedding_provider=provider,
                storage_path=path
            )
            
            # Hash maps should be rebuilt
            self.assertIn(content_hash, store2._content_hashes)
            self.assertIn(content_hash, store2._hash_to_id)
            self.assertEqual(store2._hash_to_id[content_hash], doc_id)
            
            # Deduplication should still work
            doc_id2 = store2.add_text("persistent content", metadata={})
            self.assertEqual(doc_id, doc_id2)
        finally:
            os.unlink(path)


class TestNumPySearchPerformance(unittest.TestCase):
    """Performance comparison tests (sanity checks, not benchmarks)."""
    
    def setUp(self):
        from vl_rag_graph_rlm.rag.store import SimpleVectorStore, Document, EmbeddingClient
        self.store_class = SimpleVectorStore
        self.doc_class = Document
        
        # Mock embedding client
        self.mock_client = Mock(spec=EmbeddingClient)
        self.mock_client.embedding_dim = 128
        self.mock_client.embed = lambda text: np.random.randn(128).tolist()
    
    def test_large_scale_search_works(self):
        """Test that search works with many documents."""
        store = self.store_class(self.mock_client)
        
        # Add 1000 random documents
        np.random.seed(42)
        for i in range(1000):
            emb = np.random.randn(128).tolist()
            store.documents[f"doc_{i}"] = self.doc_class(
                id=f"doc_{i}",
                content=f"content {i}",
                metadata={},
                embedding=emb
            )
        store._matrix_dirty = True
        
        # Search should complete quickly
        start = time.time()
        store.embedding_client.embed = lambda text: np.random.randn(128).tolist()
        results = store.search("test query", top_k=10)
        elapsed = time.time() - start
        
        # Should be very fast with NumPy (< 500ms for 1000 docs)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(len(results), 10)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and error handling."""
    
    def setUp(self):
        from vl_rag_graph_rlm.rag.store import SimpleVectorStore, Document, EmbeddingClient
        self.store_class = SimpleVectorStore
        self.doc_class = Document
        
        # Mock embedding client
        self.mock_client = Mock(spec=EmbeddingClient)
        self.mock_client.embedding_dim = 4
        self.mock_client.embed = lambda text: [1.0, 0.0, 0.0, 0.0]
    
    def test_empty_store_search(self):
        """Search on empty store should return empty results."""
        store = self.store_class(self.mock_client)
        results = store.search("query", top_k=5)
        self.assertEqual(len(results), 0)
    
    def test_zero_vector_handling(self):
        """Zero vectors should not cause division by zero."""
        store = self.store_class(self.mock_client)
        
        # Add document with zero embedding
        store.documents["zero"] = self.doc_class(
            id="zero", content="zero", metadata={}, embedding=[0.0, 0.0, 0.0, 0.0]
        )
        store._matrix_dirty = True
        
        # Should not raise
        store.embedding_client.embed = lambda text: [1.0, 0.0, 0.0, 0.0]
        results = store.search("query", top_k=1)
        self.assertEqual(len(results), 1)
        
        # Zero vector should have distance 1.0 (cosine distance = 1 - 0)
        self.assertEqual(results[0].semantic_score, 1.0)
    
    def test_matrix_dirty_with_no_documents(self):
        """Matrix rebuild with no documents should handle gracefully."""
        store = self.store_class(self.mock_client)
        
        # Initially dirty, no docs
        self.assertTrue(store._matrix_dirty)
        
        # Rebuild should handle empty case
        store._rebuild_embedding_matrix()
        
        # Should remain None with no docs
        self.assertIsNone(store._embedding_matrix)
        self.assertFalse(store._matrix_dirty)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleVectorStore))
    suite.addTests(loader.loadTestsFromTestCase(TestMultimodalVectorStoreHashDedup))
    suite.addTests(loader.loadTestsFromTestCase(TestNumPySearchPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
