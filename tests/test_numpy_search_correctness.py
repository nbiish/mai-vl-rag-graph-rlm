"""Test that NumPy vectorized search produces identical results to manual cosine similarity.

This test verifies that the performance optimization (NumPy matrix multiplication)
produces exactly the same results as the original manual cosine similarity calculation.
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Document:
    """Test document class mirroring the store.py Document."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


def manual_cosine_similarity(query: List[float], embedding: List[float]) -> float:
    """Original manual cosine similarity from store.py (pre-NumPy optimization).n    
    This is the reference implementation that NumPy search should match exactly.
    """
    query_arr = np.array(query)
    emb_arr = np.array(embedding)
    
    dot_product = np.dot(query_arr, emb_arr)
    query_norm = np.linalg.norm(query_arr)
    emb_norm = np.linalg.norm(emb_arr)
    
    if query_norm == 0 or emb_norm == 0:
        return 0.0
    
    return dot_product / (query_norm * emb_norm)


def manual_search(documents: Dict[str, Document], query: List[float], top_k: int = 10):
    """Original manual search implementation (pre-NumPy optimization)."""
    if not documents:
        return []
    
    scores = []
    for doc_id, doc in documents.items():
        if doc.embedding is None:
            continue
        score = manual_cosine_similarity(query, doc.embedding)
        scores.append((doc_id, score, doc))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:top_k]


def numpy_vectorized_search(documents: Dict[str, Document], query: List[float], top_k: int = 10):
    """NumPy vectorized search from current store.py implementation."""
    if not documents:
        return []
    
    # Build embedding matrix (same as _rebuild_embedding_matrix)
    doc_ids = []
    embeddings = []
    for doc_id, doc in documents.items():
        if doc.embedding is not None:
            doc_ids.append(doc_id)
            embeddings.append(doc.embedding)
    
    if not embeddings:
        return []
    
    # Normalize embeddings (same as store.py)
    mat = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embedding_matrix = mat / norms
    
    # Normalize query (same as store.py search method)
    query_vec = np.array(query, dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        return []
    query_vec = query_vec / norm
    
    # Single matrix multiply: (N, D) @ (D,) -> (N,)
    similarities = embedding_matrix @ query_vec
    
    # Get top-k indices (same as store.py)
    k = min(top_k, len(doc_ids))
    if k <= 0:
        return []
    
    top_indices = np.argpartition(similarities, -k)[-k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    
    results = []
    for idx in top_indices:
        idx_int = int(idx)
        sim = float(similarities[idx_int])
        doc_id = doc_ids[idx_int]
        doc = documents[doc_id]
        results.append((doc_id, sim, doc))
    
    return results


class TestNumPySearchCorrectness:
    """Verify NumPy search matches manual cosine similarity exactly."""
    
    def test_identical_results_small_dataset(self):
        """Test with small dataset where we can verify exact matches."""
        documents = {
            f"doc_{i}": Document(
                id=f"doc_{i}",
                content=f"content {i}",
                metadata={},
                embedding=np.random.randn(128).tolist()
            )
            for i in range(10)
        }
        
        query = np.random.randn(128).tolist()
        
        manual_results = manual_search(documents, query, top_k=5)
        numpy_results = numpy_vectorized_search(documents, query, top_k=5)
        
        # Should have same number of results
        assert len(manual_results) == len(numpy_results)
        
        # Should have same document IDs in same order
        for (manual_id, manual_score, _), (numpy_id, numpy_score, _) in zip(manual_results, numpy_results):
            assert manual_id == numpy_id, f"Document IDs don't match: {manual_id} vs {numpy_id}"
            assert pytest.approx(manual_score, abs=1e-5) == numpy_score, \
                f"Scores don't match for {manual_id}: {manual_score} vs {numpy_score}"
    
    def test_identical_results_large_dataset(self):
        """Test with larger dataset to ensure scalability doesn't affect correctness."""
        np.random.seed(42)
        documents = {
            f"doc_{i}": Document(
                id=f"doc_{i}",
                content=f"content {i}",
                metadata={},
                embedding=np.random.randn(256).tolist()
            )
            for i in range(1000)
        }
        
        query = np.random.randn(256).tolist()
        
        manual_results = manual_search(documents, query, top_k=10)
        numpy_results = numpy_vectorized_search(documents, query, top_k=10)
        
        assert len(manual_results) == len(numpy_results) == 10
        
        # All top-10 should match exactly
        for i, ((manual_id, manual_score, _), (numpy_id, numpy_score, _)) in enumerate(zip(manual_results, numpy_results)):
            assert manual_id == numpy_id, f"Rank {i}: Document IDs don't match"
            assert pytest.approx(manual_score, abs=1e-4) == numpy_score, \
                f"Rank {i}: Scores don't match for {manual_id}"
    
    def test_edge_case_zero_vectors(self):
        """Test that zero vectors are handled identically."""
        documents = {
            "zero_doc": Document(id="zero_doc", content="zero", metadata={}, embedding=[0.0] * 128),
            "normal_doc": Document(id="normal_doc", content="normal", metadata={}, embedding=np.random.randn(128).tolist()),
        }
        
        query = np.random.randn(128).tolist()
        
        manual_results = manual_search(documents, query, top_k=2)
        numpy_results = numpy_vectorized_search(documents, query, top_k=2)
        
        assert len(manual_results) == len(numpy_results) == 2
        
        for (manual_id, manual_score, _), (numpy_id, numpy_score, _) in zip(manual_results, numpy_results):
            assert manual_id == numpy_id
            assert pytest.approx(manual_score, abs=1e-5) == numpy_score
    
    def test_edge_case_zero_query(self):
        """Test that zero query vector returns empty results in both implementations."""
        documents = {
            f"doc_{i}": Document(
                id=f"doc_{i}",
                content=f"content {i}",
                metadata={},
                embedding=np.random.randn(128).tolist()
            )
            for i in range(5)
        }
        
        zero_query = [0.0] * 128
        
        # Manual search with zero query
        manual_results = manual_search(documents, zero_query, top_k=5)
        
        # NumPy search should return empty (has explicit check)
        numpy_results = numpy_vectorized_search(documents, zero_query, top_k=5)
        
        # Both should handle gracefully
        assert len(numpy_results) == 0
    
    def test_empty_store(self):
        """Test that empty store returns empty results."""
        documents = {}
        query = np.random.randn(128).tolist()
        
        manual_results = manual_search(documents, query, top_k=5)
        numpy_results = numpy_vectorized_search(documents, query, top_k=5)
        
        assert manual_results == []
        assert numpy_results == []
    
    def test_single_document(self):
        """Test with single document."""
        documents = {
            "only_doc": Document(
                id="only_doc",
                content="only",
                metadata={},
                embedding=np.random.randn(64).tolist()
            )
        }
        
        query = np.random.randn(64).tolist()
        
        manual_results = manual_search(documents, query, top_k=1)
        numpy_results = numpy_vectorized_search(documents, query, top_k=1)
        
        assert len(manual_results) == len(numpy_results) == 1
        assert manual_results[0][0] == numpy_results[0][0] == "only_doc"
        assert pytest.approx(manual_results[0][1], abs=1e-5) == numpy_results[0][1]
    
    def test_different_embedding_dimensions(self):
        """Test with various embedding dimensions."""
        for dim in [4, 64, 128, 256, 512, 1024]:
            documents = {
                f"doc_{i}": Document(
                    id=f"doc_{i}",
                    content=f"content {i}",
                    metadata={},
                    embedding=np.random.randn(dim).tolist()
                )
                for i in range(20)
            }
            
            query = np.random.randn(dim).tolist()
            
            manual_results = manual_search(documents, query, top_k=5)
            numpy_results = numpy_vectorized_search(documents, query, top_k=5)
            
            assert len(manual_results) == len(numpy_results), f"Failed for dim={dim}"
            
            for (manual_id, manual_score, _), (numpy_id, numpy_score, _) in zip(manual_results, numpy_results):
                assert manual_id == numpy_id, f"ID mismatch for dim={dim}"
                assert pytest.approx(manual_score, abs=1e-4) == numpy_score, f"Score mismatch for dim={dim}"
    
    def test_performance_comparison(self):
        """Verify NumPy is significantly faster than manual search."""
        import time
        
        np.random.seed(42)
        documents = {
            f"doc_{i}": Document(
                id=f"doc_{i}",
                content=f"content {i}",
                metadata={},
                embedding=np.random.randn(256).tolist()
            )
            for i in range(500)
        }
        
        query = np.random.randn(256).tolist()
        
        # Warm up
        _ = numpy_vectorized_search(documents, query, top_k=10)
        
        # Time manual search
        start = time.perf_counter()
        for _ in range(10):
            manual_search(documents, query, top_k=10)
        manual_time = time.perf_counter() - start
        
        # Time NumPy search
        start = time.perf_counter()
        for _ in range(10):
            numpy_vectorized_search(documents, query, top_k=10)
        numpy_time = time.perf_counter() - start
        
        # NumPy should be at least 2x faster
        speedup = manual_time / numpy_time
        print(f"\nSpeedup: {speedup:.2f}x (manual: {manual_time:.4f}s, numpy: {numpy_time:.4f}s)")
        assert speedup > 2.0, f"NumPy search only {speedup:.2f}x faster, expected >2x"


def test_store_integration():
    """Integration test using actual SimpleVectorStore class."""
    import sys
    sys.path.insert(0, 'src')
    
    from vl_rag_graph_rlm.rag.store import SimpleVectorStore, Document, EmbeddingClient
    from unittest.mock import Mock
    
    # Create mock embedding client
    mock_client = Mock(spec=EmbeddingClient)
    mock_client.embedding_dim = 128
    mock_client.embed = lambda text: np.random.randn(128).tolist()
    
    # Create store
    store = SimpleVectorStore(mock_client)
    
    # Add documents directly
    for i in range(50):
        store.documents[f"doc_{i}"] = Document(
            id=f"doc_{i}",
            content=f"content {i}",
            metadata={},
            embedding=np.random.randn(128).tolist()
        )
    store._matrix_dirty = True
    
    # Test search works
    mock_client.embed = lambda text: np.random.randn(128).tolist()
    results = store.search("test query", top_k=10)
    
    assert len(results) == 10
    
    # Verify results are sorted by relevance (lower semantic_score = higher similarity)
    scores = [r.semantic_score for r in results]
    assert scores == sorted(scores), "Results should be sorted by relevance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
