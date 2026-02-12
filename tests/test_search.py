"""Unit tests for keyword search and RRF fusion.

Tests BM25 keyword search, Reciprocal Rank Fusion, and related utilities.
"""

import unittest
from typing import List, Dict, Any


class TestBM25KeywordSearch(unittest.TestCase):
    """Test BM25 keyword search functionality."""

    def test_bm25_basic_scoring(self):
        """Test basic BM25 scoring works."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self.skipTest("rank-bm25 not installed")
        
        corpus = [
            "machine learning is great",
            "deep learning uses neural networks",
            "natural language processing with transformers"
        ]
        
        tokenized = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        
        # Query about machine learning
        scores = bm25.get_scores(["machine", "learning"])
        
        # First doc should score higher
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

    def test_simple_token_overlap_fallback(self):
        """Test fallback token overlap when BM25 not available."""
        import re
        
        documents = [
            {"content": "machine learning algorithms"},
            {"content": "deep learning neural networks"},
            {"content": "data science and statistics"}
        ]
        
        query = "machine learning"
        query_terms = set(re.findall(r"\w+", query.lower()))
        
        results = []
        for doc in documents:
            content_lower = doc["content"].lower()
            matches = sum(1 for term in query_terms if term in content_lower)
            if matches > 0:
                score = matches / len(query_terms)
                results.append((doc, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # First result should be the machine learning doc
        self.assertEqual(results[0][0]["content"], "machine learning algorithms")
        self.assertEqual(results[0][1], 1.0)  # Perfect match


class TestReciprocalRankFusion(unittest.TestCase):
    """Test Reciprocal Rank Fusion implementation."""

    def setUp(self):
        """Set up RRF instance."""
        try:
            from vl_rag_graph_rlm.rag.reciprocal_rank_fusion import ReciprocalRankFusion
            self.rrf = ReciprocalRankFusion(k=60)
        except ImportError:
            self.skipTest("RRF not available")

    def test_rrf_single_list(self):
        """Test RRF with a single ranked list."""
        results = [
            {"id": 1, "content": "doc1", "score": 0.9},
            {"id": 2, "content": "doc2", "score": 0.8},
            {"id": 3, "content": "doc3", "score": 0.7},
        ]
        
        fused = self.rrf.fuse([results])
        
        # Should preserve order with single list
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0].id, 1)
        self.assertEqual(fused[1].id, 2)
        self.assertEqual(fused[2].id, 3)

    def test_rrf_two_lists(self):
        """Test RRF fusion with two different ranked lists."""
        # Dense search results
        dense = [
            {"id": 1, "content": "doc1", "score": 0.9},
            {"id": 2, "content": "doc2", "score": 0.8},
            {"id": 4, "content": "doc4", "score": 0.6},
        ]
        
        # Keyword search results (different order)
        keyword = [
            {"id": 2, "content": "doc2", "score": 1.0},
            {"id": 3, "content": "doc3", "score": 0.9},
            {"id": 1, "content": "doc1", "score": 0.7},
        ]
        
        fused = self.rrf.fuse([dense, keyword])
        
        # Should have 4 unique docs
        self.assertEqual(len(fused), 4)
        
        # Doc 2 appears in both lists at good ranks, should be high
        doc2_rank = next(i for i, r in enumerate(fused) if r.id == 2)
        self.assertLess(doc2_rank, 2)  # Should be in top 2

    def test_rrf_with_weights(self):
        """Test RRF with weighted inputs."""
        dense = [
            {"id": 1, "content": "doc1", "score": 0.9},
            {"id": 2, "content": "doc2", "score": 0.8},
        ]
        
        keyword = [
            {"id": 2, "content": "doc2", "score": 1.0},
            {"id": 1, "content": "doc1", "score": 0.7},
        ]
        
        # Weight dense 2x more than keyword
        fused = self.rrf.fuse([dense, keyword], weights=[2.0, 1.0])
        
        # Doc 1 should rank higher due to dense weight
        doc1_idx = next(i for i, r in enumerate(fused) if r.id == 1)
        doc2_idx = next(i for i, r in enumerate(fused) if r.id == 2)
        
        # Doc 1 should be ranked higher (lower index) due to weighting
        self.assertLess(doc1_idx, doc2_idx)


class TestSearchResultTypes(unittest.TestCase):
    """Test SearchResult dataclass and related types."""

    def test_search_result_creation(self):
        """Test creating SearchResult objects."""
        try:
            from vl_rag_graph_rlm.rag.types import SearchResult
        except ImportError:
            self.skipTest("SearchResult type not available")
        
        result = SearchResult(
            id="doc1",
            content="test content",
            metadata={"type": "text"},
            semantic_score=0.9,
            keyword_score=0.8,
            composite_score=0.85
        )
        
        self.assertEqual(result.id, "doc1")
        self.assertEqual(result.content, "test content")
        self.assertEqual(result.composite_score, 0.85)


if __name__ == '__main__':
    unittest.main()
