"""Unit tests for DocumentProcessor.

Tests document processing for PPTX, TXT, MD, PDF, DOCX, CSV, and Excel files.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from vl_rag_graph_rlm.document_processor import DocumentProcessor
            self.DocumentProcessor = DocumentProcessor
            self.processor = DocumentProcessor()
        except ImportError:
            self.skipTest("DocumentProcessor not available")

    def test_process_text_file(self):
        """Test processing a plain text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\n\nIt has multiple paragraphs.\n")
            temp_path = f.name
        
        try:
            result = self.processor.process_file(Path(temp_path))
            self.assertEqual(result["type"], "text")
            self.assertIn("test document", result["content"])
            self.assertTrue(len(result["chunks"]) > 0)
        finally:
            Path(temp_path).unlink()

    def test_process_markdown_file(self):
        """Test processing a markdown file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Heading\n\nSome content here.\n\n## Subheading\n\nMore content.")
            temp_path = f.name
        
        try:
            result = self.processor.process_file(Path(temp_path))
            self.assertEqual(result["type"], "text")
            self.assertIn("Heading", result["content"])
        finally:
            Path(temp_path).unlink()

    def test_sliding_window_chunks(self):
        """Test sliding window chunking utility."""
        from vl_rag_graph_rlm.document_processor import sliding_window_chunks
        
        text = "Word " * 100  # 500 chars with spaces
        chunks = sliding_window_chunks(text, chunk_size=100, overlap=20)
        
        self.assertTrue(len(chunks) > 0)
        # Each chunk should be roughly chunk_size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 120)  # Allow some flexibility for word boundaries

    def test_unsupported_file(self):
        """Test handling of unsupported file types."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("Some content")
            temp_path = f.name
        
        try:
            result = self.processor.process_file(Path(temp_path))
            self.assertEqual(result["type"], "unsupported")
        finally:
            Path(temp_path).unlink()


class TestSlidingWindowChunks(unittest.TestCase):
    """Test sliding window chunking functionality."""

    def test_empty_text(self):
        """Test chunking empty text."""
        from vl_rag_graph_rlm.document_processor import sliding_window_chunks
        result = sliding_window_chunks("", chunk_size=100, overlap=20)
        self.assertEqual(result, [])

    def test_short_text(self):
        """Test chunking text shorter than chunk_size."""
        from vl_rag_graph_rlm.document_processor import sliding_window_chunks
        text = "Short text"
        result = sliding_window_chunks(text, chunk_size=100, overlap=20)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Short text")

    def test_overlap_consistency(self):
        """Test that overlap creates consistent chunks."""
        from vl_rag_graph_rlm.document_processor import sliding_window_chunks
        text = "Word " * 50
        chunks = sliding_window_chunks(text, chunk_size=100, overlap=20)

        # Check that consecutive chunks share some content
        if len(chunks) >= 2:
            # The end of chunk 0 should overlap with start of chunk 1
            chunk0_end = chunks[0][-30:]
            chunk1_start = chunks[1][:30]
            # There should be some overlap
            self.assertTrue(
                any(word in chunk1_start for word in chunk0_end.split()),
                "Consecutive chunks should have overlapping content"
            )


class TestCSVProcessing(unittest.TestCase):
    """Test CSV/Excel document processing."""

    def test_process_csv(self):
        """Test processing a CSV file."""
        try:
            from vl_rag_graph_rlm.document_processor import DocumentProcessor
        except ImportError:
            self.skipTest("DocumentProcessor not available")
        
        import csv
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Age", "City"])
            writer.writerow(["Alice", "30", "NYC"])
            writer.writerow(["Bob", "25", "LA"])
            temp_path = f.name
        
        try:
            processor = DocumentProcessor()
            result = processor.process_file(Path(temp_path))
            
            # CSV should be processed as text
            self.assertEqual(result["type"], "tabular")
            # Content should include row information
            content = result["content"]
            self.assertIn("Alice", content)
            self.assertIn("NYC", content)
        finally:
            Path(temp_path).unlink()


class TestKeywordSearch(unittest.TestCase):
    """Test keyword/BM25 search functionality."""

    def test_basic_keyword_search(self):
        """Test basic keyword matching."""
        try:
            from vl_rag_graph_rlm.rag.store import SimpleVectorStore
        except ImportError:
            self.skipTest("SimpleVectorStore not available")

        store = SimpleVectorStore()
        store.add("The quick brown fox jumps over the lazy dog")
        store.add("Python is a programming language")
        store.add("Machine learning is fascinating")

        results = store.keyword_search("python programming", top_k=2)

        self.assertTrue(len(results) > 0)
        # Python document should be ranked higher
        self.assertTrue(any("python" in r.content.lower() for r in results))


class TestRRFFusion(unittest.TestCase):
    """Test Reciprocal Rank Fusion."""

    def test_rrf_basic_fusion(self):
        """Test basic RRF fusion of two result sets."""
        try:
            from vl_rag_graph_rlm.dynamic_hybrid_search import DynamicHybridSearcher
        except ImportError:
            self.skipTest("DynamicHybridSearcher not available")

        # Mock vector store
        mock_vector_store = Mock()
        mock_result1 = Mock()
        mock_result1.id = "doc1"
        mock_result1.semantic_score = 0.9
        mock_result2 = Mock()
        mock_result2.id = "doc2"
        mock_result2.semantic_score = 0.8
        mock_result3 = Mock()
        mock_result3.id = "doc3"
        mock_result3.semantic_score = 0.7
        mock_vector_store.search.return_value = [mock_result1, mock_result2, mock_result3]

        # Mock keyword index
        mock_keyword_index = Mock()
        mock_keyword_index.search.return_value = {
            "doc2": 0.95,
            "doc3": 0.85,
            "doc4": 0.75,
        }

        searcher = DynamicHybridSearcher(
            mock_vector_store,
            mock_keyword_index,
            default_dense_weight=4.0,
            default_keyword_weight=1.0,
        )

        results = searcher.search("test query", top_k=5, use_dynamic=False)

        self.assertTrue(len(results) > 0)
        # All unique docs should be present
        doc_ids = [r[0] for r in results]
        self.assertIn("doc1", doc_ids)  # Only in dense
        self.assertIn("doc2", doc_ids)  # In both
        self.assertIn("doc3", doc_ids)  # In both
        self.assertIn("doc4", doc_ids)  # Only in keyword


class TestCollectionCRUD(unittest.TestCase):
    """Test collection CRUD operations."""

    def test_create_collection(self):
        """Test creating a new collection."""
        try:
            from vl_rag_graph_rlm.collections import create_collection, collection_exists, delete_collection
        except ImportError:
            self.skipTest("Collections module not available")

        import tempfile
        import shutil

        # Create a temp directory for collections
        temp_dir = tempfile.mkdtemp()

        with patch('vl_rag_graph_rlm.collections._collections_base_dir', temp_dir):
            meta = create_collection("test-collection-unit")
            self.assertEqual(meta["name"], "test-collection-unit")
            self.assertTrue(collection_exists("test-collection-unit"))

            # Cleanup
            delete_collection("test-collection-unit")

        shutil.rmtree(temp_dir)

    def test_delete_collection(self):
        """Test deleting a collection."""
        try:
            from vl_rag_graph_rlm.collections import create_collection, delete_collection, collection_exists
        except ImportError:
            self.skipTest("Collections module not available")

        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()

        with patch('vl_rag_graph_rlm.collections._collections_base_dir', temp_dir):
            create_collection("to-delete-unit")
            self.assertTrue(collection_exists("to-delete-unit"))

            result = delete_collection("to-delete-unit")
            self.assertTrue(result)
            self.assertFalse(collection_exists("to-delete-unit"))

        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
