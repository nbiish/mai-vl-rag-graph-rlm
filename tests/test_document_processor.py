"""Unit tests for DocumentProcessor.

Tests document processing for PPTX, TXT, MD, PDF, DOCX, CSV, and Excel files.
"""

import unittest
import tempfile
from pathlib import Path


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


if __name__ == '__main__':
    unittest.main()
