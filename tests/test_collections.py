"""Unit tests for collection CRUD operations.

Tests create, load, save, delete, list, and merge operations for collections.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json


class TestCollectionCRUD(unittest.TestCase):
    """Test collection CRUD operations."""

    def setUp(self):
        """Set up temporary collections root."""
        self.temp_dir = tempfile.mkdtemp()
        self.collections_root = Path(self.temp_dir) / "collections"
        self.collections_root.mkdir(parents=True, exist_ok=True)
        
        # Monkey-patch collections root
        try:
            from vl_rag_graph_rlm import collections
            self._original_root = collections.COLLECTIONS_ROOT
            collections.COLLECTIONS_ROOT = self.collections_root
            self.collections = collections
        except ImportError:
            self.skipTest("Collections module not available")

    def tearDown(self):
        """Clean up temporary directory."""
        # Restore original root
        if hasattr(self, '_original_root'):
            self.collections.COLLECTIONS_ROOT = self._original_root
        
        # Clean up temp dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_collection(self):
        """Test creating a new collection."""
        meta = self.collections.create_collection("test-coll", description="Test collection")
        
        self.assertEqual(meta["name"], "test-coll")
        self.assertEqual(meta["display_name"], "test-coll")
        self.assertEqual(meta["description"], "Test collection")
        self.assertIn("created", meta)
        self.assertIn("updated", meta)

    def test_collection_exists(self):
        """Test checking if collection exists."""
        # Should not exist yet
        self.assertFalse(self.collections.collection_exists("new-coll"))
        
        # Create it
        self.collections.create_collection("new-coll")
        
        # Should exist now
        self.assertTrue(self.collections.collection_exists("new-coll"))

    def test_load_collection_meta(self):
        """Test loading collection metadata."""
        # Create first
        created = self.collections.create_collection("load-test", description="For loading")
        
        # Load
        loaded = self.collections.load_collection_meta("load-test")
        
        self.assertEqual(loaded["name"], created["name"])
        self.assertEqual(loaded["description"], "For loading")

    def test_save_collection_meta(self):
        """Test saving updated metadata."""
        self.collections.create_collection("save-test")
        
        # Load and modify
        meta = self.collections.load_collection_meta("save-test")
        meta["custom_field"] = "custom_value"
        
        # Save
        self.collections.save_collection_meta("save-test", meta)
        
        # Reload and verify
        reloaded = self.collections.load_collection_meta("save-test")
        self.assertEqual(reloaded["custom_field"], "custom_value")

    def test_delete_collection(self):
        """Test deleting a collection."""
        self.collections.create_collection("delete-test")
        self.assertTrue(self.collections.collection_exists("delete-test"))
        
        # Delete
        result = self.collections.delete_collection("delete-test")
        self.assertTrue(result)
        
        # Should not exist anymore
        self.assertFalse(self.collections.collection_exists("delete-test"))

    def test_delete_nonexistent_collection(self):
        """Test deleting a collection that doesn't exist."""
        result = self.collections.delete_collection("does-not-exist")
        self.assertFalse(result)

    def test_list_collections(self):
        """Test listing all collections."""
        # Create a few
        self.collections.create_collection("coll-a")
        self.collections.create_collection("coll-b")
        self.collections.create_collection("coll-c")
        
        # List
        all_colls = self.collections.list_collections()
        names = [c["name"] for c in all_colls]
        
        self.assertIn("coll-a", names)
        self.assertIn("coll-b", names)
        self.assertIn("coll-c", names)
        self.assertEqual(len(all_colls), 3)

    def test_sanitize_name(self):
        """Test name sanitization."""
        test_cases = [
            ("Test Collection", "test-collection"),
            ("My_Collection_123", "my_collection_123"),
            ("  spaced  ", "spaced"),
        ]
        
        for input_name, expected_slug in test_cases:
            slug = self.collections._sanitize_name(input_name)
            self.assertEqual(slug, expected_slug)

    def test_record_source(self):
        """Test recording a source."""
        self.collections.create_collection("source-test")
        
        self.collections.record_source(
            "source-test",
            "/path/to/docs",
            doc_count=10,
            chunk_count=50,
            embedding_model="Qwen/Qwen3-VL-Embedding-2B",
            reranker_model="ms-marco-MiniLM-L-12-v2"
        )
        
        meta = self.collections.load_collection_meta("source-test")
        self.assertEqual(meta["document_count"], 10)
        self.assertEqual(meta["chunk_count"], 50)
        self.assertEqual(len(meta["sources"]), 1)
        self.assertEqual(meta["embedding_model"], "Qwen/Qwen3-VL-Embedding-2B")


class TestCollectionMerge(unittest.TestCase):
    """Test collection merging."""

    def setUp(self):
        """Set up temporary collections root."""
        self.temp_dir = tempfile.mkdtemp()
        self.collections_root = Path(self.temp_dir) / "collections"
        self.collections_root.mkdir(parents=True, exist_ok=True)
        
        try:
            from vl_rag_graph_rlm import collections
            self._original_root = collections.COLLECTIONS_ROOT
            collections.COLLECTIONS_ROOT = self.collections_root
            self.collections = collections
        except ImportError:
            self.skipTest("Collections module not available")

    def tearDown(self):
        """Clean up."""
        if hasattr(self, '_original_root'):
            self.collections.COLLECTIONS_ROOT = self._original_root
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_merge_collections(self):
        """Test merging two collections."""
        # Create source
        self.collections.create_collection("source")
        self.collections.record_source("source", "/docs", doc_count=5, chunk_count=20)
        
        # Create embeddings.json for source
        source_emb_path = self.collections_root / "source" / "embeddings.json"
        source_emb_path.write_text(json.dumps({
            "documents": {"0": {"content": "doc1"}, "1": {"content": "doc2"}},
            "embeddings": [{"doc_id": "0"}, {"doc_id": "1"}],
            "next_id": 2
        }))
        
        # Save KG for source
        self.collections.save_kg("source", "Source KG content")
        
        # Create target
        self.collections.create_collection("target")
        self.collections.record_source("target", "/other", doc_count=3, chunk_count=10)
        
        target_emb_path = self.collections_root / "target" / "embeddings.json"
        target_emb_path.write_text(json.dumps({
            "documents": {"0": {"content": "doc3"}},
            "embeddings": [{"doc_id": "0"}],
            "next_id": 1
        }))
        
        self.collections.save_kg("target", "Target KG content")
        
        # Merge source into target
        result = self.collections.merge_collections("source", "target")
        
        # Verify
        self.assertEqual(result["document_count"], 8)  # 5 + 3
        self.assertEqual(result["chunk_count"], 30)   # 20 + 10
        self.assertIn("source", result.get("merged_sources", []))


if __name__ == '__main__':
    unittest.main()
