"""SQLite backend for multimodal vector store.

Provides persistent storage using SQLite with proper indexing and
transaction support for the MultimodalVectorStore.
"""

import json
import sqlite3
import numpy as np
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from datetime import datetime


class SQLiteVectorStore:
    """SQLite-backed storage for vector embeddings and documents.
    
    Replaces JSON file storage with proper database backend for:
    - Better performance with large collections
    - Transaction safety
    - Concurrent read access
    - Indexed queries
    """
    
    def __init__(self, db_path: str):
        """Initialize SQLite vector store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_tables()
    
    def _ensure_tables(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    doc_type TEXT DEFAULT 'text',
                    metadata TEXT,  -- JSON
                    image_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS embeddings (
                    doc_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,  -- numpy bytes
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                        ON DELETE CASCADE
                );
                
                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(content_hash);
                CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type);
            """)
            conn.commit()
    
    def save_document(
        self,
        doc_id: str,
        content: str,
        content_hash: str,
        embedding: np.ndarray,
        doc_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None,
    ) -> None:
        """Save a document with its embedding."""
        with sqlite3.connect(self.db_path) as conn:
            # Insert or replace document
            conn.execute(
                """INSERT OR REPLACE INTO documents 
                   (doc_id, content, content_hash, doc_type, metadata, image_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    content,
                    content_hash,
                    doc_type,
                    json.dumps(metadata) if metadata else None,
                    image_path,
                ),
            )
            
            # Insert or replace embedding
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (doc_id, embedding) VALUES (?, ?)",
                (doc_id, embedding.tobytes()),
            )
            conn.commit()
    
    def load_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Load a single document by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT d.doc_id, d.content, d.doc_type, d.metadata, 
                          d.image_path, e.embedding
                   FROM documents d
                   LEFT JOIN embeddings e ON d.doc_id = e.doc_id
                   WHERE d.doc_id = ?""",
                (doc_id,),
            ).fetchone()
            
            if not row:
                return None
            
            return self._row_to_doc(row)
    
    def load_all_documents(self) -> Dict[str, Dict[str, Any]]:
        """Load all documents from the store."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT d.doc_id, d.content, d.doc_type, d.metadata, 
                          d.image_path, e.embedding
                   FROM documents d
                   LEFT JOIN embeddings e ON d.doc_id = e.doc_id"""
            ).fetchall()
            
            return {row[0]: self._row_to_doc(row) for row in rows}
    
    def _row_to_doc(self, row: tuple) -> Dict[str, Any]:
        """Convert database row to document dict."""
        doc_id, content, doc_type, metadata_json, image_path, embedding_bytes = row
        
        doc = {
            "id": doc_id,
            "content": content,
            "type": doc_type,
            "metadata": json.loads(metadata_json) if metadata_json else {},
        }
        
        if image_path:
            doc["image_path"] = image_path
        
        if embedding_bytes:
            # Reconstruct numpy array (assuming float32)
            doc["embedding"] = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        return doc
    
    def content_exists(self, content_hash: str) -> bool:
        """Check if content hash already exists."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            return row is not None
    
    def get_doc_id_by_hash(self, content_hash: str) -> Optional[str]:
        """Get document ID by content hash."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT doc_id FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            return row[0] if row else None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE doc_id = ?",
                (doc_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with sqlite3.connect(self.db_path) as conn:
            doc_count = conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            
            type_counts = conn.execute(
                "SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type"
            ).fetchall()
            
            return {
                "total_documents": doc_count,
                "by_type": {t: c for t, c in type_counts},
            }
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM embeddings")
            conn.execute("DELETE FROM documents")
            conn.commit()
    
    def set_metadata(self, key: str, value: str) -> None:
        """Store metadata key-value pair."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO store_metadata (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()
    
    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value by key."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM store_metadata WHERE key = ?",
                (key,),
            ).fetchone()
            return row[0] if row else None
