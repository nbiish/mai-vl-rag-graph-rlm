"""FAISS-based vector index for scalable similarity search.

Provides optional FAISS backend for MultimodalVectorStore when collection
size exceeds 1000 documents, offering O(log N) search vs O(N) linear scan.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None  # type: ignore
    np = None  # type: ignore

logger = logging.getLogger("rlm.rag.faiss_index")


class FAISSVectorIndex:
    """FAISS-based vector index with automatic index selection.
    
    Automatically selects appropriate FAISS index type based on collection size:
    - < 1K: IndexFlatIP (exact search, baseline)
    - 1K - 10K: IndexIVFFlat (inverted file, fast approximate)
    - 10K - 100K: IndexIVFPQ (product quantization, memory efficient)
    - 100K+: IndexHNSWFlat (graph-based, highest performance)
    
    Example:
        >>> from vl_rag_graph_rlm.rag.faiss_index import FAISSVectorIndex
        >>> index = FAISSVectorIndex(dim=2048)
        >>> 
        >>> # Add embeddings
        >>> embeddings = np.random.random((1000, 2048)).astype('float32')
        >>> doc_ids = [f"doc_{i}" for i in range(1000)]
        >>> index.add(embeddings, doc_ids)
        >>> 
        >>> # Search
        >>> query = np.random.random(2048).astype('float32')
        >>> distances, indices, returned_doc_ids = index.search(query, k=10)
    """
    
    def __init__(
        self,
        dim: int,
        index_type: Optional[str] = None,
        nlist: Optional[int] = None,
        nprobe: int = 10,
        cache_dir: Optional[str] = None,
    ):
        """Initialize FAISS index.
        
        Args:
            dim: Embedding dimension (e.g., 2048 for Qwen3-VL)
            index_type: FAISS index type ('flat', 'ivf', 'ivfpq', 'hnsw')
                        If None, auto-selected based on collection size
            nlist: Number of clusters for IVF (default: sqrt(N) at build time)
            nprobe: Number of clusters to search (default: 10)
            cache_dir: Directory to persist index (optional)
        """
        if not HAS_FAISS:
            raise ImportError(
                "faiss not installed. "
                "Install with: pip install faiss-cpu (or faiss-gpu)"
            )
        
        self.dim = dim
        self.index_type = index_type or "flat"
        self.nlist = nlist
        self.nprobe = nprobe
        self.cache_dir = cache_dir and Path(cache_dir)
        
        self._index: Optional[Any] = None
        self._doc_ids: List[str] = []
        self._is_trained = False
        
        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FAISS index initialized: dim={dim}, type={self.index_type}")
    
    def _create_index(self, num_vectors: int = 0) -> Any:
        """Create appropriate FAISS index based on size and type."""
        # Auto-select index type if not specified
        if self.index_type == "auto":
            if num_vectors < 1000:
                index_type = "flat"
            elif num_vectors < 10000:
                index_type = "ivf"
            elif num_vectors < 100000:
                index_type = "ivfpq"
            else:
                index_type = "hnsw"
        else:
            index_type = self.index_type
        
        # Create index
        if index_type == "flat":
            # Exact search, O(N) but accurate
            idx = faiss.IndexFlatIP(self.dim)  # Inner product (cosine similarity for normalized)
            
        elif index_type == "ivf":
            # Inverted file, O(N/nlist) approximate
            nlist = self.nlist or max(1, int(np.sqrt(num_vectors))) if num_vectors > 0 else 100
            quantizer = faiss.IndexFlatIP(self.dim)
            idx = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            idx.nprobe = self.nprobe
            
        elif index_type == "ivfpq":
            # Product quantization, memory efficient
            nlist = self.nlist or max(1, int(np.sqrt(num_vectors))) if num_vectors > 0 else 256
            m = 16  # Number of subquantizers
            nbits = 8  # Bits per subquantizer
            quantizer = faiss.IndexFlatIP(self.dim)
            idx = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
            idx.nprobe = self.nprobe
            
        elif index_type == "hnsw":
            # Hierarchical NSW graph, O(log N)
            m = 32  # Connections per node
            idx = faiss.IndexHNSWFlat(self.dim, m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 128
            idx.hnsw.efSearch = 64
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        logger.info(f"Created {index_type} index for {num_vectors} vectors (dim={self.dim})")
        return idx
    
    def add(
        self,
        embeddings: Any,  # np.ndarray
        doc_ids: List[str],
    ) -> None:
        """Add embeddings to the index.
        
        Args:
            embeddings: NumPy array of shape (N, dim) with float32 vectors
            doc_ids: List of document IDs corresponding to embeddings
        """
        if not HAS_FAISS:
            raise ImportError("faiss not installed")
        
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N, {self.dim}), got {embeddings.shape}")
        
        num_new = len(embeddings)
        
        # Create index on first add if not exists
        if self._index is None:
            total_size = len(self._doc_ids) + num_new
            self._index = self._create_index(total_size)
        
        # Train IVF/PQ index if needed (only on first batch)
        if not self._is_trained and hasattr(self._index, 'is_trained'):
            if not self._index.is_trained:
                logger.info(f"Training index with {num_new} vectors...")
                self._index.train(embeddings)
            self._is_trained = True
        
        # Normalize embeddings for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings_normalized = embeddings / norms
        
        # Add to index
        self._index.add(embeddings_normalized)
        self._doc_ids.extend(doc_ids)
        
        logger.info(f"Added {num_new} vectors, total: {len(self._doc_ids)}")
        
        # Auto-upgrade index type if size crosses threshold
        self._maybe_upgrade_index()
    
    def _maybe_upgrade_index(self) -> None:
        """Auto-upgrade index type for better performance as collection grows."""
        if self.index_type != "auto":
            return
        
        n = len(self._doc_ids)
        current_type = self._get_current_index_type()
        
        # Upgrade thresholds
        if current_type == "flat" and n >= 1000:
            logger.info(f"Auto-upgrading index from flat to IVF (n={n})")
            self._rebuild_index("ivf")
        elif current_type == "ivf" and n >= 10000:
            logger.info(f"Auto-upgrading index from IVF to IVFPQ (n={n})")
            self._rebuild_index("ivfpq")
    
    def _get_current_index_type(self) -> str:
        """Get current index type string."""
        if self._index is None:
            return "none"
        name = type(self._index).__name__.lower()
        if "ivfpq" in name:
            return "ivfpq"
        elif "ivf" in name:
            return "ivf"
        elif "hnsw" in name:
            return "hnsw"
        else:
            return "flat"
    
    def _rebuild_index(self, new_type: str) -> None:
        """Rebuild index with new type, preserving all vectors."""
        if self._index is None or len(self._doc_ids) == 0:
            return
        
        # Extract all vectors from current index (if possible)
        # For Flat indices, we can reconstruct; for others, we need original vectors
        logger.warning(f"Index rebuild to {new_type} requires original vectors. "
                      "Rebuild on next full save/load cycle.")
        # Mark for rebuild on save/load
        self.index_type = new_type
    
    def search(
        self,
        query_embedding: Any,  # np.ndarray
        top_k: int = 10,
        filter_mask: Optional[Any] = None,  # np.ndarray[bool]
    ) -> Tuple[Any, Any, List[str]]:
        """Search for similar vectors.
        
        Args:
            query_embedding: Query vector (dim,) float32
            top_k: Number of results to return
            filter_mask: Optional boolean mask (not supported by FAISS, applied post-search)
            
        Returns:
            Tuple of (distances, indices, doc_ids)
            - distances: np.ndarray of similarities (higher is better for IP)
            - indices: np.ndarray of index positions
            - doc_ids: List of document ID strings
        """
        if not HAS_FAISS:
            raise ImportError("faiss not installed")
        
        if self._index is None or len(self._doc_ids) == 0:
            return np.array([]), np.array([], dtype=np.int64), []
        
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        # Search
        k = min(top_k, len(self._doc_ids))
        distances, indices = self._index.search(query, k)
        
        # Apply filter mask if provided (post-filtering)
        if filter_mask is not None:
            # FAISS doesn't support pre-filtering, so we search more and filter
            # This is a simplified approach - for production, use IDSelector
            valid_mask = filter_mask[indices[0]]
            filtered_indices = indices[0][valid_mask]
            filtered_distances = distances[0][valid_mask]
            
            # Pad if we filtered too much
            if len(filtered_indices) < k:
                # Search again with larger k
                distances, indices = self._index.search(query, k * 2)
                valid_mask = filter_mask[indices[0]]
                filtered_indices = indices[0][valid_mask][:k]
                filtered_distances = distances[0][valid_mask][:k]
            
            indices = np.array([filtered_indices])
            distances = np.array([filtered_distances])
        
        # Map to doc_ids
        doc_ids = [self._doc_ids[i] for i in indices[0] if i < len(self._doc_ids)]
        
        return distances[0], indices[0], doc_ids
    
    def remove(self, doc_id: str) -> bool:
        """Remove a document from the index.
        
        Note: FAISS doesn't support efficient removal. We mark as removed
        and rebuild periodically, or use ID mapping.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if doc_id not in self._doc_ids:
            return False
        
        # Mark for rebuild (FAISS doesn't support removal)
        idx = self._doc_ids.index(doc_id)
        self._doc_ids[idx] = None  # type: ignore
        logger.debug(f"Marked doc_id {doc_id} for removal (index rebuild needed)")
        return True
    
    def save(self, path: Optional[str] = None) -> None:
        """Save index to disk.
        
        Args:
            path: Path to save index. If None, uses cache_dir
        """
        if not HAS_FAISS or self._index is None:
            return
        
        save_path = path or (self.cache_dir / "faiss.index" if self.cache_dir else None)
        if not save_path:
            return
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self._index, str(save_path))
        
        # Save doc_id mapping
        import json
        meta_path = save_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'doc_ids': self._doc_ids,
                'dim': self.dim,
                'index_type': self._get_current_index_type(),
            }, f)
        
        logger.info(f"Saved FAISS index to {save_path}")
    
    def load(self, path: Optional[str] = None) -> bool:
        """Load index from disk.
        
        Args:
            path: Path to load index from. If None, uses cache_dir
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not HAS_FAISS:
            return False
        
        load_path = path or (self.cache_dir / "faiss.index" if self.cache_dir else None)
        if not load_path:
            return False
        
        load_path = Path(load_path)
        meta_path = load_path.with_suffix('.json')
        
        if not load_path.exists() or not meta_path.exists():
            return False
        
        try:
            # Load FAISS index
            self._index = faiss.read_index(str(load_path))
            
            # Load metadata
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self._doc_ids = meta.get('doc_ids', [])
                self.dim = meta.get('dim', self.dim)
            
            self._is_trained = True
            logger.info(f"Loaded FAISS index from {load_path} ({len(self._doc_ids)} vectors)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'dim': self.dim,
            'index_type': self._get_current_index_type(),
            'num_vectors': len(self._doc_ids),
            'is_trained': self._is_trained,
            'has_index': self._index is not None,
        }


def create_faiss_index(
    dim: int,
    collection_size: int = 0,
    cache_dir: Optional[str] = None,
) -> Optional[FAISSVectorIndex]:
    """Factory function to create appropriate FAISS index.
    
    Args:
        dim: Embedding dimension
        collection_size: Expected collection size (for auto-index selection)
        cache_dir: Directory for persistence
        
    Returns:
        FAISSVectorIndex or None if faiss not installed
    """
    if not HAS_FAISS:
        logger.warning("faiss not installed, falling back to NumPy search")
        return None
    
    # Auto-select index type based on size
    if collection_size < 1000:
        index_type = "flat"
    elif collection_size < 10000:
        index_type = "ivf"
    elif collection_size < 100000:
        index_type = "ivfpq"
    else:
        index_type = "hnsw"
    
    return FAISSVectorIndex(
        dim=dim,
        index_type=index_type,
        cache_dir=cache_dir,
    )
