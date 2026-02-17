"""Embedding quantization for reduced memory footprint.

Provides int8 and binary quantization for vector embeddings,
reducing storage by 4x (int8) or 32x (binary) with minimal accuracy loss.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger("rlm.rag.quantization")


class EmbeddingQuantizer:
    """Quantize float32 embeddings to int8 or binary for storage efficiency.
    
    Int8: 4x memory reduction, ~98-99% accuracy retention
    Binary: 32x memory reduction, ~96% accuracy with rescoring
    
    Example:
        >>> quantizer = EmbeddingQuantizer(method='int8')
        >>> quantized = quantizer.quantize(embeddings)  # embeddings: np.ndarray float32
        >>> reconstructed = quantizer.dequantize(quantized)
    """
    
    def __init__(self, method: str = 'int8', calibration_data: Optional[np.ndarray] = None):
        """Initialize quantizer.
        
        Args:
            method: 'int8' or 'binary'
            calibration_data: Optional calibration embeddings for int8 range calculation
        """
        if method not in ('int8', 'binary'):
            raise ValueError(f"Method must be 'int8' or 'binary', got {method}")
        
        self.method = method
        self.calibration_data = calibration_data
        
        # Int8 calibration parameters
        self._min_val: Optional[float] = None
        self._max_val: Optional[float] = None
        self._scales: Optional[np.ndarray] = None
        
        if method == 'int8' and calibration_data is not None:
            self._calibrate_int8(calibration_data)
    
    def _calibrate_int8(self, data: np.ndarray) -> None:
        """Calculate quantization parameters from calibration data."""
        # Per-dimension min/max for better accuracy
        self._min_val = np.min(data, axis=0)
        self._max_val = np.max(data, axis=0)
        
        # Avoid division by zero
        range_val = self._max_val - self._min_val
        range_val = np.where(range_val == 0, 1.0, range_val)
        
        # Scale factor: map [min, max] to [-128, 127]
        self._scales = range_val / 255.0
        
        logger.info(f"Calibrated int8 quantizer on {len(data)} samples")
    
    def quantize(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Quantize float32 embeddings.
        
        Args:
            embeddings: Float32 array of shape (N, D)
            
        Returns:
            Dict with quantized data and metadata for dequantization
        """
        if self.method == 'int8':
            return self._quantize_int8(embeddings)
        else:
            return self._quantize_binary(embeddings)
    
    def _quantize_int8(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Quantize to int8 (4x storage reduction)."""
        # If no calibration, use per-batch statistics
        if self._scales is None:
            min_val = np.min(embeddings, axis=0)
            max_val = np.max(embeddings, axis=0)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1.0, range_val)
            scales = range_val / 255.0
        else:
            min_val = self._min_val
            scales = self._scales
        
        # Quantize: (x - min) / scale - 128 -> [-128, 127]
        quantized = ((embeddings - min_val) / scales - 128).astype(np.int8)
        
        return {
            'data': quantized,
            'method': 'int8',
            'min_val': min_val,
            'scales': scales,
            'shape': embeddings.shape,
            'dtype': 'float32',
        }
    
    def _quantize_binary(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Quantize to binary (32x storage reduction).
        
        Threshold at 0, pack bits into uint8 array.
        """
        # Normalize embeddings for consistent thresholding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms
        
        # Binary threshold at 0
        binary = (normalized > 0).astype(np.uint8)
        
        # Pack bits into uint8 (8 values per byte)
        n, d = embeddings.shape
        packed_dim = (d + 7) // 8  # Round up to nearest byte
        packed = np.packbits(binary, axis=1)
        
        return {
            'data': packed,
            'method': 'binary',
            'original_dim': d,
            'shape': embeddings.shape,
            'dtype': 'float32',
        }
    
    def dequantize(self, quantized: Dict[str, Any]) -> np.ndarray:
        """Dequantize back to float32.
        
        Args:
            quantized: Dict from quantize()
            
        Returns:
            Float32 embeddings
        """
        if quantized['method'] == 'int8':
            return self._dequantize_int8(quantized)
        else:
            return self._dequantize_binary(quantized)
    
    def _dequantize_int8(self, quantized: Dict[str, Any]) -> np.ndarray:
        """Dequantize int8 to float32."""
        data = quantized['data'].astype(np.float32)
        min_val = quantized['min_val']
        scales = quantized['scales']
        
        # Reverse: (x + 128) * scale + min
        return (data + 128) * scales + min_val
    
    def _dequantize_binary(self, quantized: Dict[str, Any]) -> np.ndarray:
        """Dequantize binary to float32 (approximate)."""
        packed = quantized['data']
        original_dim = quantized['original_dim']
        n = quantized['shape'][0]
        
        # Unpack bits
        binary = np.unpackbits(packed, axis=1)[:, :original_dim]
        
        # Convert to float32 (0 -> -1, 1 -> +1 for cosine similarity)
        return binary.astype(np.float32) * 2 - 1
    
    def compute_similarity(
        self,
        query: np.ndarray,
        quantized_docs: Dict[str, Any],
    ) -> np.ndarray:
        """Compute similarity between query and quantized documents.
        
        For binary: uses Hamming distance approximation (fast)
        For int8: dequantizes query and computes dot product
        """
        if quantized_docs['method'] == 'binary':
            return self._hamming_similarity(query, quantized_docs)
        else:
            # Dequantize and compute
            docs_float = self.dequantize(quantized_docs)
            return query @ docs_float.T
    
    def _hamming_similarity(
        self,
        query: np.ndarray,
        quantized_docs: Dict[str, Any],
    ) -> np.ndarray:
        """Compute approximate cosine similarity using Hamming distance.
        
        Much faster than dequantizing all documents.
        """
        # Quantize query to binary
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        query_binary = (query_norm > 0).astype(np.uint8)
        query_packed = np.packbits(query_binary)
        
        # Compute Hamming distance using XOR + popcount
        docs_packed = quantized_docs['data']
        
        # XOR each doc with query
        xor_result = docs_packed ^ query_packed
        
        # Count bits set (Hamming weight)
        # Using lookup table for efficiency
        hamming_dist = np.zeros(len(docs_packed))
        for i in range(len(xor_result)):
            # Count bits in each byte
            hamming_dist[i] = sum(bin(b).count('1') for b in xor_result[i])
        
        # Convert to similarity (lower distance = higher similarity)
        max_dist = quantized_docs['original_dim']
        return 1.0 - (hamming_dist / max_dist)


class QuantizedVectorStore:
    """Vector store with transparent quantization for memory efficiency.
    
    Stores embeddings in quantized form (int8 or binary), dequantizing
    only when needed for search.
    
    Example:
        >>> store = QuantizedVectorStore(method='int8')
        >>> store.add('doc1', embedding)  # Stores as int8
        >>> results = store.search(query_embedding, top_k=10)
    """
    
    def __init__(
        self,
        method: str = 'int8',
        calibration_data: Optional[np.ndarray] = None,
        cache_dequantized: bool = True,
    ):
        """Initialize quantized store.
        
        Args:
            method: 'int8' or 'binary'
            calibration_data: Optional calibration for int8
            cache_dequantized: Cache dequantized vectors for repeated searches
        """
        self.quantizer = EmbeddingQuantizer(method, calibration_data)
        self.doc_ids: List[str] = []
        self.quantized_embeddings: List[Dict[str, Any]] = []
        self.cache_dequantized = cache_dequantized
        self._dequantized_cache: Optional[np.ndarray] = None
        self._cache_dirty = True
    
    def add(self, doc_id: str, embedding: np.ndarray) -> None:
        """Add document with embedding."""
        quantized = self.quantizer.quantize(embedding.reshape(1, -1))
        self.doc_ids.append(doc_id)
        self.quantized_embeddings.append(quantized)
        self._cache_dirty = True
    
    def add_batch(self, doc_ids: List[str], embeddings: np.ndarray) -> None:
        """Add multiple documents efficiently."""
        quantized = self.quantizer.quantize(embeddings)
        # Split into individual entries
        for i, doc_id in enumerate(doc_ids):
            entry = {
                'data': quantized['data'][i:i+1],
                'method': quantized['method'],
                'shape': (1, quantized['shape'][1]),
            }
            if 'min_val' in quantized:
                entry['min_val'] = quantized['min_val']
                entry['scales'] = quantized['scales']
            if 'original_dim' in quantized:
                entry['original_dim'] = quantized['original_dim']
            
            self.doc_ids.append(doc_id)
            self.quantized_embeddings.append(entry)
        
        self._cache_dirty = True
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        exact: bool = False,
    ) -> List[Tuple[str, float]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query vector (float32)
            top_k: Number of results
            exact: If True, dequantize all and compute exact similarity
                  If False, use fast approximate similarity
            
        Returns:
            List of (doc_id, score) tuples
        """
        if not self.doc_ids:
            return []
        
        if exact and self.cache_dequantized:
            # Use cached dequantized embeddings
            docs_matrix = self._get_dequantized_matrix()
            
            # Normalize for cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            docs_norm = docs_matrix / (np.linalg.norm(docs_matrix, axis=1, keepdims=True) + 1e-10)
            
            similarities = docs_norm @ query_norm
        else:
            # Approximate similarity
            similarities = np.array([
                self.quantizer.compute_similarity(query_embedding, q)[0]
                for q in self.quantized_embeddings
            ])
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            (self.doc_ids[i], float(similarities[i]))
            for i in top_indices
        ]
    
    def _get_dequantized_matrix(self) -> np.ndarray:
        """Get dequantized embeddings matrix (cached)."""
        if self._cache_dirty or self._dequantized_cache is None:
            embeddings = [
                self.quantizer.dequantize(q)[0]
                for q in self.quantized_embeddings
            ]
            self._dequantized_cache = np.array(embeddings)
            self._cache_dirty = False
        
        return self._dequantized_cache
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.doc_ids:
            return {'total_bytes': 0, 'docs': 0}
        
        # Calculate actual storage
        total_bytes = sum(
            q['data'].nbytes
            for q in self.quantized_embeddings
        )
        
        # Estimate float32 equivalent
        dim = self.quantized_embeddings[0]['shape'][1]
        float32_bytes = len(self.doc_ids) * dim * 4
        
        return {
            'total_bytes': total_bytes,
            'float32_equivalent_bytes': float32_bytes,
            'compression_ratio': float32_bytes / total_bytes if total_bytes > 0 else 0,
            'docs': len(self.doc_ids),
            'method': self.quantizer.method,
        }


def quantize_embeddings_file(
    input_path: Path,
    output_path: Path,
    method: str = 'int8',
    calibration_samples: int = 1000,
) -> Dict[str, Any]:
    """Quantize embeddings from a numpy file.
    
    Args:
        input_path: Path to .npy file with float32 embeddings
        output_path: Path to save quantized embeddings
        method: 'int8' or 'binary'
        calibration_samples: Number of samples for calibration (int8 only)
        
    Returns:
        Statistics dict
    """
    embeddings = np.load(input_path)
    
    # Calibrate if needed
    calibration_data = None
    if method == 'int8' and len(embeddings) > calibration_samples:
        calibration_data = embeddings[:calibration_samples]
    
    quantizer = EmbeddingQuantizer(method, calibration_data)
    quantized = quantizer.quantize(embeddings)
    
    # Save
    np.savez(
        output_path,
        data=quantized['data'],
        method=quantized['method'],
        **{k: v for k, v in quantized.items() if k != 'data' and k != 'method'}
    )
    
    # Stats
    original_bytes = embeddings.nbytes
    quantized_bytes = quantized['data'].nbytes
    
    stats = {
        'original_bytes': original_bytes,
        'quantized_bytes': quantized_bytes,
        'compression_ratio': original_bytes / quantized_bytes,
        'method': method,
    }
    
    logger.info(f"Quantized {len(embeddings)} embeddings: {stats}")
    return stats
