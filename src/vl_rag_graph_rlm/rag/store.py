"""Vector store with provider-agnostic embedding support.

Supports any embedding model available through the existing providers:
- OpenAI (text-embedding-3-small, text-embedding-3-large)
- OpenRouter (various embedding models)
- LiteLLM (universal access to 100+ providers)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from vl_rag_graph_rlm.rag import SearchResult
from vl_rag_graph_rlm.clients import get_client, BaseLM

logger = logging.getLogger("rlm.rag.store")


@dataclass
class Document:
    """A document chunk with metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class EmbeddingClient:
    """Wrapper to get embeddings from any existing provider.
    
    Supports OpenAI, OpenRouter, and LiteLLM embedding endpoints.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        embedding_dim: int = 1536
    ):
        self.provider = provider
        self.model = model
        self.embedding_dim = embedding_dim
        
        # Initialize client
        client_kwargs = {"model_name": model}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["api_base"] = api_base
            
        self.client = get_client(provider, **client_kwargs)
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def embed(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        # Check cache
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            # Try native embedding support first
            if hasattr(self.client, 'embeddings'):
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[text]
                )
                embedding = response.data[0].embedding
            else:
                # Fallback: use completion with prompt engineering
                # This works with any provider that supports chat completions
                embedding = self._embed_via_completion(text)
            
            self._embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dim
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        return [self.embed(text) for text in texts]
    
    def _embed_via_completion(self, text: str) -> List[float]:
        """Generate embedding via completion API using structured output.
        
        This is a fallback for providers without native embedding endpoints.
        Uses a pseudo-embedding technique based on model's internal representation.
        """
        # Request the model to output a structured vector representation
        prompt = f"""Analyze this text and output a semantic fingerprint as a JSON array of {self.embedding_dim} floats between -1 and 1.
This should capture the meaning and key concepts of the text.

Text: {text[:1000]}

Output only the JSON array, nothing else."""

        try:
            response = self.client.completion(prompt)
            # Parse the response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            embedding = json.loads(response.strip())
            
            # Normalize to expected dimension
            if len(embedding) < self.embedding_dim:
                embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
            elif len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
                
            return embedding
            
        except Exception as e:
            logger.warning(f"Completion-based embedding failed: {e}")
            return [0.0] * self.embedding_dim


class SimpleVectorStore:
    """Simple in-memory vector store with persistence.
    
    For production use, replace with Milvus, Chroma, or Weaviate.
    """
    
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        storage_path: Optional[str] = None
    ):
        self.embedding_client = embedding_client
        self.documents: Dict[str, Document] = {}
        self.storage_path = storage_path

        # NumPy embedding matrix cache for vectorized search
        self._embedding_matrix: Optional[np.ndarray] = None
        self._matrix_doc_ids: List[str] = []
        self._matrix_dirty: bool = True
        
        if storage_path and os.path.exists(storage_path):
            self._load()
    
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> str:
        """Add a document to the store."""
        doc_id = doc_id or f"doc_{len(self.documents)}"
        metadata = metadata or {}
        
        # Generate embedding
        embedding = self.embedding_client.embed(content)
        
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata,
            embedding=embedding
        )
        
        self.documents[doc_id] = doc
        self._matrix_dirty = True
        
        if self.storage_path:
            self._save()
        
        return doc_id
    
    def add_batch(self, contents: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Add multiple documents."""
        metadatas = metadatas or [{}] * len(contents)
        
        # Batch embed
        embeddings = self.embedding_client.embed_batch(contents)
        
        ids = []
        for i, (content, metadata, embedding) in enumerate(zip(contents, metadatas, embeddings)):
            doc_id = f"doc_{len(self.documents) + i}"
            doc = Document(
                id=doc_id,
                content=content,
                metadata=metadata,
                embedding=embedding
            )
            self.documents[doc_id] = doc
            ids.append(doc_id)
        
        self._matrix_dirty = True

        if self.storage_path:
            self._save()
        
        return ids
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Optional[Callable[[Document], bool]] = None
    ) -> List[SearchResult]:
        """Search for similar documents (NumPy-vectorized)."""
        if not self.documents:
            return []
        
        query_embedding = self.embedding_client.embed(query)

        # Rebuild matrix if dirty
        if self._matrix_dirty or self._embedding_matrix is None:
            self._rebuild_embedding_matrix()

        if self._embedding_matrix is None or len(self._matrix_doc_ids) == 0:
            return []

        # Normalize query vector
        query_vec = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            return []
        query_vec = query_vec / norm

        # Single matrix multiply: (N, D) @ (D,) -> (N,)
        similarities = self._embedding_matrix @ query_vec

        # Apply filter mask if provided
        if filter_fn is not None:
            mask = np.array([
                filter_fn(self.documents[doc_id])
                for doc_id in self._matrix_doc_ids
            ], dtype=bool)
            similarities = np.where(mask, similarities, -np.inf)

        # Get top-k indices
        k = min(top_k, len(self._matrix_doc_ids))
        if k <= 0:
            return []

        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results: List[SearchResult] = []
        for idx in top_indices:
            idx_int = int(idx)
            sim = float(similarities[idx_int])
            if sim == -np.inf:
                continue
            doc_id = self._matrix_doc_ids[idx_int]
            doc = self.documents[doc_id]
            distance = 1.0 - sim
            results.append(SearchResult(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                semantic_score=distance,
            ))

        return results
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Optional[Callable[[Document], bool]] = None
    ) -> List[SearchResult]:
        """Simple keyword search."""
        if not self.documents:
            return []
        
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        results = []
        for doc in self.documents.values():
            if filter_fn and not filter_fn(doc):
                continue
            
            content_lower = doc.content.lower()
            
            # Calculate keyword match score
            matches = sum(1 for term in query_terms if term in content_lower)
            score = matches / len(query_terms) if query_terms else 0
            
            # Bonus for exact match
            if query_lower in content_lower:
                score += 1.0
            
            results.append(SearchResult(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                keyword_score=score * 100,
                semantic_score=0.0
            ))
        
        # Sort by keyword score
        results.sort(key=lambda x: x.keyword_score, reverse=True)
        
        return results[:top_k]
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Optional[Callable[[Document], bool]] = None
    ) -> List[SearchResult]:
        """Hybrid search combining dense and keyword search with RRF."""
        from vl_rag_graph_rlm.rag import HybridSearcher
        
        dense_results = self.search(query, top_k=top_k * 5, filter_fn=filter_fn)
        keyword_results = self.keyword_search(query, top_k=top_k * 5, filter_fn=filter_fn)
        
        searcher = HybridSearcher()
        return searcher.search(query, dense_results, keyword_results, top_k=top_k)
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._matrix_dirty = True
            if self.storage_path:
                self._save()
            return True
        return False

    def _rebuild_embedding_matrix(self) -> None:
        """Rebuild the normalized NumPy embedding matrix from documents."""
        doc_ids = []
        embeddings = []
        for doc_id, doc in self.documents.items():
            if doc.embedding is not None:
                doc_ids.append(doc_id)
                embeddings.append(doc.embedding)

        if not embeddings:
            self._embedding_matrix = None
            self._matrix_doc_ids = []
            self._matrix_dirty = False
            return

        mat = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embedding_matrix = mat / norms
        self._matrix_doc_ids = doc_ids
        self._matrix_dirty = False
    
    def _save(self):
        """Persist to disk."""
        data = {
            doc_id: {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding
            }
            for doc_id, doc in self.documents.items()
        }
        
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f)
    
    def _load(self):
        """Load from disk."""
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        self.documents = {
            doc_id: Document(
                id=d["id"],
                content=d["content"],
                metadata=d["metadata"],
                embedding=d["embedding"]
            )
            for doc_id, d in data.items()
        }


# Factory function
def create_vector_store(
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    storage_path: Optional[str] = None,
    embedding_dim: int = 1536
) -> SimpleVectorStore:
    """Create a vector store with the specified embedding provider."""
    embedding_client = EmbeddingClient(
        provider=provider,
        model=model,
        api_key=api_key,
        embedding_dim=embedding_dim
    )
    
    return SimpleVectorStore(embedding_client, storage_path)
