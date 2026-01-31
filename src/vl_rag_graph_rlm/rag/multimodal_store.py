"""Multimodal vector store for documents with text, images, and video.

Extends SimpleVectorStore to support Qwen3-VL embeddings for multimodal content.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from vl_rag_graph_rlm.rag import SearchResult
from vl_rag_graph_rlm.rag.store import SimpleVectorStore, Document
from vl_rag_graph_rlm.rag.qwen3vl import Qwen3VLEmbeddingProvider, MultimodalDocument

logger = logging.getLogger("rlm.rag.multimodal_store")


class MultimodalVectorStore:
    """Vector store supporting text, image, and video documents.
    
    Uses Qwen3-VL for generating unified embeddings across modalities.
    
    Example:
        >>> from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder
        >>> from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore
        >>> 
        >>> embedder = create_qwen3vl_embedder("Qwen/Qwen3-VL-Embedding-2B")
        >>> store = MultimodalVectorStore(embedder)
        >>> 
        >>> # Add text document
        >>> store.add_text("Climate change is caused by greenhouse gases.")
        >>> 
        >>> # Add image document
        >>> store.add_image("path/to/chart.png", description="Temperature chart")
        >>> 
        >>> # Add PDF page as image
        >>> store.add_pdf_page("document.pdf", page_num=1)
        >>> 
        >>> # Search with text query
        >>> results = store.search("What causes climate change?")
        >>> 
        >>> # Search with image query
        >>> results = store.search_image("path/to/query_image.jpg")
    """
    
    def __init__(
        self,
        embedding_provider: Qwen3VLEmbeddingProvider,
        storage_path: Optional[str] = None,
        use_qwen_reranker: bool = False,
        reranker_provider: Optional[Any] = None
    ):
        """Initialize multimodal vector store.
        
        Args:
            embedding_provider: Qwen3-VL embedding provider
            storage_path: Optional path for persistence
            use_qwen_reranker: Whether to use Qwen3-VL reranker
            reranker_provider: Optional Qwen3-VL reranker provider
        """
        self.embedding_provider = embedding_provider
        self.storage_path = storage_path
        self.documents: Dict[str, MultimodalDocument] = {}
        self.use_qwen_reranker = use_qwen_reranker
        self.reranker = reranker_provider
        
        if storage_path and os.path.exists(storage_path):
            self._load()
    
    def add_text(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        instruction: Optional[str] = None
    ) -> str:
        """Add a text document.
        
        Args:
            content: Text content
            metadata: Optional metadata
            doc_id: Optional document ID
            instruction: Optional embedding instruction
            
        Returns:
            Document ID
        """
        doc_id = doc_id or f"doc_{len(self.documents)}"
        metadata = metadata or {}
        
        # Generate embedding
        embedding = self.embedding_provider.embed_text(
            content,
            instruction=instruction
        )
        
        doc = MultimodalDocument(
            id=doc_id,
            content=content,
            metadata=metadata,
            embedding=embedding
        )
        
        self.documents[doc_id] = doc
        
        if self.storage_path:
            self._save()
        
        return doc_id
    
    def add_image(
        self,
        image_path: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        instruction: Optional[str] = None
    ) -> str:
        """Add an image document.
        
        Args:
            image_path: Path to image file
            description: Optional text description
            metadata: Optional metadata
            doc_id: Optional document ID
            instruction: Optional embedding instruction
            
        Returns:
            Document ID
        """
        doc_id = doc_id or f"img_{len(self.documents)}"
        metadata = metadata or {}
        metadata["type"] = "image"
        metadata["image_path"] = image_path
        
        # Generate multimodal embedding
        if description:
            embedding = self.embedding_provider.embed_multimodal(
                text=description,
                image=image_path,
                instruction=instruction
            )
            content = description
        else:
            embedding = self.embedding_provider.embed_image(
                image_path,
                instruction=instruction
            )
            content = f"[Image: {image_path}]"
        
        doc = MultimodalDocument(
            id=doc_id,
            content=content,
            metadata=metadata,
            embedding=embedding,
            image_path=image_path
        )
        
        self.documents[doc_id] = doc
        
        if self.storage_path:
            self._save()
        
        return doc_id
    
    def add_video(
        self,
        video_path: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        fps: float = 1.0,
        max_frames: int = 64,
        instruction: Optional[str] = None
    ) -> str:
        """Add a video document.
        
        Args:
            video_path: Path to video file
            description: Optional text description
            metadata: Optional metadata
            doc_id: Optional document ID
            fps: Frame sampling rate
            max_frames: Maximum frames to sample
            instruction: Optional embedding instruction
            
        Returns:
            Document ID
        """
        doc_id = doc_id or f"vid_{len(self.documents)}"
        metadata = metadata or {}
        metadata["type"] = "video"
        metadata["video_path"] = video_path
        
        # Generate multimodal embedding
        if description:
            embedding = self.embedding_provider.embed_multimodal(
                text=description,
                video=video_path,
                fps=fps,
                max_frames=max_frames,
                instruction=instruction
            )
            content = description
        else:
            embedding = self.embedding_provider.embed_video(
                video_path,
                fps=fps,
                max_frames=max_frames,
                instruction=instruction
            )
            content = f"[Video: {video_path}]"
        
        doc = MultimodalDocument(
            id=doc_id,
            content=content,
            metadata=metadata,
            embedding=embedding,
            video_path=video_path
        )
        
        self.documents[doc_id] = doc
        
        if self.storage_path:
            self._save()
        
        return doc_id
    
    def add_pdf_page(
        self,
        pdf_path: str,
        page_num: int,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        dpi: int = 200,
        instruction: Optional[str] = None
    ) -> str:
        """Add a PDF page as an image document.
        
        Requires pdf2image: pip install pdf2image
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            description: Optional text description
            metadata: Optional metadata
            doc_id: Optional document ID
            dpi: DPI for PDF rendering
            instruction: Optional embedding instruction
            
        Returns:
            Document ID
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "pdf2image not installed. "
                "Install with: pip install pdf2image"
            )
        
        doc_id = doc_id or f"pdf_{len(self.documents)}_p{page_num}"
        metadata = metadata or {}
        metadata["type"] = "pdf_page"
        metadata["pdf_path"] = pdf_path
        metadata["page_num"] = page_num
        
        # Convert PDF page to image
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=dpi)
        if not images:
            raise ValueError(f"Could not extract page {page_num} from {pdf_path}")
        
        image = images[0]
        
        # Generate embedding
        if description:
            embedding = self.embedding_provider.embed_multimodal(
                text=description,
                image=image,
                instruction=instruction
            )
            content = description
        else:
            embedding = self.embedding_provider.embed_image(
                image,
                instruction=instruction
            )
            content = f"[PDF Page {page_num}: {pdf_path}]"
        
        doc = MultimodalDocument(
            id=doc_id,
            content=content,
            metadata=metadata,
            embedding=embedding
        )
        
        self.documents[doc_id] = doc
        
        if self.storage_path:
            self._save()
        
        return doc_id
    
    def add_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dpi: int = 200,
        instruction: Optional[str] = None
    ) -> List[str]:
        """Add all or selected pages from a PDF.
        
        Args:
            pdf_path: Path to PDF file
            pages: Optional list of page numbers (1-indexed, all if None)
            metadata: Optional metadata (applied to all pages)
            dpi: DPI for PDF rendering
            instruction: Optional embedding instruction
            
        Returns:
            List of document IDs
        """
        try:
            from pdf2image import convert_from_path
            from pdf2image.exceptions import PDFPageCountError
        except ImportError:
            raise ImportError(
                "pdf2image not installed. "
                "Install with: pip install pdf2image"
            )
        
        # Get total pages
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
        except ImportError:
            # Fallback: convert all and count
            logger.warning("PyMuPDF not installed, converting all pages")
            images = convert_from_path(pdf_path, dpi=dpi)
            total_pages = len(images)
            pages_to_process = [(i+1, img) for i, img in enumerate(images)]
        else:
            pages_to_process = pages if pages else list(range(1, total_pages + 1))
        
        doc_ids = []
        for page_num in (pages_to_process if isinstance(pages_to_process[0], int) else [p[0] for p in pages_to_process]):
            page_metadata = (metadata or {}).copy()
            page_metadata["total_pages"] = total_pages
            
            doc_id = self.add_pdf_page(
                pdf_path,
                page_num,
                metadata=page_metadata,
                dpi=dpi,
                instruction=instruction
            )
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Optional[Callable[[MultimodalDocument], bool]] = None,
        instruction: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for similar documents using text query.
        
        Args:
            query: Text query
            top_k: Number of results
            filter_fn: Optional filter function
            instruction: Optional embedding instruction
            
        Returns:
            List of search results
        """
        if not self.documents:
            return []
        
        query_embedding = self.embedding_provider.embed_text(
            query,
            instruction=instruction
        )
        
        return self._search_with_embedding(
            query_embedding,
            top_k,
            filter_fn
        )
    
    def search_image(
        self,
        image: Union[str, Any],
        top_k: int = 10,
        filter_fn: Optional[Callable[[MultimodalDocument], bool]] = None,
        instruction: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for similar documents using image query.
        
        Args:
            image: Image path or PIL Image
            top_k: Number of results
            filter_fn: Optional filter function
            instruction: Optional embedding instruction
            
        Returns:
            List of search results
        """
        if not self.documents:
            return []
        
        query_embedding = self.embedding_provider.embed_image(
            image,
            instruction=instruction
        )
        
        return self._search_with_embedding(
            query_embedding,
            top_k,
            filter_fn
        )
    
    def search_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Any]] = None,
        video: Optional[Union[str, List]] = None,
        top_k: int = 10,
        filter_fn: Optional[Callable[[MultimodalDocument], bool]] = None,
        instruction: Optional[str] = None
    ) -> List[SearchResult]:
        """Search using multimodal query.
        
        Args:
            text: Optional text query
            image: Optional image query
            video: Optional video query
            top_k: Number of results
            filter_fn: Optional filter function
            instruction: Optional embedding instruction
            
        Returns:
            List of search results
        """
        if not self.documents:
            return []
        
        query_embedding = self.embedding_provider.embed_multimodal(
            text=text,
            image=image,
            video=video,
            instruction=instruction
        )
        
        return self._search_with_embedding(
            query_embedding,
            top_k,
            filter_fn
        )
    
    def rerank(
        self,
        query: Dict[str, Any],
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Rerank results using Qwen3-VL reranker.
        
        Args:
            query: Query dict with text/image/video
            results: Initial search results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        if not self.reranker or not self.use_qwen_reranker:
            logger.warning("No reranker configured, returning original results")
            return results[:top_k] if top_k else results
        
        # Format documents for reranking
        documents = []
        for result in results:
            doc = self.documents.get(result.id)
            if doc:
                doc_dict: Dict[str, Any] = {"text": doc.content}
                if doc.image_path:
                    doc_dict["image"] = doc.image_path
                if doc.video_path:
                    doc_dict["video"] = doc.video_path
                documents.append(doc_dict)
        
        # Rerank
        reranked = self.reranker.rerank(
            query=query,
            documents=documents
        )
        
        # Reorder results
        reordered = []
        for idx, score in reranked:
            result = results[idx]
            result.composite_score = score * 100
            reordered.append(result)
        
        return reordered[:top_k] if top_k else reordered
    
    def hybrid_search_with_rerank(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 100,
        filter_fn: Optional[Callable[[MultimodalDocument], bool]] = None,
        instruction: Optional[str] = None
    ) -> List[SearchResult]:
        """Full pipeline: embed search + reranking.
        
        Args:
            query: Text query
            top_k: Final number of results
            rerank_top_k: Number of results to rerank
            filter_fn: Optional filter function
            instruction: Optional embedding instruction
            
        Returns:
            Reranked search results
        """
        # Initial retrieval
        results = self.search(
            query,
            top_k=rerank_top_k,
            filter_fn=filter_fn,
            instruction=instruction
        )
        
        if not self.use_qwen_reranker or not self.reranker:
            return results[:top_k]
        
        # Rerank
        query_dict = {"text": query}
        if instruction:
            query_dict["instruction"] = instruction
        
        return self.rerank(query_dict, results, top_k=top_k)
    
    def _search_with_embedding(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_fn: Optional[Callable[[MultimodalDocument], bool]]
    ) -> List[SearchResult]:
        """Search using pre-computed query embedding."""
        import math
        
        results = []
        for doc in self.documents.values():
            if filter_fn and not filter_fn(doc):
                continue
            
            if doc.embedding is None:
                continue
            
            # Cosine similarity
            dot = sum(x * y for x, y in zip(query_embedding, doc.embedding))
            norm_q = math.sqrt(sum(x * x for x in query_embedding))
            norm_d = math.sqrt(sum(x * x for x in doc.embedding))
            
            if norm_q == 0 or norm_d == 0:
                similarity = 0.0
            else:
                similarity = dot / (norm_q * norm_d)
            
            distance = 1.0 - similarity
            
            results.append(SearchResult(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                semantic_score=distance
            ))
        
        results.sort(key=lambda x: x.semantic_score)
        return results[:top_k]
    
    def get(self, doc_id: str) -> Optional[MultimodalDocument]:
        """Get document by ID."""
        return self.documents.get(doc_id)
    
    def delete(self, doc_id: str) -> bool:
        """Delete document by ID."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            if self.storage_path:
                self._save()
            return True
        return False
    
    def _save(self):
        """Persist to disk."""
        data = {
            doc_id: {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding,
                "image_path": doc.image_path,
                "video_path": doc.video_path
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
            doc_id: MultimodalDocument(
                id=d["id"],
                content=d["content"],
                metadata=d["metadata"],
                embedding=d["embedding"],
                image_path=d.get("image_path"),
                video_path=d.get("video_path")
            )
            for doc_id, d in data.items()
        }


# Factory function
def create_multimodal_store(
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
    storage_path: Optional[str] = None,
    use_reranker: bool = False,
    reranker_model: str = "Qwen/Qwen3-VL-Reranker-2B",
    **model_kwargs
) -> MultimodalVectorStore:
    """Create a multimodal vector store with Qwen3-VL.
    
    Args:
        model_name: Embedding model name
        storage_path: Optional storage path
        use_reranker: Whether to use reranker
        reranker_model: Reranker model name
        **model_kwargs: Additional model loading kwargs
        
    Returns:
        Configured MultimodalVectorStore
    """
    from vl_rag_graph_rlm.rag.qwen3vl import create_qwen3vl_embedder, create_qwen3vl_reranker
    
    embedder = create_qwen3vl_embedder(model_name, **model_kwargs)
    
    reranker = None
    if use_reranker:
        reranker = create_qwen3vl_reranker(reranker_model, **model_kwargs)
    
    return MultimodalVectorStore(
        embedding_provider=embedder,
        storage_path=storage_path,
        use_qwen_reranker=use_reranker,
        reranker_provider=reranker
    )
