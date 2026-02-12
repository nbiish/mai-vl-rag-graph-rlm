"""RAG Context Provider for integration with RLM core.

This module integrates the hybrid search capabilities into the RLM system,
allowing RLM to use retrieved context instead of raw context.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from vl_rag_graph_rlm.rag import SearchResult, HybridSearcher
from vl_rag_graph_rlm.rag.store import SimpleVectorStore, EmbeddingClient

# Optional multimodal support
try:
    from vl_rag_graph_rlm.rag.multimodal_store import MultimodalVectorStore, create_multimodal_store
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False

logger = logging.getLogger("rlm.rag.provider")


@dataclass
class RAGConfig:
    """Configuration for RAG context retrieval."""
    top_k: int = 10
    dense_weight: float = 4.0
    keyword_weight: float = 1.0
    rerank: bool = True
    context_format: str = "detailed"  # "simple", "detailed", "citations"
    max_context_length: int = 8000


class RAGContextProvider:
    """Provides retrieved context for RLM using hybrid search.
    
    Integrates with existing RLM core to replace raw context with
    retrieved, relevant context from a vector store.
    
    Example:
        >>> from vl_rag_graph_rlm import RLM
        >>> from vl_rag_graph_rlm.rag import RAGContextProvider, create_vector_store
        >>>
        >>> # Create vector store with OpenAI embeddings
        >>> store = create_vector_store(provider="openai", model="text-embedding-3-small")
        >>>
        >>> # Add documents
        >>> store.add("Document content here...", metadata={"source": "doc1", "page": 1})
        >>>
        >>> # Create RAG provider
        >>> rag = RAGContextProvider(store, top_k=5)
        >>>
        >>> # Use with RLM
        >>> rlm = RLM(provider="openrouter", model="gpt-4o")
        >>> 
        >>> # Retrieve context and complete
        >>> context = rag.retrieve("What is the main idea?")
        >>> result = rlm.completion("Summarize the main idea", context=context)
    """
    
    def __init__(
        self,
        vector_store: SimpleVectorStore,
        config: Optional[RAGConfig] = None
    ):
        self.store = vector_store
        self.config = config or RAGConfig()
        self.searcher = HybridSearcher(
            dense_weight=self.config.dense_weight,
            keyword_weight=self.config.keyword_weight
        )
        self.query_history: List[str] = []
    
    def retrieve(self, query: str, filter_fn: Optional[Callable] = None) -> str:
        """Retrieve relevant context for a query.
        
        Args:
            query: The search query
            filter_fn: Optional filter function for documents
            
        Returns:
            Formatted context string
        """
        # Track query for potential query expansion
        self.query_history.append(query)
        
        # Perform hybrid search
        results = self._hybrid_search(query, filter_fn)
        
        if not results:
            return "No relevant documents found."
        
        # Format results based on config
        return self._format_context(results)
    
    def retrieve_with_sources(
        self,
        query: str,
        filter_fn: Optional[Callable] = None
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Retrieve context with source metadata.
        
        Returns:
            Tuple of (formatted_context, sources_list)
        """
        results = self._hybrid_search(query, filter_fn)
        
        if not results:
            return "No relevant documents found.", []
        
        context = self._format_context(results)
        sources = self._extract_sources(results)
        
        return context, sources
    
    def _hybrid_search(
        self,
        query: str,
        filter_fn: Optional[Callable] = None
    ) -> List[SearchResult]:
        """Execute hybrid search."""
        dense_results = self.store.search(
            query,
            top_k=self.config.top_k * 2,
            filter_fn=filter_fn
        )
        
        keyword_results = self.store.keyword_search(
            query,
            top_k=self.config.top_k * 2,
            filter_fn=filter_fn
        )
        
        return self.searcher.search(
            query,
            dense_results,
            keyword_results,
            top_k=self.config.top_k,
            rerank=self.config.rerank
        )
    
    def _format_context(self, results: List[SearchResult]) -> str:
        """Format search results into context string."""
        if self.config.context_format == "simple":
            return self._format_simple(results)
        elif self.config.context_format == "detailed":
            return self._format_detailed(results)
        elif self.config.context_format == "citations":
            return self._format_citations(results)
        else:
            return self._format_detailed(results)
    
    def _format_simple(self, results: List[SearchResult]) -> str:
        """Simple concatenation of content."""
        return "\n\n".join(r.content for r in results)
    
    def _format_detailed(self, results: List[SearchResult]) -> str:
        """Detailed format with metadata."""
        parts = []
        for i, result in enumerate(results, 1):
            metadata = result.metadata
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "N/A")
            score = result.composite_score or (1.0 - result.semantic_score) * 100
            
            part = f"[{i}] Source: {source} (Page {page}) [Relevance: {score:.1f}%]\n{result.content}"
            parts.append(part)
        
        return "\n\n---\n\n".join(parts)
    
    def _format_citations(self, results: List[SearchResult]) -> str:
        """Format with citation markers for attribution."""
        parts = []
        for i, result in enumerate(results, 1):
            part = f"[{i}] {result.content}"
            parts.append(part)
        
        return "\n\n".join(parts)
    
    def _extract_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Extract source metadata from results."""
        sources = []
        for result in results:
            sources.append({
                "id": result.id,
                "content": result.content,
                "metadata": result.metadata,
                "score": result.composite_score or (1.0 - result.semantic_score) * 100
            })
        return sources


class RAGEnhancedVLRAGGraphRLM:
    """RLM with integrated RAG capabilities.
    
    Combines RLM with RAG context retrieval for a complete
    retrieval-augmented generation system.
    
    Example:
        >>> from vl_rag_graph_rlm.rag import RAGEnhancedVLRAGGraphRLM, create_vector_store
        >>>
        >>> # Initialize
        >>> rag_rlm = RAGEnhancedVLRAGGraphRLM(
        ...     llm_provider="openrouter",
        ...     llm_model="gpt-4o",
        ...     embedding_provider="openai",
        ...     embedding_model="text-embedding-3-small"
        ... )
        >>>
        >>> # Add documents
        >>> rag_rlm.add_documents([
        ...     {"content": "Document 1...", "metadata": {"source": "doc1"}},
        ...     {"content": "Document 2...", "metadata": {"source": "doc2"}}
        ... ])
        >>>
        >>> # Query with automatic retrieval
        >>> result = rag_rlm.query("What is the main topic?")
        >>> print(result.response)
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        llm_api_key: Optional[str] = None,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: Optional[str] = None,
        storage_path: Optional[str] = None,
        rag_config: Optional[RAGConfig] = None
    ):
        from vl_rag_graph_rlm import RLM
        
        # Initialize RLM
        self.rlm = RLM(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key
        )
        
        # Initialize vector store
        self.store = create_vector_store(
            provider=embedding_provider,
            model=embedding_model,
            api_key=embedding_api_key,
            storage_path=storage_path
        )
        
        # Initialize RAG provider
        self.rag = RAGContextProvider(self.store, rag_config)
    
    def add_document(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Add a single document."""
        return self.store.add(content, metadata)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents.
        
        Args:
            documents: List of dicts with 'content' and optional 'metadata'
        """
        contents = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        return self.store.add_batch(contents, metadatas)
    
    def query(
        self,
        query: str,
        filter_fn: Optional[Callable] = None,
        **rlm_kwargs
    ):
        """Execute a RAG-enhanced query.
        
        1. Retrieves relevant context using hybrid search
        2. Passes context to RLM for completion
        
        Args:
            query: User query
            filter_fn: Optional filter for document retrieval
            **rlm_kwargs: Additional arguments for RLM
            
        Returns:
            RLMChatCompletion with retrieved context attached
        """
        # Retrieve context
        context = self.rag.retrieve(query, filter_fn)
        
        # Execute RLM with retrieved context
        result = self.rlm.completion(query, context, **rlm_kwargs)
        
        # Attach retrieval metadata
        result.retrieved_context = context
        
        return result
    
    def query_with_sources(
        self,
        query: str,
        filter_fn: Optional[Callable] = None,
        **rlm_kwargs
    ):
        """Query with source attribution."""
        context, sources = self.rag.retrieve_with_sources(query, filter_fn)
        
        result = self.rlm.completion(query, context, **rlm_kwargs)
        
        result.retrieved_context = context
        result.sources = sources
        
        return result


class MultimodalVLRAGGraphRLM:
    """Multimodal RAG using Qwen3-VL for text, image, and video retrieval.
    
    Supports querying across all modalities in a unified embedding space.
    
    Example:
        >>> from vl_rag_graph_rlm.rag.provider import MultimodalVLRAGGraphRLM
        >>> 
        >>> rag = MultimodalVLRAGGraphRLM(
        ...     llm_provider="openrouter",
        ...     llm_model="gpt-4o",
        ...     embedding_model="Qwen/Qwen3-VL-Embedding-2B"
        ... )
        >>> 
        >>> # Add documents
        >>> rag.add_pdf("document.pdf")
        >>> rag.add_image("diagram.png", description="System architecture")
        >>> 
        >>> # Query (text, image, or both)
        >>> result = rag.query("What is the system architecture?")
        >>> 
        >>> # Query with image
        >>> result = rag.query_image("query_image.jpg")
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        llm_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        use_reranker: bool = True,
        reranker_model: Optional[str] = None,
        storage_path: Optional[str] = None,
        **model_kwargs
    ):
        if not HAS_MULTIMODAL:
            raise ImportError(
                "Multimodal RAG requires Qwen3-VL dependencies. "
                "Install with: pip install torch transformers pillow qwen-vl-utils"
            )
        
        from vl_rag_graph_rlm import RLM
        
        # Initialize RLM
        self.rlm = RLM(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key
        )
        
        # Initialize multimodal store
        self.store = create_multimodal_store(
            model_name=embedding_model,
            storage_path=storage_path,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            **model_kwargs
        )
    
    def add_text(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Add a text document."""
        return self.store.add_text(content, metadata)
    
    def add_image(
        self,
        image_path: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add an image document."""
        return self.store.add_image(image_path, description, metadata)
    
    def add_video(
        self,
        video_path: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a video document."""
        return self.store.add_video(video_path, description, metadata)
    
    def add_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """Add PDF pages as images."""
        return self.store.add_pdf(pdf_path, pages, metadata)
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        use_reranker: bool = True
    ):
        """Query with text.
        
        Args:
            query_text: Text query
            top_k: Number of results to retrieve
            use_reranker: Whether to use Qwen3-VL reranker
            
        Returns:
            RLMChatCompletion with retrieved context
        """
        if use_reranker and self.store.use_qwen_reranker:
            results = self.store.hybrid_search_with_rerank(
                query_text,
                top_k=top_k
            )
        else:
            results = self.store.search(query_text, top_k=top_k)
        
        # Format context
        context = self._format_context(results)
        
        # Query LLM
        result = self.rlm.completion(query_text, context)
        result.retrieved_context = context
        result.sources = results
        
        return result
    
    def query_image(
        self,
        image_path: str,
        query_text: Optional[str] = None,
        top_k: int = 5
    ):
        """Query with image.
        
        Args:
            image_path: Path to query image
            query_text: Optional text query
            top_k: Number of results
            
        Returns:
            RLMChatCompletion with retrieved context
        """
        results = self.store.search_multimodal(
            text=query_text,
            image=image_path,
            top_k=top_k
        )
        
        context = self._format_context(results)
        
        prompt = query_text or "Describe the relevant content found."
        result = self.rlm.completion(prompt, context)
        result.retrieved_context = context
        result.sources = results
        
        return result
    
    def query_multimodal(
        self,
        query_text: str,
        image_path: Optional[str] = None,
        top_k: int = 5
    ):
        """Query with both text and image.
        
        Args:
            query_text: Text query
            image_path: Optional image path
            top_k: Number of results
            
        Returns:
            RLMChatCompletion with retrieved context
        """
        results = self.store.search_multimodal(
            text=query_text,
            image=image_path,
            top_k=top_k
        )
        
        context = self._format_context(results)
        
        result = self.rlm.completion(query_text, context)
        result.retrieved_context = context
        result.sources = results
        
        return result
    
    def _format_context(self, results: List[SearchResult]) -> str:
        """Format search results into context string."""
        if not results:
            return "No relevant documents found."
        
        parts = []
        for i, result in enumerate(results, 1):
            metadata = result.metadata
            doc_type = metadata.get("type", "text")
            source = metadata.get("source", metadata.get("pdf_path", "Unknown"))
            
            if doc_type == "pdf_page":
                page = metadata.get("page_num", "N/A")
                header = f"[{i}] PDF Page {page} from {source}"
            elif doc_type == "image":
                header = f"[{i}] Image: {source}"
            elif doc_type == "video":
                header = f"[{i}] Video: {source}"
            else:
                header = f"[{i}] Document: {source}"
            
            content = result.content
            if result.composite_score:
                header += f" [Score: {result.composite_score:.1f}%]"
            
            parts.append(f"{header}\n{content}")
        
        return "\n\n---\n\n".join(parts)


# Import for convenience
from vl_rag_graph_rlm.rag.store import create_vector_store

__all__ = [
    "RAGContextProvider",
    "RAGEnhancedVLRAGGraphRLM",
    "RAGConfig",
    "create_vector_store"
]

if HAS_MULTIMODAL:
    __all__.extend([
        "MultimodalVLRAGGraphRLM",
        "create_multimodal_store"
    ])
