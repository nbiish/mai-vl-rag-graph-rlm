"""Vision and multimodal support for RLM.

Extends RLM with image understanding capabilities using existing providers
that support multimodal inputs (OpenAI, OpenRouter, Gemini, etc.).

Inspired by Paddle-ERNIE-RAG's vision capabilities.
"""

import base64
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass

from vl_rag_graph_rlm.clients import get_client, BaseLM
from vl_rag_graph_rlm.types import RLMChatCompletion, UsageSummary

logger = logging.getLogger("rlm.vision")


@dataclass
class ImageContent:
    """Represents an image for multimodal input."""
    source: Union[str, bytes, Path]  # URL, base64, bytes, or file path
    mime_type: Optional[str] = None
    
    SUPPORTED_FORMATS = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible image format."""
        if isinstance(self.source, str) and self.source.startswith(("http://", "https://")):
            return {
                "type": "image_url",
                "image_url": {"url": self.source}
            }
        else:
            # File path or bytes - convert to base64
            base64_data = self._to_base64()
            mime = self.mime_type or "image/jpeg"
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{base64_data}"
                }
            }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic-compatible image format."""
        base64_data = self._to_base64()
        mime = self.mime_type or "image/jpeg"
        
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": base64_data
            }
        }
    
    def to_gemini_format(self):
        """Convert to Gemini format."""
        try:
            from google.genai import types

            if isinstance(self.source, str) and self.source.startswith(("http://", "https://")):
                return types.Part.from_uri(file_uri=self.source, mime_type=self.mime_type or "image/jpeg")
            else:
                base64_data = self._to_base64()
                return types.Part.from_bytes(
                    data=base64.b64decode(base64_data),
                    mime_type=self.mime_type or "image/jpeg",
                )
        except ImportError:
            logger.error("google-genai not installed")
            return None
    
    def _to_base64(self) -> str:
        """Convert image to base64 string."""
        if isinstance(self.source, str) and not self.source.startswith(("http://", "https://")):
            # File path
            with open(self.source, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(self.source, (str, bytes)):
            # Already base64 or bytes
            if isinstance(self.source, bytes):
                return base64.b64encode(self.source).decode("utf-8")
            return self.source
        elif isinstance(self.source, Path):
            with open(self.source, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image source type: {type(self.source)}")


class MultimodalClient(BaseLM):
    """Wrapper for multimodal-capable clients.
    
    Extends any provider with image understanding capabilities.
    
    Example:
        >>> from vl_rag_graph_rlm.vision import MultimodalClient, ImageContent
        >>>
        >>> client = MultimodalClient(provider="openai", model="gpt-4o")
        >>>
        >>> image = ImageContent("path/to/chart.png")
        >>> result = client.analyze_image(
        ...     image=image,
        ...     prompt="What trends do you see in this chart?"
        ... )
        >>> print(result)
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ):
        self.provider = provider.lower()
        self.model = model
        
        client_kwargs = {"model_name": model, **kwargs}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["api_base"] = api_base
            
        self.client = get_client(provider, **client_kwargs)
        self._usage_history: List[UsageSummary] = []
    
    def analyze_image(
        self,
        image: Union[ImageContent, str, Path],
        prompt: str = "Describe this image.",
        **kwargs
    ) -> str:
        """Analyze a single image.
        
        Args:
            image: Image to analyze (ImageContent, file path, or URL)
            prompt: Question or instruction about the image
            **kwargs: Additional completion parameters
            
        Returns:
            Model's analysis of the image
        """
        if not isinstance(image, ImageContent):
            image = ImageContent(image)
        
        messages = self._build_multimodal_messages([image], prompt)
        return self.client.completion(messages, **kwargs)
    
    def analyze_images(
        self,
        images: List[Union[ImageContent, str, Path]],
        prompt: str = "Describe these images.",
        **kwargs
    ) -> str:
        """Analyze multiple images.
        
        Args:
            images: List of images to analyze
            prompt: Question or instruction about the images
            **kwargs: Additional completion parameters
            
        Returns:
            Model's analysis of the images
        """
        image_contents = [
            img if isinstance(img, ImageContent) else ImageContent(img)
            for img in images
        ]
        
        messages = self._build_multimodal_messages(image_contents, prompt)
        return self.client.completion(messages, **kwargs)
    
    def analyze_document_with_images(
        self,
        text: str,
        images: List[Union[ImageContent, str, Path]],
        prompt: str = "Analyze this document and its images.",
        **kwargs
    ) -> str:
        """Analyze text with associated images.
        
        Inspired by Paddle-ERNIE-RAG's multimodal Q&A.
        
        Args:
            text: Document text content
            images: Associated images (figures, charts, etc.)
            prompt: Question or instruction
            **kwargs: Additional completion parameters
            
        Returns:
            Combined analysis of text and images
        """
        image_contents = [
            img if isinstance(img, ImageContent) else ImageContent(img)
            for img in images
        ]
        
        # Build combined message with text context
        full_prompt = f"""{prompt}

Document Context:
{text[:2000]}

Analyze the above document along with the following images."""

        messages = self._build_multimodal_messages(image_contents, full_prompt)
        return self.client.completion(messages, **kwargs)
    
    def _build_multimodal_messages(
        self,
        images: List[ImageContent],
        prompt: str
    ) -> List[Dict[str, Any]]:
        """Build messages in provider-specific format."""
        if self.provider in ["openai", "openrouter", "zenmux", "zai"]:
            return self._build_openai_format(images, prompt)
        elif self.provider == "anthropic":
            return self._build_anthropic_format(images, prompt)
        elif self.provider == "gemini":
            return self._build_gemini_format(images, prompt)
        else:
            # Default to OpenAI format (works with LiteLLM too)
            return self._build_openai_format(images, prompt)
    
    def _build_openai_format(
        self,
        images: List[ImageContent],
        prompt: str
    ) -> List[Dict[str, Any]]:
        """Build OpenAI-compatible multimodal messages."""
        content = [{"type": "text", "text": prompt}]
        
        for image in images:
            content.append(image.to_openai_format())
        
        return [{"role": "user", "content": content}]
    
    def _build_anthropic_format(
        self,
        images: List[ImageContent],
        prompt: str
    ) -> List[Dict[str, Any]]:
        """Build Anthropic-compatible multimodal messages."""
        content = []
        
        for image in images:
            content.append(image.to_anthropic_format())
        
        content.append({"type": "text", "text": prompt})
        
        return [{"role": "user", "content": content}]
    
    def _build_gemini_format(
        self,
        images: List[ImageContent],
        prompt: str
    ):
        """Build Gemini-compatible multimodal messages."""
        parts = [prompt]
        
        for image in images:
            gemini_image = image.to_gemini_format()
            if gemini_image:
                parts.append(gemini_image)
        
        return parts
    
    def completion(self, prompt: str, **kwargs) -> str:
        """Standard text completion (non-multimodal)."""
        return self.client.completion(prompt, **kwargs)
    
    async def acompletion(self, prompt: str, **kwargs) -> str:
        """Async text completion."""
        return await self.client.acompletion(prompt, **kwargs)
    
    def get_usage_summary(self) -> UsageSummary:
        """Get aggregated usage."""
        return self.client.get_usage_summary()
    
    def get_last_usage(self):
        """Get last call usage."""
        return self.client.get_last_usage()


class VisionRAG:
    """RAG with vision support for analyzing documents with images.
    
    Combines RAG text retrieval with image analysis for comprehensive
    document understanding.
    
    Example:
        >>> from vl_rag_graph_rlm.vision import VisionRAG
        >>>
        >>> vr = VisionRAG(
        ...     llm_provider="openai",
        ...     llm_model="gpt-4o",
        ...     embedding_provider="openai"
        ... )
        >>>
        >>> # Add document with images
        >>> vr.add_document(
        ...     text="Document text content...",
        ...     images=["figure1.png", "figure2.png"],
        ...     metadata={"source": "paper.pdf", "page": 1}
        ... )
        >>>
        >>> # Query that may use both text and images
        >>> result = vr.query("Explain the trends in Figure 1")
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        llm_api_key: Optional[str] = None,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: Optional[str] = None,
        storage_path: Optional[str] = None
    ):
        from vl_rag_graph_rlm.rag import RAGContextProvider, create_vector_store, RAGConfig
        
        # Initialize multimodal client
        self.vision = MultimodalClient(
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
        
        # Initialize RAG
        self.rag = RAGContextProvider(self.store, RAGConfig(top_k=5))
        
        # Track document images
        self.doc_images: Dict[str, List[str]] = {}
    
    def add_document(
        self,
        text: str,
        images: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a document with optional images."""
        doc_id = self.store.add(text, metadata)
        
        if images:
            self.doc_images[doc_id] = images
        
        return doc_id
    
    def query(
        self,
        query: str,
        use_vision: bool = True,
        max_images: int = 3
    ) -> str:
        """Query with optional vision analysis.
        
        1. Retrieves relevant text chunks
        2. If use_vision=True, retrieves associated images
        3. Combines for comprehensive answer
        """
        # Get text context
        context_results = self.rag._hybrid_search(query)
        
        if not context_results:
            return "No relevant documents found."
        
        # Get associated images
        images_to_analyze = []
        if use_vision:
            for result in context_results[:max_images]:
                doc_id = result.id
                if doc_id in self.doc_images:
                    images_to_analyze.extend(self.doc_images[doc_id])
        
        # Build context
        text_context = self.rag._format_context(context_results)
        
        if images_to_analyze and use_vision:
            # Multimodal analysis
            return self.vision.analyze_document_with_images(
                text=text_context,
                images=images_to_analyze[:max_images],
                prompt=query
            )
        else:
            # Text-only analysis
            return self.vision.completion(
                f"Based on the following context, answer the question.\n\nContext:\n{text_context}\n\nQuestion: {query}"
            )


# Convenience functions
def analyze_image(
    image: Union[str, Path],
    prompt: str = "Describe this image.",
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: Optional[str] = None
) -> str:
    """Quick image analysis."""
    client = MultimodalClient(provider=provider, model=model, api_key=api_key)
    return client.analyze_image(image, prompt)


__all__ = [
    "ImageContent",
    "MultimodalClient",
    "VisionRAG",
    "analyze_image"
]
