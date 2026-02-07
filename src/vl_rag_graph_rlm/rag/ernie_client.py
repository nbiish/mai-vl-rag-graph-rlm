"""ERNIE Client for Baidu AI Studio API.

Provides OpenAI-compatible interface for ERNIE models with rate limiting
and multimodal (vision) support.
"""

import os
import base64
import time
import random
import logging
from typing import Optional, List, Dict, Any
from openai import OpenAI

logger = logging.getLogger("vl_rag_graph_rlm.ernie")


class ERNIEClient:
    """Baidu ERNIE / OpenAI-compatible client with rate limiting.
    
    Optimized for Qianfan native API 429 rate limiting.
    
    Example:
        >>> client = ERNIEClient(
        ...     llm_api_key="your-aistudio-token",
        ...     llm_model="ernie-5.0-thinking-preview"
        ... )
        >>> response = client.chat([{"role": "user", "content": "Hello"}])
    """
    
    def __init__(
        self,
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        embed_api_base: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        embed_model: Optional[str] = None,
        qps: float = 0.8
    ):
        # LLM configuration
        self.llm_base = (llm_api_base or "https://aistudio.baidu.com/llm/lmapi/v3").rstrip('/')
        self.llm_key = llm_api_key or os.getenv("AISTUDIO_ACCESS_TOKEN", "")
        self.chat_model_name = llm_model or "ernie-5.0-thinking-preview"
        
        # Embedding configuration
        self.embed_base = (embed_api_base or "https://aistudio.baidu.com/llm/lmapi/v3").rstrip('/')
        self.embed_key = embed_api_key or os.getenv("AISTUDIO_ACCESS_TOKEN", "")
        self.embedding_model_name = embed_model or "embedding-v1"
        
        # Rate control
        self.target_qps = float(qps) if qps > 0 else 0.8
        self.current_delay = 1.0 / self.target_qps
        
        self.last_embed_time = 0
        self.last_chat_time = 0
        self.max_retries = 5
        
        # Initialize clients
        self.chat_client: Optional[OpenAI] = None
        self.embed_client: Optional[OpenAI] = None
        self._init_clients()
    
    def _init_clients(self):
        """Initialize OpenAI clients."""
        if self.llm_key:
            try:
                self.chat_client = OpenAI(
                    base_url=self.llm_base,
                    api_key=self.llm_key,
                    max_retries=self.max_retries,
                    timeout=120.0
                )
            except Exception as e:
                logger.error(f"LLM Client init error: {e}")
        
        if self.embed_key:
            try:
                self.embed_client = OpenAI(
                    base_url=self.embed_base,
                    api_key=self.embed_key,
                    max_retries=self.max_retries,
                    timeout=120.0
                )
            except Exception as e:
                logger.error(f"Embedding Client init error: {e}")
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """Read and encode image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Image encode error: {e}")
            return None
    
    def chat_with_image(self, query: str, image_path: str) -> str:
        """Send vision-capable chat request.
        
        Args:
            query: Text query
            image_path: Path to image file
            
        Returns:
            Model response text
        """
        base64_image = self._encode_image(image_path)
        
        if not base64_image:
            logger.warning("Image encode failed, falling back to text")
            return self.chat([{"role": "user", "content": query}])
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        return self.chat(messages)
    
    def _wait_for_rate_limit(self, is_embedding: bool = True):
        """Rate limit wait."""
        now = time.time()
        last_time = self.last_embed_time if is_embedding else self.last_chat_time
        elapsed = now - last_time
        if elapsed < self.current_delay:
            time.sleep(self.current_delay - elapsed)
        
        if is_embedding:
            self.last_embed_time = time.time()
        else:
            self.last_chat_time = time.time()
    
    def _adaptive_slow_down(self):
        """Adaptive slowdown on rate limit."""
        self.current_delay = min(self.current_delay * 2.0, 15.0)
        logger.warning(f"Rate limit hit, slowing down: new delay {self.current_delay:.2f}s")
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """Send chat completion request.
        
        Args:
            messages: List of message dicts with role and content
            model: Override model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        use_model = model or self.chat_model_name
        self._wait_for_rate_limit(is_embedding=False)
        
        if not self.chat_client:
            return "Error: Client not initialized"
        
        try:
            response = self.chat_client.chat.completions.create(
                model=use_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            self.last_chat_time = time.time()
            content = response.choices[0].message.content
            return content or "Model returned empty content"
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise
    
    def get_embedding(self, text: str, max_retries: int = 5) -> Optional[List[float]]:
        """Get embedding for text.
        
        Args:
            text: Text to embed
            max_retries: Maximum retry attempts
            
        Returns:
            Embedding vector or None on failure
        """
        if not text:
            return None
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit(is_embedding=True)
                
                if self.embed_client:
                    response = self.embed_client.embeddings.create(
                        model=self.embedding_model_name,
                        input=[text]
                    )
                    self.last_embed_time = time.time()
                    if response and response.data:
                        return response.data[0].embedding
                    
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "429" in error_str or
                    "rate limit" in error_str or
                    "rpm_rate_limit_exceeded" in error_str or
                    "tpm_rate_limit_exceeded" in error_str
                )
                
                if is_rate_limit:
                    self._adaptive_slow_down()
                    wait_time = (2 ** attempt) + random.uniform(1.0, 3.0)
                    logger.warning(f"Rate limit, backing off {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Embedding error (attempt {attempt + 1}): {e}")
                    time.sleep(1)
        
        logger.error("Embedding final failure")
        return None
    
    def get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Batch get embeddings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        return [self.get_embedding(t) for t in texts]
    
    def answer_question(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Answer question based on context chunks.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks with content, filename, page
            
        Returns:
            Generated answer
        """
        if not context_chunks:
            prompt = f"User question: {question}"
        else:
            context_str = ""
            for i, chunk in enumerate(context_chunks):
                content = chunk.get('content', '').replace('\n', ' ')[:800]
                fname = chunk.get('filename', 'Unknown')
                page = chunk.get('page', 0)
                context_str += f"[Ref {i+1} ({fname} P{page})]: {content}\n\n"
            prompt = f"Based on the following references answer the question:\n\n[References]:\n{context_str}\n\n[Question]:\n{question}"
        
        result = self.chat([{"role": "user", "content": prompt}])
        return result or "Answer generation failed"
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate summary of text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in characters
            
        Returns:
            Generated summary
        """
        if not text:
            return "No content"
        prompt = f"Generate a concise summary (within {max_length} chars) of:\n\n{text[:5000]}"
        result = self.chat([{"role": "user", "content": prompt}])
        return result or "Summary generation failed"
    
    def rewrite_query(self, query: str) -> str:
        """Rewrite query for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query with more context
        """
        prompt = f"""Rewrite the following search query into a more detailed statement with additional context keywords for better vector retrieval.

Original: "{query}"
Rewritten:"""
        result = self.chat([{"role": "user", "content": prompt}], max_tokens=200)
        return result if result else query
