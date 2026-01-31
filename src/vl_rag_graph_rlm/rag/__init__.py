"""Hybrid search and reranking algorithms inspired by Paddle-ERNIE-RAG.

Implements:
- Reciprocal Rank Fusion (RRF) for dense + keyword search
- Multi-factor reranking (fuzzy matching, keyword coverage, semantic scoring)
- Provider-agnostic embedding support
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("rlm.rag")


@dataclass
class SearchResult:
    """A single search result with metadata."""
    id: Any
    content: str
    metadata: Dict[str, Any]
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    composite_score: float = 0.0


class ReciprocalRankFusion:
    """RRF algorithm for combining multiple search result lists.
    
    Based on Paddle-ERNIE-RAG's hybrid search implementation.
    Formula: score = weight * (1 / (k + rank))
    """
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(
        self,
        result_lists: List[List[SearchResult]],
        weights: Optional[List[float]] = None
    ) -> List[SearchResult]:
        """Fuse multiple ranked lists into a single ranked list.
        
        Args:
            result_lists: List of ranked result lists
            weights: Optional weights for each list (default: equal weights)
            
        Returns:
            Fused and reranked list
        """
        if not result_lists:
            return []
        
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        scores: Dict[Any, Tuple[SearchResult, float]] = {}
        
        for results, weight in zip(result_lists, weights):
            for rank, result in enumerate(results):
                doc_id = result.id
                rrf_score = weight * (1.0 / (self.k + rank))
                
                if doc_id in scores:
                    existing_result, existing_score = scores[doc_id]
                    scores[doc_id] = (existing_result, existing_score + rrf_score)
                else:
                    scores[doc_id] = (result, rrf_score)
        
        # Sort by combined score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Update composite scores
        final_results = []
        for result, score in sorted_results:
            result.composite_score = score
            final_results.append(result)
        
        return final_results


class MultiFactorReranker:
    """Multi-factor reranking combining multiple signals.
    
    Based on Paddle-ERNIE-RAG's reranker_v2 implementation.
    Combines:
    - Fuzzy string matching
    - Keyword coverage
    - Semantic similarity
    - Position bias
    - Length normalization
    - Proper noun matching
    """
    
    def __init__(self):
        self.stop_words = {
            '的', '了', '是', '在', '和', '与', '或', '等', '中', '上', '下', 
            '为', '有', '以', '及', '将', '对', '从', '到', '由', '被', '把',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be'
        }
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        fuzzy_weight: float = 0.25,
        keyword_weight: float = 0.25,
        semantic_weight: float = 0.35,
        length_weight: float = 0.15
    ) -> List[SearchResult]:
        """Rerank results using multiple factors.
        
        Args:
            query: Original query
            results: Search results to rerank
            fuzzy_weight: Weight for fuzzy matching
            keyword_weight: Weight for keyword coverage
            semantic_weight: Weight for semantic similarity
            length_weight: Weight for length normalization
            
        Returns:
            Reranked results
        """
        try:
            from fuzzywuzzy import fuzz
        except ImportError:
            logger.warning("fuzzywuzzy not installed, using basic string similarity")
            fuzz = None
        
        query_keywords = self._extract_keywords(query)
        query_proper_nouns = self._extract_proper_nouns(query)
        
        for rank, result in enumerate(results, 1):
            content = result.content
            
            # 1. Fuzzy matching
            if fuzz:
                fuzzy_score = fuzz.partial_ratio(query.lower(), content.lower())
            else:
                fuzzy_score = self._basic_similarity(query, content)
            
            # 2. Keyword coverage
            content_keywords = self._extract_keywords(content)
            if query_keywords:
                keyword_hits = len(query_keywords & content_keywords)
                keyword_coverage = (keyword_hits / len(query_keywords)) * 100
            else:
                keyword_coverage = 0
            
            # 3. Semantic similarity (convert distance to similarity)
            if result.semantic_score > 0:
                # Assuming L2 distance: convert to similarity score
                semantic_similarity = 100 / (1 + result.semantic_score * 0.1)
            else:
                semantic_similarity = 80.0  # Default
            
            # 4. Position bonus (early results get boost)
            position_bonus = max(0, 20 - rank)
            
            # 5. Length normalization
            content_len = len(content)
            if 200 <= content_len <= 600:
                length_score = 100
            elif content_len < 200:
                length_score = 50 + (content_len / 200) * 50
            else:
                length_score = 100 - min(50, (content_len - 600) / 20)
            
            # 6. Proper noun bonus
            content_proper_nouns = self._extract_proper_nouns(content)
            proper_noun_bonus = 0
            if query_proper_nouns:
                hits = len(query_proper_nouns & content_proper_nouns)
                if hits > 0:
                    proper_noun_bonus = 30
            
            # Calculate composite score
            base_score = (
                fuzzy_score * fuzzy_weight +
                keyword_coverage * keyword_weight +
                semantic_similarity * semantic_weight +
                length_score * length_weight
            )
            
            result.composite_score = base_score + position_bonus + proper_noun_bonus
        
        # Sort by composite score
        return sorted(results, key=lambda x: x.composite_score, reverse=True)
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text, removing stop words."""
        words = set()
        
        # Chinese characters
        for char in text:
            if '\u4e00' <= char <= '\u9fff' and char not in self.stop_words:
                words.add(char)
        
        # English words
        english_words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        words.update([w for w in english_words if w not in self.stop_words])
        
        return words
    
    def _extract_proper_nouns(self, text: str) -> set:
        """Extract potential proper nouns/entities."""
        # English proper nouns: capitalized words
        en_pattern = r'\b[A-Z][a-z]+\b|[A-Z]{2,}'
        nouns = set(re.findall(en_pattern, text))
        
        # Chinese: sequences of 2+ characters
        zh_words = [w for w in re.split(r'[^\u4e00-\u9fff]', text) if len(w) >= 2]
        nouns.update(zh_words)
        
        return nouns
    
    def _basic_similarity(self, s1: str, s2: str) -> float:
        """Basic string similarity without fuzzywuzzy."""
        s1_lower = s1.lower()
        s2_lower = s2.lower()
        
        # Simple substring matching
        if s1_lower in s2_lower or s2_lower in s1_lower:
            return 80.0
        
        # Token overlap
        tokens1 = set(s1_lower.split())
        tokens2 = set(s2_lower.split())
        
        if not tokens1:
            return 0.0
        
        overlap = len(tokens1 & tokens2)
        return (overlap / len(tokens1)) * 100


class HybridSearcher:
    """Hybrid search combining dense vector and keyword search.
    
    Inspired by Paddle-ERNIE-RAG's vector_store implementation.
    """
    
    def __init__(
        self,
        rrf_k: int = 60,
        dense_weight: float = 4.0,
        keyword_weight: float = 1.0
    ):
        self.rrf = ReciprocalRankFusion(k=rrf_k)
        self.reranker = MultiFactorReranker()
        self.dense_weight = dense_weight
        self.keyword_weight = keyword_weight
    
    def search(
        self,
        query: str,
        dense_results: List[SearchResult],
        keyword_results: List[SearchResult],
        top_k: int = 10,
        rerank: bool = True
    ) -> List[SearchResult]:
        """Execute hybrid search with RRF fusion and optional reranking.
        
        Args:
            query: Search query
            dense_results: Results from dense vector search
            keyword_results: Results from keyword/BM25 search
            top_k: Number of results to return
            rerank: Whether to apply multi-factor reranking
            
        Returns:
            Final ranked results
        """
        # RRF Fusion
        fused = self.rrf.fuse(
            [dense_results, keyword_results],
            weights=[self.dense_weight, self.keyword_weight]
        )
        
        # Multi-factor reranking
        if rerank:
            fused = self.reranker.rerank(query, fused)
        
        return fused[:top_k]


# Convenience function
def hybrid_search(
    query: str,
    dense_results: List[SearchResult],
    keyword_results: List[SearchResult],
    top_k: int = 10
) -> List[SearchResult]:
    """Simple hybrid search with sensible defaults."""
    searcher = HybridSearcher()
    return searcher.search(query, dense_results, keyword_results, top_k)


# Import Qwen3-VL providers (optional - only available if dependencies installed)
try:
    from vl_rag_graph_rlm.rag.qwen3vl import (
        Qwen3VLEmbeddingProvider,
        Qwen3VLRerankerProvider,
        create_qwen3vl_embedder,
        create_qwen3vl_reranker,
        MultimodalDocument
    )
    __all__ = [
        "SearchResult",
        "ReciprocalRankFusion",
        "MultiFactorReranker",
        "HybridSearcher",
        "hybrid_search",
        "Qwen3VLEmbeddingProvider",
        "Qwen3VLRerankerProvider",
        "create_qwen3vl_embedder",
        "create_qwen3vl_reranker",
        "MultimodalDocument"
    ]
except ImportError:
    __all__ = [
        "SearchResult",
        "ReciprocalRankFusion",
        "MultiFactorReranker",
        "HybridSearcher",
        "hybrid_search"
    ]
