"""Composite reranker with fuzzy matching, keyword coverage, and semantic scoring.

Based on Paddle-ERNIE-RAG RerankerV2 implementation.
"""

import re
import logging
from typing import List, Dict, Any, Tuple

try:
    from fuzzywuzzy import fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

logger = logging.getLogger("vl_rag_graph_rlm.reranker")


class CompositeReranker:
    """Composite reranker combining multiple scoring signals.
    
    Combines:
    - Fuzzy string matching
    - Keyword coverage
    - Semantic similarity
    - Position bonus
    - Length scoring
    - Proper noun matching
    
    Example:
        >>> reranker = CompositeReranker()
        >>> chunks = [{"content": "..."}, ...]
        >>> sorted_chunks, status = reranker.process("query", chunks)
    """
    
    def __init__(self):
        self.use_paddlenlp = False
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords (removing stop words)."""
        stop_words = {
            '的', '了', '是', '在', '和', '与', '或', '等', '中', '上', '下',
            '为', '有', '以', '及', '将', '对', '从', '到', '由', '被', '把',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'
        }
        
        words = set()
        for char in text:
            if '\u4e00' <= char <= '\u9fff' and char not in stop_words:
                words.add(char)
        
        english_words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        words.update([w for w in english_words if w not in stop_words])
        
        return words
    
    def _calculate_composite_score(self, query: str, chunk: Dict[str, Any]) -> float:
        """Calculate composite relevance score."""
        content = chunk.get('content', '')
        
        # Fuzzy match
        if HAS_FUZZY:
            fuzzy_score = fuzz.partial_ratio(query, content)
        else:
            # Simple fallback
            query_lower = query.lower()
            content_lower = content.lower()
            if query_lower in content_lower:
                fuzzy_score = 80 + 20 * (len(query) / len(content)) if content else 80
            else:
                fuzzy_score = 50
        
        # Keyword coverage
        query_keywords = self._extract_keywords(query)
        content_keywords = self._extract_keywords(content)
        
        if query_keywords:
            keyword_hits = len(query_keywords & content_keywords)
            keyword_coverage = (keyword_hits / len(query_keywords)) * 100
        else:
            keyword_coverage = 0
        
        # Semantic similarity from vector search
        milvus_distance = chunk.get('semantic_score')
        if milvus_distance is None:
            milvus_similarity = 80.0
        else:
            milvus_similarity = 100 / (1 + milvus_distance * 0.1)
        
        # Position bonus
        position_bonus = 0
        if 'milvus_rank' in chunk:
            rank = chunk['milvus_rank']
            position_bonus = max(0, 20 - rank)
        
        # Length scoring
        content_len = len(content)
        if 200 <= content_len <= 600:
            length_score = 100
        elif content_len < 200:
            length_score = 50 + (content_len / 200) * 50
        else:
            length_score = 100 - min(50, (content_len - 600) / 20)
        
        # Proper noun bonus
        en_pattern = r'\b[A-Z][a-z]+\b|[A-Z]{2,}'
        
        def extract_potential_nouns(text):
            res = set(re.findall(en_pattern, text))
            zh_words = [w for w in re.split(r'[^\u4e00-\u9fa5]', text) if len(w) >= 2]
            res.update(zh_words)
            return res
        
        query_proper_nouns = extract_potential_nouns(query)
        content_proper_nouns = extract_potential_nouns(content)
        proper_noun_bonus = 0
        if query_proper_nouns:
            hits = len(query_proper_nouns & content_proper_nouns)
            if hits > 0:
                proper_noun_bonus = 30
        
        # Composite score
        base_score = (
            fuzzy_score * 0.25 +
            keyword_coverage * 0.25 +
            milvus_similarity * 0.35 +
            length_score * 0.15
        )
        
        final_score = base_score + position_bonus + proper_noun_bonus
        
        # Store details
        chunk['score_details'] = {
            'final': final_score,
            'fuzzy': fuzzy_score,
            'kw': keyword_coverage,
            'vec': milvus_similarity
        }
        
        return final_score
    
    def process(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        fuzzy_threshold: int = 10
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Rerank chunks by composite score.
        
        Args:
            query: Search query
            chunks: List of chunk dicts with 'content' and metadata
            fuzzy_threshold: Minimum fuzzy score threshold
            
        Returns:
            Tuple of (sorted_chunks, status)
        """
        if not chunks:
            return [], "no_chunks"
        
        # Add rank position
        for rank, chunk in enumerate(chunks, 1):
            chunk['milvus_rank'] = rank
        
        # Calculate scores
        for chunk in chunks:
            score = self._calculate_composite_score(query, chunk)
            chunk['composite_score'] = score
        
        # Sort by score descending
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get('composite_score', 0),
            reverse=True
        )
        
        return sorted_chunks, "success"
