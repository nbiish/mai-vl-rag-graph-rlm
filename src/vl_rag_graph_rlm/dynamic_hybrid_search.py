"""Dynamic hybrid search with query-adaptive weighting.

Automatically adjusts dense vs keyword search weights based on query type,
improving retrieval accuracy for different query patterns.
"""

import re
import logging
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

logger = logging.getLogger("rlm.rag.hybrid_search")


@dataclass
class QueryType:
    """Classification of query type for weight tuning."""
    name: str
    dense_weight: float
    keyword_weight: float
    description: str


class QueryClassifier:
    """Classify queries to determine optimal hybrid search weights.
    
    Analyzes query patterns to decide when to favor:
    - Dense (semantic) search: Conceptual, exploratory queries
    - Keyword (BM25) search: Technical, specific, ID/code queries
    """
    
    # Query type patterns
    PATTERNS = {
        'technical_code': {
            'regex': [
                r'\b[a-z]+_[a-z_]+\b',  # snake_case identifiers
                r'\b[A-Z][a-z]+[A-Z]\w*\b',  # CamelCase
                r'\b0x[0-9a-fA-F]+\b',  # hex codes
                r'\b\d{4,}\b',  # long numbers (IDs, codes)
                r'[=<>!]+',  # operators
                r'function|class|method|api|endpoint',
            ],
            'dense_weight': 0.3,
            'keyword_weight': 0.7,
            'description': 'Technical/code query - prefer keyword matching',
        },
        'conceptual': {
            'regex': [
                r'\b(what|how|why|explain|describe|compare)\b',
                r'\b(concept|theory|approach|methodology)\b',
                r'\b(difference between|similarities|relationship)\b',
            ],
            'dense_weight': 0.8,
            'keyword_weight': 0.2,
            'description': 'Conceptual query - prefer semantic matching',
        },
        'list_enumeration': {
            'regex': [
                r'\b(list|enumerate|what are|name all)\b',
                r'\b(top \d+|main|key|important)\b',
            ],
            'dense_weight': 0.6,
            'keyword_weight': 0.4,
            'description': 'List/enumeration query - balanced approach',
        },
        'specific_fact': {
            'regex': [
                r'\b(when|where|who|which)\b',
                r'\b(date|year|location|name|version)\b',
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # dates
            ],
            'dense_weight': 0.5,
            'keyword_weight': 0.5,
            'description': 'Specific fact query - balanced approach',
        },
    }
    
    def classify(self, query: str) -> QueryType:
        """Classify query and return optimal weights.
        
        Args:
            query: User query string
            
        Returns:
            QueryType with recommended weights
        """
        query_lower = query.lower()
        scores = {}
        
        for type_name, config in self.PATTERNS.items():
            score = 0
            for pattern in config['regex']:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            scores[type_name] = score
        
        # Find best match
        if not scores or max(scores.values()) == 0:
            # Default: balanced
            return QueryType(
                name='default',
                dense_weight=0.7,
                keyword_weight=0.3,
                description='Default balanced weighting'
            )
        
        best_type = max(scores, key=scores.get)
        config = self.PATTERNS[best_type]
        
        return QueryType(
            name=best_type,
            dense_weight=config['dense_weight'],
            keyword_weight=config['keyword_weight'],
            description=config['description']
        )


class DynamicHybridSearcher:
    """Hybrid search with dynamic weight adjustment.
    
    Combines dense (embedding) and keyword (BM25) search with
    query-type-aware weighting for optimal results.
    
    Example:
        >>> searcher = DynamicHybridSearcher(vector_store, keyword_index)
        >>> results = searcher.search("What is the API rate limit?", top_k=10)
        >>> # Automatically uses higher keyword weight for technical query
    """
    
    def __init__(
        self,
        vector_store: Any,  # MultimodalVectorStore
        keyword_index: Any,  # BM25 or similar
        default_dense_weight: float = 0.7,
        default_keyword_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        """Initialize hybrid searcher.
        
        Args:
            vector_store: Dense vector store
            keyword_index: Keyword search index (BM25)
            default_dense_weight: Default weight for dense search
            default_keyword_weight: Default weight for keyword search
            rrf_k: RRF ranking constant
        """
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.default_weights = (default_dense_weight, default_keyword_weight)
        self.rrf_k = rrf_k
        self.classifier = QueryClassifier()
        
        # Stats for tuning
        self.query_history: List[Dict[str, Any]] = []
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_dynamic: bool = True,
        custom_weights: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[str, float]]:
        """Execute hybrid search with optional dynamic weighting.
        
        Args:
            query: Search query
            top_k: Number of results
            use_dynamic: If True, auto-adjust weights by query type
            custom_weights: Optional (dense_weight, keyword_weight) override
            
        Returns:
            List of (doc_id, score) tuples
        """
        # Determine weights
        if custom_weights:
            dense_w, keyword_w = custom_weights
            query_type = QueryType('custom', dense_w, keyword_w, 'User-specified')
        elif use_dynamic:
            query_type = self.classifier.classify(query)
            dense_w, keyword_w = query_type.dense_weight, query_type.keyword_weight
        else:
            dense_w, keyword_w = self.default_weights
            query_type = QueryType('default', dense_w, keyword_w, 'Default')
        
        logger.debug(f"Query '{query[:50]}...' classified as {query_type.name}: "
                    f"dense={dense_w:.2f}, keyword={keyword_w:.2f}")
        
        # Execute searches
        dense_results = self._dense_search(query, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Fuse with RRF
        fused = self._rrf_fuse(
            dense_results, keyword_results,
            dense_weight=dense_w,
            keyword_weight=keyword_w,
        )
        
        # Store history for analysis
        self.query_history.append({
            'query': query,
            'type': query_type.name,
            'weights': (dense_w, keyword_w),
            'dense_results': len(dense_results),
            'keyword_results': len(keyword_results),
        })
        
        return fused[:top_k]
    
    def _dense_search(self, query: str, top_k: int) -> Dict[str, float]:
        """Execute dense vector search."""
        results = self.vector_store.search(query, top_k=top_k)
        return {r.id: r.semantic_score for r in results}
    
    def _keyword_search(self, query: str, top_k: int) -> Dict[str, float]:
        """Execute keyword (BM25) search."""
        # Assuming keyword_index has a search method
        if hasattr(self.keyword_index, 'search'):
            return self.keyword_index.search(query, top_k=top_k)
        # Fallback: use vector store's keyword search if available
        elif hasattr(self.vector_store, '_keyword_search'):
            return self.vector_store._keyword_search(query, top_k=top_k)
        else:
            return {}
    
    def _rrf_fuse(
        self,
        dense_results: Dict[str, float],
        keyword_results: Dict[str, float],
        dense_weight: float,
        keyword_weight: float,
    ) -> List[Tuple[str, float]]:
        """Fuse results using Reciprocal Rank Fusion with weights.
        
        RRF: score = Σ 1/(k + rank) for each list containing the item
        """
        all_ids = set(dense_results.keys()) | set(keyword_results.keys())
        
        # Rank results (lower rank = better)
        dense_ranked = sorted(dense_results.items(), key=lambda x: x[1])
        keyword_ranked = sorted(keyword_results.items(), key=lambda x: x[1], reverse=True)
        
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_ranked)}
        keyword_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(keyword_ranked)}
        
        # Compute weighted RRF scores
        scores = {}
        for doc_id in all_ids:
            score = 0.0
            
            if doc_id in dense_ranks:
                score += dense_weight / (self.rrf_k + dense_ranks[doc_id])
            
            if doc_id in keyword_ranks:
                score += keyword_weight / (self.rrf_k + keyword_ranks[doc_id])
            
            scores[doc_id] = score
        
        # Sort by score descending
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics and weight distribution."""
        if not self.query_history:
            return {'total_queries': 0}
        
        type_counts = {}
        for entry in self.query_history:
            t = entry['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            'total_queries': len(self.query_history),
            'type_distribution': type_counts,
            'avg_dense_weight': sum(e['weights'][0] for e in self.query_history) / len(self.query_history),
            'avg_keyword_weight': sum(e['weights'][1] for e in self.query_history) / len(self.query_history),
        }


def get_optimal_rrf_weights(
    query: str,
    base_dense_weight: float = 4.0,
    base_keyword_weight: float = 1.0,
) -> Tuple[float, float]:
    """Get RRF weights optimized for query type.
    
    Returns weights for RRF fusion (not the similarity weights).
    These are used in the RRF formula: w / (k + rank)
    
    Args:
        query: User query
        base_dense_weight: Base weight for dense results
        base_keyword_weight: Base weight for keyword results
        
    Returns:
        (dense_rrf_weight, keyword_rrf_weight)
    """
    classifier = QueryClassifier()
    query_type = classifier.classify(query)
    
    # Adjust base weights by query type preference
    dense_w = base_dense_weight * (query_type.dense_weight / 0.7)  # Normalize to default
    keyword_w = base_keyword_weight * (query_type.keyword_weight / 0.3)
    
    return dense_w, keyword_w
