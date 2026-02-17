"""Parallel recursive exploration for RLM.

Provides branching factor exploration (2-3 branches) for recursive
language model calls to explore multiple reasoning paths simultaneously.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("rlm.parallel_explore")


@dataclass
class ExplorationBranch:
    """A single exploration branch result."""
    branch_id: int
    query: str
    context: str
    response: str
    quality_score: float
    depth: int
    execution_time: float


class ParallelRecursiveExplorer:
    """Explore multiple reasoning paths in parallel.
    
    Creates 2-3 branches for each recursive call, evaluates responses,
    and selects the best path or combines results.
    
    Example:
        >>> explorer = ParallelRecursiveExplorer(branching_factor=3)
        >>> results = await explorer.explore(
        ...     query="Solve this problem",
        ...     context="Problem context",
        ...     rlm_client=rlm
        ... )
        >>> best_result = explorer.select_best(results)
    """
    
    def __init__(
        self,
        branching_factor: int = 3,
        max_parallel: int = 5,
        quality_threshold: float = 0.7,
        combination_strategy: str = "best",  # "best", "vote", "merge"
    ):
        """Initialize parallel explorer.
        
        Args:
            branching_factor: Number of parallel branches (2-3 recommended)
            max_parallel: Maximum concurrent executions
            quality_threshold: Minimum quality to accept a branch
            combination_strategy: How to combine results
        """
        if not 2 <= branching_factor <= 5:
            raise ValueError("Branching factor must be 2-5")
        
        self.branching_factor = branching_factor
        self.max_parallel = max_parallel
        self.quality_threshold = quality_threshold
        self.combination_strategy = combination_strategy
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)
    
    async def explore(
        self,
        query: str,
        context: str,
        rlm_client: Any,
        depth: int = 0,
        branch_queries: Optional[List[str]] = None,
    ) -> List[ExplorationBranch]:
        """Execute parallel exploration branches.
        
        Args:
            query: Main query
            context: Context for the query
            rlm_client: RLM client with `completion()` method
            depth: Current recursion depth
            branch_queries: Optional custom queries for each branch
            
        Returns:
            List of exploration branch results
        """
        # Generate branch queries if not provided
        if branch_queries is None:
            branch_queries = self._generate_branch_queries(query, context)
        
        # Limit to branching_factor
        branch_queries = branch_queries[:self.branching_factor]
        
        # Execute branches in parallel
        tasks = []
        for i, bq in enumerate(branch_queries):
            task = self._execute_branch(i, bq, context, rlm_client, depth)
            tasks.append(task)
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and low-quality results
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Branch failed: {r}")
                continue
            if r.quality_score >= self.quality_threshold:
                valid_results.append(r)
        
        return valid_results
    
    async def _execute_branch(
        self,
        branch_id: int,
        query: str,
        context: str,
        rlm_client: Any,
        depth: int,
    ) -> ExplorationBranch:
        """Execute a single exploration branch."""
        import time
        
        start = time.time()
        
        try:
            # Run RLM completion
            if hasattr(rlm_client, 'acompletion'):
                response = await rlm_client.acompletion(query, context)
            else:
                # Run sync in executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self._executor,
                    lambda: rlm_client.completion(query, context)
                )
            
            exec_time = time.time() - start
            
            # Assess quality
            quality = self._assess_quality(response)
            
            return ExplorationBranch(
                branch_id=branch_id,
                query=query,
                context=context,
                response=response.response if hasattr(response, 'response') else str(response),
                quality_score=quality,
                depth=depth,
                execution_time=exec_time,
            )
            
        except Exception as e:
            logger.error(f"Branch {branch_id} execution failed: {e}")
            raise
    
    def _generate_branch_queries(self, query: str, context: str) -> List[str]:
        """Generate variant queries for different exploration branches."""
        # Create different angles on the same problem
        branches = [
            query,  # Original
            f"Approach from first principles: {query}",
            f"Consider edge cases and verify: {query}",
            f"Break down step by step: {query}",
            f"Alternative perspective: {query}",
        ]
        return branches[:self.branching_factor]
    
    def _assess_quality(self, response: Any) -> float:
        """Assess response quality (0.0 to 1.0)."""
        text = response.response if hasattr(response, 'response') else str(response)
        
        score = 0.5  # Base score
        
        # Length factor (optimal around 200-2000 chars)
        length = len(text)
        if 100 <= length <= 3000:
            score += 0.2
        
        # Structure indicators
        if any(marker in text for marker in ['1.', '2.', '3.', '- ', '###']):
            score += 0.15
        
        # Completeness indicators
        if any(marker in text.lower() for marker in ['conclusion', 'summary', 'therefore', 'thus']):
            score += 0.15
        
        return min(score, 1.0)
    
    def select_best(self, branches: List[ExplorationBranch]) -> Optional[ExplorationBranch]:
        """Select the highest quality branch."""
        if not branches:
            return None
        return max(branches, key=lambda b: b.quality_score)
    
    def vote_consensus(self, branches: List[ExplorationBranch]) -> str:
        """Extract consensus from multiple branches via voting."""
        if not branches:
            return "No valid responses from exploration."
        
        if len(branches) == 1:
            return branches[0].response
        
        # Simple approach: concatenate with quality weighting
        weighted_responses = []
        for b in branches:
            weight = int(b.quality_score * 10)
            weighted_responses.extend([b.response] * weight)
        
        # Return highest quality response
        best = self.select_best(branches)
        return best.response if best else branches[0].response
    
    def merge_responses(self, branches: List[ExplorationBranch]) -> str:
        """Merge multiple responses into a comprehensive answer."""
        if not branches:
            return "No valid responses from exploration."
        
        if len(branches) == 1:
            return branches[0].response
        
        # Sort by quality
        sorted_branches = sorted(branches, key=lambda b: b.quality_score, reverse=True)
        
        # Build merged response
        parts = ["## Comprehensive Analysis (Merged from Multiple Perspectives)\n"]
        
        for i, b in enumerate(sorted_branches[:3], 1):
            parts.append(f"\n### Perspective {i} (Quality: {b.quality_score:.2f})\n")
            parts.append(b.response[:1000])  # Truncate long responses
            parts.append("\n---\n")
        
        return "".join(parts)
    
    def get_combined_result(self, branches: List[ExplorationBranch]) -> str:
        """Get final result using configured combination strategy."""
        if self.combination_strategy == "best":
            best = self.select_best(branches)
            return best.response if best else "No valid responses."
        elif self.combination_strategy == "vote":
            return self.vote_consensus(branches)
        elif self.combination_strategy == "merge":
            return self.merge_responses(branches)
        else:
            raise ValueError(f"Unknown strategy: {self.combination_strategy}")


async def parallel_recursive_completion(
    query: str,
    context: str,
    rlm_client: Any,
    branching_factor: int = 3,
    combination_strategy: str = "best",
) -> str:
    """Convenience function for parallel recursive exploration.
    
    Args:
        query: User query
        context: Context for the query
        rlm_client: RLM client instance
        branching_factor: Number of parallel branches (2-3 recommended)
        combination_strategy: How to combine results ("best", "vote", "merge")
        
    Returns:
        Final combined response
    """
    explorer = ParallelRecursiveExplorer(
        branching_factor=branching_factor,
        combination_strategy=combination_strategy,
    )
    
    branches = await explorer.explore(query, context, rlm_client)
    return explorer.get_combined_result(branches)
