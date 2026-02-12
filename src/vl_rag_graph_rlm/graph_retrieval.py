"""Graph-augmented retrieval for knowledge graph context expansion.

Traverses knowledge graph edges to expand context around retrieved entities,
enabling multi-hop reasoning and richer context for RAG queries.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

from vl_rag_graph_rlm.kg_visualization import KnowledgeGraphParser


@dataclass
class GraphContext:
    """Represents expanded context from graph traversal."""
    entity_name: str
    entity_type: str
    description: str
    related_entities: List[Dict[str, Any]]  # name, relation, context
    depth: int  # Hop distance from seed entity


class GraphAugmentedRetrieval:
    """Expand retrieval context by traversing knowledge graph edges.
    
    Given a set of seed entities (from initial retrieval), traverses the
    knowledge graph to find related entities within N hops, providing
    richer context for multi-hop reasoning.
    """
    
    def __init__(self, kg_text: str, max_hops: int = 2):
        """
        Args:
            kg_text: Knowledge graph markdown text
            max_hops: Maximum traversal depth (default: 2)
        """
        self.parser = KnowledgeGraphParser(kg_text)
        self.max_hops = max_hops
        
        # Build adjacency list for efficient traversal
        self.entity_map: Dict[str, Any] = {}
        self.adjacency: Dict[str, List[Tuple[str, str, str]]] = {}  # entity -> [(target, relation, context)]
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build adjacency list from parsed relationships."""
        # Map entity names to their definitions
        for entity in self.parser.entities:
            self.entity_map[entity.name] = entity
            self.adjacency[entity.name] = []
        
        # Add relationships as edges (bidirectional for traversal)
        for rel in self.parser.relationships:
            # Forward edge: source -> target
            if rel.source not in self.adjacency:
                self.adjacency[rel.source] = []
            self.adjacency[rel.source].append((rel.target, rel.relation, rel.context))
            
            # Backward edge: target -> source (for bidirectional traversal)
            if rel.target not in self.adjacency:
                self.adjacency[rel.target] = []
            # Inverse relation name
            inverse = f"inverse_{rel.relation}"
            self.adjacency[rel.target].append((rel.source, inverse, rel.context))
    
    def expand_context(
        self, 
        seed_entities: List[str],
        max_results: int = 20,
        exclude_self: bool = True
    ) -> List[GraphContext]:
        """Expand context around seed entities via graph traversal.
        
        Args:
            seed_entities: List of entity names to start traversal from
            max_results: Maximum number of expanded entities to return
            exclude_self: If True, exclude seed entities from results
            
        Returns:
            List of GraphContext with expanded entities and their context
        """
        visited: Set[str] = set()
        if exclude_self:
            visited.update(seed_entities)
        
        results: List[GraphContext] = []
        queue: deque[Tuple[str, int]] = deque([(name, 0) for name in seed_entities])
        
        while queue and len(results) < max_results:
            current_name, depth = queue.popleft()
            
            if depth >= self.max_hops:
                continue
            
            if current_name not in self.adjacency:
                continue
            
            # Explore neighbors
            for target_name, relation, context in self.adjacency[current_name]:
                if target_name in visited:
                    continue
                
                visited.add(target_name)
                
                # Get entity info if available
                entity_info = self.entity_map.get(target_name)
                entity_type = entity_info.entity_type if entity_info else "Unknown"
                description = entity_info.description if entity_info else ""
                
                # Find related entities for this node (next hop)
                related = []
                if target_name in self.adjacency and depth + 1 < self.max_hops:
                    for next_target, next_rel, next_ctx in self.adjacency[target_name][:3]:
                        related.append({
                            "name": next_target,
                            "relation": next_rel,
                            "context": next_ctx
                        })
                
                graph_ctx = GraphContext(
                    entity_name=target_name,
                    entity_type=entity_type,
                    description=description,
                    related_entities=related,
                    depth=depth + 1
                )
                results.append(graph_ctx)
                
                # Queue for further exploration
                queue.append((target_name, depth + 1))
        
        return results
    
    def find_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 3
    ) -> List[List[Tuple[str, str, str]]]:
        """Find all paths between two entities up to max_depth.
        
        Args:
            source: Starting entity name
            target: Target entity name
            max_depth: Maximum path length
            
        Returns:
            List of paths, each path is list of (entity, relation, next_entity)
        """
        paths: List[List[Tuple[str, str, str]]] = []
        visited: Set[str] = set()
        
        def dfs(current: str, target: str, path: List[Tuple[str, str, str]], depth: int):
            if depth > max_depth:
                return
            if current == target and path:
                paths.append(path[:])
                return
            
            visited.add(current)
            
            if current in self.adjacency:
                for next_entity, relation, context in self.adjacency[current]:
                    if next_entity not in visited:
                        path.append((current, relation, next_entity))
                        dfs(next_entity, target, path, depth + 1)
                        path.pop()
            
            visited.remove(current)
        
        dfs(source, target, [], 0)
        return paths
    
    def get_entity_context_string(
        self,
        seed_entities: List[str],
        max_results: int = 20
    ) -> str:
        """Get expanded context as formatted string for RAG.
        
        Args:
            seed_entities: List of seed entity names
            max_results: Maximum expanded entities to include
            
        Returns:
            Formatted context string with graph-expanded information
        """
        expanded = self.expand_context(seed_entities, max_results)
        
        if not expanded:
            return ""
        
        lines = ["\n### Graph-Augmented Context", ""]
        
        # Group by hop depth
        by_depth: Dict[int, List[GraphContext]] = {}
        for ctx in expanded:
            by_depth.setdefault(ctx.depth, []).append(ctx)
        
        for depth in sorted(by_depth.keys()):
            lines.append(f"**Hop {depth} Connections:**")
            for ctx in by_depth[depth]:
                lines.append(f"- **{ctx.entity_name}** ({ctx.entity_type})")
                if ctx.description:
                    lines.append(f"  - Description: {ctx.description}")
                if ctx.related_entities:
                    rel_str = ", ".join([
                        f"{r['name']} ({r['relation']})" 
                        for r in ctx.related_entities[:3]
                    ])
                    lines.append(f"  - Related: {rel_str}")
            lines.append("")
        
        return "\n".join(lines)


def extract_entities_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    """Extract potential entity mentions from retrieved chunks.
    
    Simple heuristic: look for capitalized phrases and quoted terms.
    In production, would use NER (Named Entity Recognition).
    
    Args:
        chunks: Retrieved chunks with content
        
    Returns:
        List of potential entity names
    """
    import re
    
    entities = set()
    
    for chunk in chunks:
        content = chunk.get("content", "")
        
        # Capitalized phrases (potential named entities)
        # Pattern: Word Word Word (2-4 capitalized words)
        pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3}\b'
        for match in re.finditer(pattern, content):
            entity = match.group(0)
            # Filter out common false positives
            if entity.lower() not in {'the', 'a', 'an', 'this', 'that'}:
                entities.add(entity)
        
        # Quoted terms
        for match in re.finditer(r'"([^"]{2,30})"', content):
            entities.add(match.group(1))
    
    return list(entities)


def augment_retrieval_with_graph(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    kg_text: str,
    max_hops: int = 2,
    max_expanded: int = 15
) -> str:
    """Augment retrieval context with knowledge graph traversal.
    
    This is the main entry point for graph-augmented retrieval.
    
    Args:
        query: Original query
        retrieved_chunks: Chunks from initial retrieval
        kg_text: Knowledge graph markdown
        max_hops: Maximum traversal depth
        max_expanded: Maximum expanded entities to include
        
    Returns:
        Augmented context string to append to retrieval context
    """
    if not kg_text or not kg_text.strip():
        return ""
    
    # Extract seed entities from retrieved chunks
    seed_entities = extract_entities_from_chunks(retrieved_chunks)
    
    if not seed_entities:
        return ""
    
    # Initialize graph retrieval
    gar = GraphAugmentedRetrieval(kg_text, max_hops=max_hops)
    
    # Get expanded context
    return gar.get_entity_context_string(seed_entities, max_expanded)
