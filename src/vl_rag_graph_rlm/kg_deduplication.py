"""Entity deduplication and coreference resolution for knowledge graphs.

Provides utilities to:
1. Detect duplicate entities with similar names
2. Resolve coreferences (pronouns referring to named entities)
3. Merge duplicate entities and update relationships
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class EntityNode:
    """Represents an entity for deduplication."""
    name: str
    entity_type: str
    mentions: List[str]  # Alternative mentions of this entity


class EntityDeduplicator:
    """Deduplicate entities in knowledge graphs using fuzzy matching."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider entities as duplicates
        """
        self.similarity_threshold = similarity_threshold
        self.merged_map: Dict[str, str] = {}  # old_name -> canonical_name
    
    def normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        # Lowercase
        normalized = name.lower()
        # Remove common suffixes/prefixes
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        normalized = re.sub(r'\s+(inc\.?|llc|ltd\.?|corp\.?|corporation|company)$', '', normalized)
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        return normalized
    
    def calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two entity names."""
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Check if one contains the other
        if norm1 in norm2 or norm2 in norm1:
            # Partial containment - weight by length ratio
            longer = max(len(norm1), len(norm2))
            shorter = min(len(norm1), len(norm2))
            return 0.8 + (0.2 * shorter / longer)
        
        # Use sequence matcher for fuzzy similarity
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def find_duplicates(self, entities: List[EntityNode]) -> List[Tuple[str, str, float]]:
        """Find duplicate entity pairs.
        
        Args:
            entities: List of entity nodes
            
        Returns:
            List of (name1, name2, similarity) tuples for duplicates
        """
        duplicates = []
        seen = set()
        
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                # Skip if different types (usually distinct entities)
                if ent1.entity_type != ent2.entity_type:
                    continue
                
                # Check all mention combinations
                max_sim = 0.0
                for m1 in [ent1.name] + ent1.mentions:
                    for m2 in [ent2.name] + ent2.mentions:
                        sim = self.calculate_similarity(m1, m2)
                        max_sim = max(max_sim, sim)
                
                if max_sim >= self.similarity_threshold:
                    pair = tuple(sorted([ent1.name, ent2.name]))
                    if pair not in seen:
                        seen.add(pair)
                        duplicates.append((ent1.name, ent2.name, max_sim))
        
        return sorted(duplicates, key=lambda x: -x[2])  # Highest similarity first
    
    def merge_entities(self, entities: List[EntityNode], 
                       duplicates: List[Tuple[str, str, float]]) -> Dict[str, str]:
        """Create merge mapping for duplicate entities.
        
        Uses union-find style merging where the longest/most canonical name
        becomes the representative.
        
        Returns:
            Mapping from old name to canonical name
        """
        # Group duplicates into clusters
        clusters: Dict[str, Set[str]] = {}
        name_to_cluster: Dict[str, str] = {}
        
        for name1, name2, _ in duplicates:
            # Find existing clusters
            c1 = name_to_cluster.get(name1)
            c2 = name_to_cluster.get(name2)
            
            if c1 and c2:
                # Merge clusters
                if c1 != c2:
                    clusters[c1].update(clusters[c2])
                    for name in clusters[c2]:
                        name_to_cluster[name] = c1
                    del clusters[c2]
            elif c1:
                clusters[c1].add(name2)
                name_to_cluster[name2] = c1
            elif c2:
                clusters[c2].add(name1)
                name_to_cluster[name1] = c2
            else:
                # New cluster
                cluster_id = name1
                clusters[cluster_id] = {name1, name2}
                name_to_cluster[name1] = cluster_id
                name_to_cluster[name2] = cluster_id
        
        # Choose canonical name for each cluster (longest normalized name)
        for cluster_id, names in clusters.items():
            canonical = max(names, key=lambda n: len(self.normalize_name(n)))
            for name in names:
                if name != canonical:
                    self.merged_map[name] = canonical
        
        return self.merged_map


class CoreferenceResolver:
    """Simple coreference resolution for knowledge graphs.
    
    Resolves pronouns and abbreviated references to full entity names.
    """
    
    # Common pronouns and determiners to resolve
    PRONOUNS = {'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'their', 'its'}
    
    # Demonstratives that often need resolution
    DEMONSTRATIVES = {'this', 'that', 'these', 'those', 'such'}
    
    def __init__(self):
        self.entity_context: Dict[str, str] = {}  # pronoun -> last mentioned entity
        self.recent_entities: List[str] = []  # Recently mentioned entities (for resolution)
        self.max_context = 5  # Number of recent entities to track
    
    def update_context(self, entity_name: str) -> None:
        """Update context with newly mentioned entity."""
        self.recent_entities.insert(0, entity_name)
        self.recent_entities = self.recent_entities[:self.max_context]
    
    def resolve_coreference(self, text: str, available_entities: List[str]) -> str:
        """Resolve coreferences in text using available entity list.
        
        Args:
            text: Text to resolve coreferences in
            available_entities: List of available entity names for matching
            
        Returns:
            Text with coreferences resolved where possible
        """
        words = text.split()
        resolved = []
        
        for word in words:
            clean_word = word.lower().strip('.,;:!?()[]{}"\'') 
            
            # Check if it's a pronoun
            if clean_word in self.PRONOUNS:
                # Try to resolve to most recent matching entity by gender/number
                resolved_entity = self._resolve_pronoun(clean_word)
                if resolved_entity:
                    resolved.append(f"{word} [{resolved_entity}]")
                    continue
            
            # Check for partial matches with available entities
            for entity in available_entities:
                # If this word is a substring of an entity or vice versa
                if clean_word in entity.lower() or entity.lower() in clean_word:
                    if clean_word != entity.lower():
                        self.update_context(entity)
                    break
            
            resolved.append(word)
        
        return ' '.join(resolved)
    
    def _resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve pronoun to most likely entity from recent context."""
        if not self.recent_entities:
            return None
        
        # Simple heuristic: return most recent entity
        # (More sophisticated approach would use gender/number agreement)
        if pronoun in {'he', 'him', 'his'}:
            # Look for person entities
            for entity in self.recent_entities:
                # Assume entities with common person indicators
                if any(indicator in entity.lower() for indicator in ['mr', 'mrs', 'dr', 'prof']):
                    return entity
        
        elif pronoun in {'it', 'its'}:
            # Look for non-person entities
            for entity in self.recent_entities:
                if not any(indicator in entity.lower() for indicator in ['mr', 'mrs', 'person']):
                    return entity
        
        # Default: most recent entity
        return self.recent_entities[0] if self.recent_entities else None
    
    def resolve_relationships(self, relationships: List[Tuple[str, str, str]], 
                              entity_names: List[str]) -> List[Tuple[str, str, str]]:
        """Resolve coreferences in relationship triples.
        
        Args:
            relationships: List of (source, relation, target) tuples
            entity_names: List of available entity names
            
        Returns:
            Relationships with coreferences resolved
        """
        resolved = []
        name_set = set(entity_names)
        
        for source, relation, target in relationships:
            # Resolve source if it's a pronoun
            if source.lower() in self.PRONOUNS:
                resolved_source = self._resolve_pronoun(source.lower()) or source
            else:
                resolved_source = source
            
            # Resolve target if it's a pronoun
            if target.lower() in self.PRONOUNS:
                resolved_target = self._resolve_pronoun(target.lower()) or target
            else:
                resolved_target = target
            
            # Only keep relationships where both endpoints are known entities
            if resolved_source in name_set and resolved_target in name_set:
                resolved.append((resolved_source, relation, resolved_target))
                self.update_context(resolved_source)
                self.update_context(resolved_target)
        
        return resolved


def deduplicate_knowledge_graph(kg_text: str, 
                                 similarity_threshold: float = 0.85) -> Tuple[str, Dict[str, str]]:
    """Deduplicate entities in a knowledge graph.
    
    Args:
        kg_text: Knowledge graph markdown text
        similarity_threshold: Minimum similarity to consider duplicates
        
    Returns:
        (deduplicated_kg_text, merge_mapping)
    """
    from vl_rag_graph_rlm.kg_visualization import KnowledgeGraphParser
    
    parser = KnowledgeGraphParser(kg_text)
    
    # Convert to EntityNode format
    entities = [
        EntityNode(e.name, e.entity_type, [e.name])
        for e in parser.entities
    ]
    
    # Deduplicate
    dedup = EntityDeduplicator(similarity_threshold)
    duplicates = dedup.find_duplicates(entities)
    merge_map = dedup.merge_entities(entities, duplicates)
    
    if not merge_map:
        return kg_text, {}
    
    # Apply merges to text
    result_text = kg_text
    for old_name, canonical in merge_map.items():
        # Replace in entity section
        result_text = re.sub(
            rf'- \*\*{re.escape(old_name)}\*\*',
            f'- **{canonical}**',
            result_text
        )
        # Replace in relationship section
        result_text = re.sub(
            rf'\b{re.escape(old_name)}\b',
            canonical,
            result_text
        )
    
    return result_text, merge_map


def get_deduplication_report(kg_text: str, 
                             similarity_threshold: float = 0.85) -> Dict:
    """Get a report of what would be deduplicated without applying changes.
    
    Args:
        kg_text: Knowledge graph markdown text
        similarity_threshold: Minimum similarity to consider duplicates
        
    Returns:
        Dictionary with deduplication report
    """
    from vl_rag_graph_rlm.kg_visualization import KnowledgeGraphParser
    
    parser = KnowledgeGraphParser(kg_text)
    entities = [
        EntityNode(e.name, e.entity_type, [e.name])
        for e in parser.entities
    ]
    
    dedup = EntityDeduplicator(similarity_threshold)
    duplicates = dedup.find_duplicates(entities)
    merge_map = dedup.merge_entities(entities, duplicates)
    
    return {
        "total_entities": len(entities),
        "duplicates_found": len(duplicates),
        "entities_merged": len(merge_map),
        "merge_groups": [
            {"from": old, "to": new, "similarity": next(
                (sim for n1, n2, sim in duplicates 
                 if (n1 == old and n2 == new) or (n1 == new and n2 == old)), 1.0
            )}
            for old, new in merge_map.items()
        ],
        "duplicate_pairs": [
            {"entity1": n1, "entity2": n2, "similarity": round(sim, 3)}
            for n1, n2, sim in duplicates
        ]
    }
