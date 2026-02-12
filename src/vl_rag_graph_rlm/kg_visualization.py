"""Knowledge graph serialization and visualization.

Provides NetworkX graph export and Mermaid/Graphviz visualization
for the extracted knowledge graphs.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    name: str
    entity_type: str
    description: str


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source: str
    relation: str
    target: str
    context: str


class KnowledgeGraphParser:
    """Parse knowledge graph markdown into structured format."""
    
    def __init__(self, kg_text: str):
        self.kg_text = kg_text
        self.entities: List[Entity] = []
        self.relationships: List[Relationship] = []
        self._parse()
    
    def _parse(self) -> None:
        """Parse the knowledge graph markdown text."""
        lines = self.kg_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if line.startswith('**Entities**') or line.startswith('## Entities'):
                current_section = 'entities'
                continue
            elif line.startswith('**Relationships**') or line.startswith('## Relationships'):
                current_section = 'relationships'
                continue
            
            # Parse entities
            if current_section == 'entities' and line.startswith('- '):
                entity = self._parse_entity_line(line)
                if entity:
                    self.entities.append(entity)
            
            # Parse relationships
            elif current_section == 'relationships' and line.startswith('- '):
                rel = self._parse_relationship_line(line)
                if rel:
                    self.relationships.append(rel)
    
    def _parse_entity_line(self, line: str) -> Optional[Entity]:
        """Parse an entity line like: - **Name** (Type): Description"""
        # Pattern: - **Name** (Type): Description
        match = re.match(r'- \*\*(.+?)\*\* \((.+?)\):?\s*(.*)', line)
        if match:
            return Entity(
                name=match.group(1).strip(),
                entity_type=match.group(2).strip(),
                description=match.group(3).strip()
            )
        
        # Alternative: - Name (Type): Description
        match = re.match(r'- (.+?) \((.+?)\):?\s*(.*)', line)
        if match:
            return Entity(
                name=match.group(1).strip(),
                entity_type=match.group(2).strip(),
                description=match.group(3).strip()
            )
        
        return None
    
    def _parse_relationship_line(self, line: str) -> Optional[Relationship]:
        """Parse a relationship line like: - EntityA → relation → EntityB (context)"""
        # Pattern: - Source → relation → Target (context)
        match = re.match(r'- (.+?) → (.+?) → (.+?) \((.*)\)', line)
        if match:
            return Relationship(
                source=match.group(1).strip(),
                relation=match.group(2).strip(),
                target=match.group(3).strip(),
                context=match.group(4).strip()
            )
        
        # Alternative without context
        match = re.match(r'- (.+?) → (.+?) → (.+?)$', line)
        if match:
            return Relationship(
                source=match.group(1).strip(),
                relation=match.group(2).strip(),
                target=match.group(3).strip(),
                context=""
            )
        
        return None


def export_to_networkx(kg_text: str) -> Optional[Any]:
    """Export knowledge graph to NetworkX graph.
    
    Args:
        kg_text: Knowledge graph markdown text
        
    Returns:
        NetworkX DiGraph or None if networkx not installed
    """
    try:
        import networkx as nx
    except ImportError:
        return None
    
    parser = KnowledgeGraphParser(kg_text)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add entities as nodes
    for entity in parser.entities:
        G.add_node(
            entity.name,
            entity_type=entity.entity_type,
            description=entity.description
        )
    
    # Add relationships as edges
    for rel in parser.relationships:
        # Ensure nodes exist
        if rel.source not in G:
            G.add_node(rel.source, entity_type="Unknown")
        if rel.target not in G:
            G.add_node(rel.target, entity_type="Unknown")
        
        G.add_edge(
            rel.source,
            rel.target,
            relation=rel.relation,
            context=rel.context
        )
    
    return G


def export_to_mermaid(kg_text: str) -> str:
    """Export knowledge graph to Mermaid diagram format.
    
    Args:
        kg_text: Knowledge graph markdown text
        
    Returns:
        Mermaid diagram text
    """
    parser = KnowledgeGraphParser(kg_text)
    
    lines = ["graph TD"]
    
    # Add entity nodes with styling by type
    type_colors = {
        "Person": "#e1f5fe",
        "Organisation": "#fff3e0",
        "Organization": "#fff3e0",
        "Concept": "#f3e5f5",
        "Technology": "#e8f5e9",
        "Location": "#ffebee",
        "Event": "#fffde7",
        "Metric": "#fce4ec",
    }
    
    # Track added nodes to avoid duplicates
    added_nodes = set()
    
    for entity in parser.entities:
        node_id = _sanitize_node_id(entity.name)
        if node_id not in added_nodes:
            color = type_colors.get(entity.entity_type, "#f5f5f5")
            lines.append(f'    {node_id}["{entity.name}"]')
            lines.append(f'    style {node_id} fill:{color}')
            added_nodes.add(node_id)
    
    # Add relationships as edges
    for rel in parser.relationships:
        source_id = _sanitize_node_id(rel.source)
        target_id = _sanitize_node_id(rel.target)
        
        # Ensure nodes exist
        if source_id not in added_nodes:
            lines.append(f'    {source_id}["{rel.source}"]')
            added_nodes.add(source_id)
        if target_id not in added_nodes:
            lines.append(f'    {target_id}["{rel.target}"]')
            added_nodes.add(target_id)
        
        # Edge with label
        safe_relation = rel.relation.replace('"', '&quot;')
        lines.append(f'    {source_id} -->|"{safe_relation}"| {target_id}')
    
    return '\n'.join(lines)


def export_to_graphviz(kg_text: str) -> str:
    """Export knowledge graph to Graphviz DOT format.
    
    Args:
        kg_text: Knowledge graph markdown text
        
    Returns:
        Graphviz DOT format text
    """
    parser = KnowledgeGraphParser(kg_text)
    
    lines = ['digraph KnowledgeGraph {', '    rankdir=LR;']
    
    # Node styling by type
    type_colors = {
        "Person": "lightblue1",
        "Organisation": "lightgoldenrod1",
        "Organization": "lightgoldenrod1",
        "Concept": "plum1",
        "Technology": "palegreen1",
        "Location": "lightpink1",
        "Event": "lightyellow1",
        "Metric": "lightcoral",
    }
    
    # Track added nodes
    added_nodes = set()
    
    # Add entity nodes
    for entity in parser.entities:
        node_id = _sanitize_node_id(entity.name)
        if node_id not in added_nodes:
            color = type_colors.get(entity.entity_type, "white")
            label = entity.name.replace('"', '\\"')
            lines.append(f'    {node_id} [label="{label}", fillcolor={color}, style=filled];')
            added_nodes.add(node_id)
    
    # Add relationships
    for rel in parser.relationships:
        source_id = _sanitize_node_id(rel.source)
        target_id = _sanitize_node_id(rel.target)
        
        # Ensure nodes exist
        if source_id not in added_nodes:
            lines.append(f'    {source_id} [label="{rel.source}", fillcolor=white, style=filled];')
            added_nodes.add(source_id)
        if target_id not in added_nodes:
            lines.append(f'    {target_id} [label="{rel.target}", fillcolor=white, style=filled];')
            added_nodes.add(target_id)
        
        relation_label = rel.relation.replace('"', '\\"')
        lines.append(f'    {source_id} -> {target_id} [label="{relation_label}"];')
    
    lines.append('}')
    
    return '\n'.join(lines)


def _sanitize_node_id(name: str) -> str:
    """Convert entity name to valid node ID."""
    # Replace spaces and special chars with underscores
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure starts with letter
    if safe and safe[0].isdigit():
        safe = 'n' + safe
    return safe or 'node'


def save_graph_visualization(kg_text: str, output_path: str, format: str = "mermaid") -> None:
    """Save graph visualization to file.
    
    Args:
        kg_text: Knowledge graph markdown text
        output_path: Output file path
        format: "mermaid", "graphviz", or "networkx" (pickle)
    """
    path = Path(output_path)
    
    if format == "mermaid":
        mermaid = export_to_mermaid(kg_text)
        path.write_text(mermaid, encoding="utf-8")
    
    elif format == "graphviz":
        dot = export_to_graphviz(kg_text)
        path.write_text(dot, encoding="utf-8")
    
    elif format == "networkx":
        G = export_to_networkx(kg_text)
        if G is not None:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(G, f)
        else:
            # Fallback to JSON representation
            import json
            parser = KnowledgeGraphParser(kg_text)
            data = {
                "entities": [
                    {"name": e.name, "type": e.entity_type, "description": e.description}
                    for e in parser.entities
                ],
                "relationships": [
                    {"source": r.source, "relation": r.relation, "target": r.target, "context": r.context}
                    for r in parser.relationships
                ]
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_graph_stats(kg_text: str) -> Dict[str, Any]:
    """Get statistics about the knowledge graph.
    
    Args:
        kg_text: Knowledge graph markdown text
        
    Returns:
        Dictionary with graph statistics
    """
    parser = KnowledgeGraphParser(kg_text)
    
    # Count entity types
    type_counts = {}
    for entity in parser.entities:
        type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1
    
    return {
        "entity_count": len(parser.entities),
        "relationship_count": len(parser.relationships),
        "entity_types": type_counts,
        "connected_ratio": len(parser.relationships) / max(len(parser.entities), 1),
    }
