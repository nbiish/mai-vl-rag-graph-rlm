"""Configuration profiles for VL-RAG-Graph-RLM.

Provides preset configurations optimized for different use cases:
- fast: Quick results, minimal resources
- balanced: Good quality with reasonable speed (default)
- thorough: Maximum accuracy, comprehensive analysis
- comprehensive: All best features enabled (new default recommendation)
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ProfileConfig:
    """Configuration profile settings."""
    
    # RAG settings
    top_k_dense: int = 50
    top_k_keyword: int = 50
    rrf_k: int = 60
    rrf_dense_weight: float = 4.0
    rrf_keyword_weight: float = 1.0
    rerank_candidates: int = 30
    final_context_chunks: int = 10
    
    # RLM settings
    max_depth: int = 3
    max_iterations: int = 10
    
    # Graph settings
    graph_augmented: bool = False
    graph_hops: int = 2
    
    # Query settings
    multi_query: bool = False
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 100
    
    # Output settings
    output_format: str = "markdown"
    verbose: bool = False
    
    # Description for documentation
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Predefined profiles
PROFILES: Dict[str, ProfileConfig] = {
    "fast": ProfileConfig(
        top_k_dense=30,
        top_k_keyword=30,
        rerank_candidates=15,
        final_context_chunks=5,
        max_depth=2,
        max_iterations=5,
        chunk_size=800,
        chunk_overlap=50,
        description="Fast results with minimal resource usage. Good for quick lookups."
    ),
    
    "balanced": ProfileConfig(
        top_k_dense=50,
        top_k_keyword=50,
        rerank_candidates=30,
        final_context_chunks=10,
        max_depth=3,
        max_iterations=10,
        chunk_size=1000,
        chunk_overlap=100,
        description="Balanced quality and speed. Good for general use."
    ),
    
    "thorough": ProfileConfig(
        top_k_dense=100,
        top_k_keyword=100,
        rerank_candidates=50,
        final_context_chunks=15,
        max_depth=5,
        max_iterations=15,
        chunk_size=1500,
        chunk_overlap=150,
        multi_query=True,
        graph_augmented=True,
        graph_hops=3,
        description="Maximum accuracy with comprehensive analysis. Good for research."
    ),
    
    "comprehensive": ProfileConfig(
        # RAG: Best retrieval
        top_k_dense=100,
        top_k_keyword=100,
        rrf_dense_weight=4.0,
        rrf_keyword_weight=2.0,  # Increased keyword weight
        rerank_candidates=50,
        final_context_chunks=15,
        
        # RLM: Deep reasoning
        max_depth=5,
        max_iterations=15,
        
        # Graph: Full KG utilization
        graph_augmented=True,
        graph_hops=3,
        
        # Query: Multi-angle
        multi_query=True,
        
        # Chunking: Optimal granularity
        chunk_size=1200,
        chunk_overlap=120,
        
        # Output: Detailed
        output_format="markdown",
        verbose=True,
        
        description="All best features enabled. Maximum quality for important analysis."
    ),
}


def get_profile(name: str) -> ProfileConfig:
    """Get a configuration profile by name.
    
    Args:
        name: Profile name (fast, balanced, thorough, comprehensive)
        
    Returns:
        ProfileConfig instance
        
    Raises:
        ValueError: If profile name is unknown
    """
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    
    return PROFILES[name]


def list_profiles() -> Dict[str, str]:
    """List all available profiles with descriptions.
    
    Returns:
        Dict mapping profile names to descriptions
    """
    return {name: cfg.description for name, cfg in PROFILES.items()}


def apply_profile(args: Any, profile_name: str) -> None:
    """Apply a profile to CLI args object (modifies in place).
    
    Only sets values that haven't been explicitly overridden by user.
    
    Args:
        args: argparse.Namespace or similar object
        profile_name: Name of profile to apply
    """
    profile = get_profile(profile_name)
    
    # Mapping of profile fields to arg names
    field_map = {
        "max_depth": "max_depth",
        "max_iterations": "max_iterations",
        "graph_augmented": "graph_augmented",
        "graph_hops": "graph_hops",
        "multi_query": "multi_query",
        "chunk_size": "chunk_size",
        "chunk_overlap": "chunk_overlap",
        "output_format": "format",
        "verbose": "verbose",
    }
    
    # Only apply if not already set (check for default values)
    for profile_field, arg_name in field_map.items():
        value = getattr(profile, profile_field)
        current = getattr(args, arg_name, None)
        
        # Apply if current is None or at default
        if current is None:
            setattr(args, arg_name, value)


def get_profile_summary(profile_name: str) -> str:
    """Get a formatted summary of a profile.
    
    Args:
        profile_name: Name of the profile
        
    Returns:
        Formatted summary string
    """
    profile = get_profile(profile_name)
    
    lines = [
        f"Profile: {profile_name}",
        f"Description: {profile.description}",
        "",
        "Settings:",
        f"  RAG: top_k={profile.top_k_dense}/{profile.top_k_keyword}, "
        f"rerank={profile.rerank_candidates}, final={profile.final_context_chunks}",
        f"  RLM: depth={profile.max_depth}, iterations={profile.max_iterations}",
        f"  Graph: augmented={profile.graph_augmented}, hops={profile.graph_hops}",
        f"  Query: multi_query={profile.multi_query}",
        f"  Chunking: size={profile.chunk_size}, overlap={profile.chunk_overlap}",
    ]
    
    return "\n".join(lines)
