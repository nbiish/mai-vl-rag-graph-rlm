"""MCP server settings loader.

Minimal configuration for maximum performance:
    - Always uses comprehensive RAG (max_depth=5, max_iterations=15)
    - Always uses API provider hierarchy with auto-fallback
    - Only two settings exposed to users:
        1. VRLMRAG_LOCAL — "true" to use local models (default: false, use APIs)
        2. VRLMRAG_COLLECTIONS — "false" to disable collection tools (default: true)

All other settings are hardcoded for optimal comprehensive analysis.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Locate the VL-RAG-Graph-RLM project root directory."""
    env_root = os.getenv("VRLMRAG_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / ".env").exists() or (p / "pyproject.toml").exists():
            return p

    file_root = Path(__file__).resolve().parent.parent.parent.parent
    if (file_root / "pyproject.toml").exists():
        return file_root

    cwd = Path.cwd()
    if (cwd / ".env").exists() or (cwd / "pyproject.toml").exists():
        return cwd

    return file_root


_PROJECT_ROOT = _find_project_root()


def _parse_bool(value: str) -> bool:
    """Parse a boolean from an env var string."""
    return value.strip().lower() in ("true", "1", "yes", "on")


@dataclass
class MCPSettings:
    """MCP server settings — comprehensive defaults, minimal configuration."""

    # Hardcoded comprehensive defaults
    provider: str = "auto"  # Always use API hierarchy
    model: Optional[str] = None  # Let hierarchy resolve
    max_depth: int = 5  # Comprehensive: deep RLM recursion
    max_iterations: int = 15  # Comprehensive: high iteration limit
    temperature: float = 0.0  # Consistent, deterministic
    multi_query: bool = True  # Comprehensive: multi-query retrieval
    graph_augmented: bool = True  # Comprehensive: graph context
    graph_hops: int = 3  # Comprehensive: deep graph traversal
    verbose: bool = True  # Comprehensive: detailed output
    
    # User-configurable settings only
    use_local: bool = False  # VRLMRAG_LOCAL — default to APIs, not local models
    collections_enabled: bool = True  # VRLMRAG_COLLECTIONS — default on
    
    # Internal
    collections_root: Optional[str] = None
    log_level: str = "INFO"

    def resolve_provider_model(self) -> tuple[str, Optional[str]]:
        """Resolve effective provider and model.
        
        Returns:
            (provider, model) — always uses auto/hierarchy by default.
        """
        return self.provider, self.model


def load_settings() -> MCPSettings:
    """Load MCP settings with comprehensive defaults.
    
    Only VRLMRAG_LOCAL and VRLMRAG_COLLECTIONS are configurable.
    Everything else is hardcoded for maximum analysis quality.
    """
    settings = MCPSettings()
    
    # Apply minimal user overrides
    local_override = os.getenv("VRLMRAG_LOCAL")
    if local_override is not None:
        settings.use_local = _parse_bool(local_override)
        logger.info("VRLMRAG_LOCAL=%s → use_local=%s", local_override, settings.use_local)
    
    collections_override = os.getenv("VRLMRAG_COLLECTIONS")
    if collections_override is not None:
        settings.collections_enabled = _parse_bool(collections_override)
        logger.info("VRLMRAG_COLLECTIONS=%s → collections_enabled=%s", 
                    collections_override, settings.collections_enabled)
    
    # Root is always set via env for MCP
    root_override = os.getenv("VRLMRAG_ROOT")
    if root_override:
        logger.info("VRLMRAG_ROOT=%s", root_override)
    
    return settings
