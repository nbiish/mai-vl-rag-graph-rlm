"""MCP server settings loader.

Resolves configuration from (in priority order):
    1. Environment variables (VRLMRAG_* prefix) — highest priority
    2. $VRLMRAG_MCP_SETTINGS env var → path to a JSON file
    3. <project_root>/.vrlmrag/mcp_settings.json
    4. Built-in defaults (provider hierarchy, no model/template override)

Env vars override everything, so each MCP client (Windsurf, Claude Desktop,
Cursor, etc.) can set its own config in the ``env`` block without touching
the codebase.

Env var mapping (all optional):
    VRLMRAG_PROVIDER           — "auto", "sambanova", "nebius", etc.
    VRLMRAG_MODEL              — explicit model override
    VRLMRAG_TEMPLATE           — template shorthand name
    VRLMRAG_MAX_DEPTH          — integer
    VRLMRAG_MAX_ITERATIONS     — integer
    VRLMRAG_TEMPERATURE        — float
    VRLMRAG_COLLECTIONS        — "true" or "false"
    VRLMRAG_COLLECTIONS_ROOT   — path override for collections directory
    VRLMRAG_LOG_LEVEL          — "DEBUG", "INFO", "WARNING", etc.

Settings file schema (all fields optional):
{
    "provider": "auto",
    "model": null,
    "template": null,
    "max_depth": 3,
    "max_iterations": 10,
    "temperature": 0.0,
    "collections_enabled": true,
    "collections_root": null,
    "log_level": "INFO"
}

When provider is "auto" (the default), the hierarchy system is used:
    sambanova → nebius → groq → cerebras → zai → zenmux → openrouter → ...
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

# Well-known templates: shorthand names that map to provider + model combos
TEMPLATES: dict[str, dict[str, str]] = {
    "fast-free": {"provider": "sambanova", "model": "DeepSeek-V3.2"},
    "fast-groq": {"provider": "groq", "model": "moonshotai/kimi-k2-instruct-0905"},
    "fast-cerebras": {"provider": "cerebras", "model": "zai-glm-4.7"},
    "nebius-m2": {"provider": "nebius", "model": "MiniMaxAI/MiniMax-M2.1"},
    "openrouter-cheap": {"provider": "openrouter", "model": "minimax/minimax-m2.1"},
    "openai-mini": {"provider": "openai", "model": "gpt-4o-mini"},
    "anthropic-haiku": {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
    "gemini-flash": {"provider": "gemini", "model": "gemini-1.5-flash"},
    "deepseek-chat": {"provider": "deepseek", "model": "deepseek-chat"},
}


@dataclass
class MCPSettings:
    """Resolved MCP server settings."""

    provider: str = "auto"
    model: Optional[str] = None
    template: Optional[str] = None
    max_depth: int = 3
    max_iterations: int = 10
    temperature: float = 0.0
    use_api: bool = True
    collections_enabled: bool = True
    collections_root: Optional[str] = None
    log_level: str = "INFO"

    def resolve_provider_model(self) -> tuple[str, Optional[str]]:
        """Resolve effective provider and model from settings.

        Priority:
            1. Explicit model override (provider + model both set)
            2. Template shorthand (expands to provider + model)
            3. Provider only (model resolved by provider defaults)
            4. "auto" → hierarchy system picks provider + model

        Returns:
            (provider, model) — model may be None if hierarchy/defaults apply.
        """
        # Template takes precedence over bare provider when no explicit model
        if self.template and self.template in TEMPLATES:
            tmpl = TEMPLATES[self.template]
            # Explicit model overrides template model
            effective_model = self.model or tmpl.get("model")
            # Explicit provider overrides template provider (unless auto)
            effective_provider = (
                self.provider if self.provider != "auto" else tmpl["provider"]
            )
            return effective_provider, effective_model

        return self.provider, self.model


def _settings_paths() -> list[Path]:
    """Return candidate settings file paths in priority order."""
    paths: list[Path] = []

    env_path = os.getenv("VRLMRAG_MCP_SETTINGS")
    if env_path:
        paths.append(Path(env_path))

    paths.append(_PROJECT_ROOT / ".vrlmrag" / "mcp_settings.json")
    return paths


def _parse_bool(value: str) -> bool:
    """Parse a boolean from an env var string."""
    return value.strip().lower() in ("true", "1", "yes", "on")


def _apply_env_overrides(settings: MCPSettings) -> MCPSettings:
    """Apply VRLMRAG_* env var overrides on top of existing settings.

    Env vars always take priority over the settings file, so each MCP
    client can configure its own behavior via its ``env`` block.
    """
    env_map = {
        "VRLMRAG_PROVIDER": "provider",
        "VRLMRAG_MODEL": "model",
        "VRLMRAG_TEMPLATE": "template",
        "VRLMRAG_MAX_DEPTH": "max_depth",
        "VRLMRAG_MAX_ITERATIONS": "max_iterations",
        "VRLMRAG_TEMPERATURE": "temperature",
        "VRLMRAG_USE_API": "use_api",
        "VRLMRAG_COLLECTIONS": "collections_enabled",
        "VRLMRAG_COLLECTIONS_ROOT": "collections_root",
        "VRLMRAG_LOG_LEVEL": "log_level",
    }

    overrides: list[str] = []
    for env_key, field_name in env_map.items():
        value = os.getenv(env_key)
        if value is None:
            continue

        if field_name in ("collections_enabled", "use_api"):
            setattr(settings, field_name, _parse_bool(value))
        elif field_name in ("max_depth", "max_iterations"):
            try:
                setattr(settings, field_name, int(value))
            except ValueError:
                logger.warning("Invalid integer for %s: %s", env_key, value)
                continue
        elif field_name == "temperature":
            try:
                setattr(settings, field_name, float(value))
            except ValueError:
                logger.warning("Invalid float for %s: %s", env_key, value)
                continue
        else:
            setattr(settings, field_name, value if value else None)

        overrides.append(f"{env_key}={value}")

    if overrides:
        logger.info("Env var overrides applied: %s", ", ".join(overrides))

    return settings


def load_settings() -> MCPSettings:
    """Load MCP settings from file, then apply env var overrides.

    Resolution order (highest priority first):
        1. VRLMRAG_* environment variables
        2. Settings JSON file ($VRLMRAG_MCP_SETTINGS or .vrlmrag/mcp_settings.json)
        3. Built-in defaults
    """
    settings = MCPSettings()

    for candidate in _settings_paths():
        if candidate.is_file():
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                logger.info("Loaded MCP settings from %s", candidate)
                settings = MCPSettings(
                    provider=raw.get("provider", "auto"),
                    model=raw.get("model"),
                    template=raw.get("template"),
                    max_depth=raw.get("max_depth", 3),
                    max_iterations=raw.get("max_iterations", 10),
                    temperature=raw.get("temperature", 0.0),
                    use_api=raw.get("use_api", True),
                    collections_enabled=raw.get("collections_enabled", True),
                    collections_root=raw.get("collections_root"),
                    log_level=raw.get("log_level", "INFO"),
                )
                break
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not parse %s: %s", candidate, exc)
    else:
        logger.info("No MCP settings file found — using built-in defaults")

    return _apply_env_overrides(settings)
