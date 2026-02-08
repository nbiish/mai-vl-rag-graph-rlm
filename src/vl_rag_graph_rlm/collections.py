"""Named persistent knowledge collections for VL-RAG-Graph-RLM.

Collections are named, persistent stores of embeddings and knowledge graphs
that live inside the codebase at ``collections/``.  They can be populated
from any path, queried from anywhere, blended together, and scripted via
CLI without user interaction.

Storage layout::

    <project_root>/collections/
    ├── <name>/
    │   ├── collection.json      # metadata
    │   ├── embeddings.json      # Qwen3-VL embeddings
    │   └── knowledge_graph.md   # accumulated KG
    └── ...

Typical CLI usage::

    vrlmrag -c research --add ./papers/
    vrlmrag -c research -q "Summarize key findings"
    vrlmrag -c research -c code-docs -q "How does the code implement the paper?"
    vrlmrag --collection-list
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Collections live in <project_root>/collections/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COLLECTIONS_ROOT = _PROJECT_ROOT / "collections"


def _sanitize_name(name: str) -> str:
    """Normalise a collection name to a filesystem-safe slug."""
    slug = name.strip().lower()
    slug = slug.replace(" ", "-")
    # Keep only alphanumeric, hyphens, underscores
    slug = "".join(c for c in slug if c.isalnum() or c in ("-", "_"))
    if not slug:
        raise ValueError(f"Invalid collection name: {name!r}")
    return slug


def _collection_dir(name: str) -> Path:
    """Return the directory for a named collection."""
    return COLLECTIONS_ROOT / _sanitize_name(name)


def _meta_path(name: str) -> Path:
    return _collection_dir(name) / "collection.json"


def _embeddings_path(name: str) -> str:
    return str(_collection_dir(name) / "embeddings.json")


def _kg_path(name: str) -> Path:
    return _collection_dir(name) / "knowledge_graph.md"


# ── CRUD helpers ───────────────────────────────────────────────────────


def collection_exists(name: str) -> bool:
    """Check whether a named collection exists on disk."""
    return _meta_path(name).exists()


def create_collection(
    name: str,
    description: str = "",
) -> Dict[str, Any]:
    """Create a new empty collection (or return existing metadata)."""
    slug = _sanitize_name(name)
    cdir = _collection_dir(slug)
    meta_file = cdir / "collection.json"

    if meta_file.exists():
        return load_collection_meta(slug)

    cdir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "name": slug,
        "display_name": name.strip(),
        "description": description,
        "created": datetime.now(timezone.utc).isoformat(),
        "updated": datetime.now(timezone.utc).isoformat(),
        "sources": [],
        "document_count": 0,
        "chunk_count": 0,
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_collection_meta(name: str) -> Dict[str, Any]:
    """Load metadata for a named collection."""
    meta_file = _meta_path(name)
    if not meta_file.exists():
        raise FileNotFoundError(f"Collection '{name}' does not exist")
    return json.loads(meta_file.read_text(encoding="utf-8"))


def save_collection_meta(name: str, meta: Dict[str, Any]) -> None:
    """Persist updated metadata for a collection."""
    meta["updated"] = datetime.now(timezone.utc).isoformat()
    _meta_path(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def list_collections() -> List[Dict[str, Any]]:
    """Return metadata for every collection on disk."""
    if not COLLECTIONS_ROOT.exists():
        return []
    results = []
    for child in sorted(COLLECTIONS_ROOT.iterdir()):
        meta_file = child / "collection.json"
        if child.is_dir() and meta_file.exists():
            try:
                results.append(json.loads(meta_file.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
    return results


def delete_collection(name: str) -> bool:
    """Delete a collection and all its data.  Returns True if it existed."""
    cdir = _collection_dir(name)
    if cdir.exists():
        shutil.rmtree(cdir)
        return True
    return False


def record_source(name: str, source_path: str, doc_count: int, chunk_count: int) -> None:
    """Record that documents from *source_path* were added to the collection."""
    meta = load_collection_meta(name)
    meta["sources"].append(
        {
            "path": str(Path(source_path).resolve()),
            "added": datetime.now(timezone.utc).isoformat(),
            "documents": doc_count,
            "chunks": chunk_count,
        }
    )
    meta["document_count"] = meta.get("document_count", 0) + doc_count
    meta["chunk_count"] = meta.get("chunk_count", 0) + chunk_count
    save_collection_meta(name, meta)


# ── Knowledge-graph helpers ────────────────────────────────────────────


def load_kg(name: str) -> str:
    """Load the knowledge graph for a collection."""
    kgp = _kg_path(name)
    if kgp.exists():
        return kgp.read_text(encoding="utf-8")
    return ""


def save_kg(name: str, kg_text: str) -> None:
    """Persist the knowledge graph for a collection."""
    kgp = _kg_path(name)
    kgp.parent.mkdir(parents=True, exist_ok=True)
    kgp.write_text(kg_text, encoding="utf-8")


def merge_kg(existing: str, new_fragment: str) -> str:
    """Merge a new KG fragment into an existing knowledge graph."""
    if not existing:
        return new_fragment
    if not new_fragment:
        return existing
    return f"{existing}\n\n---\n\n{new_fragment}"
