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
import tarfile
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
    embedding_model: str = "",
) -> Dict[str, Any]:
    """Create a new empty collection (or return existing metadata).
    
    Args:
        name: Collection name (will be sanitized to filesystem-safe slug)
        description: Optional description of the collection
        embedding_model: Name of the embedding model used (e.g., "Qwen/Qwen3-VL-Embedding-2B")
    """
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
        "embedding_model": embedding_model,  # Track embedding model version
        "model_history": [],  # Track model changes over time
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


def record_source(
    name: str, 
    source_path: str, 
    doc_count: int, 
    chunk_count: int,
    embedding_model: str = "",
    reranker_model: str = "",
) -> None:
    """Record that documents from *source_path* were added to the collection.
    
    Args:
        name: Collection name
        source_path: Path to the source documents
        doc_count: Number of documents added
        chunk_count: Number of chunks added
        embedding_model: Name of the embedding model used (e.g., "Qwen/Qwen3-VL-Embedding-2B")
        reranker_model: Name of the reranker model used (e.g., "ms-marco-MiniLM-L-12-v2")
    """
    meta = load_collection_meta(name)
    
    # Check if embedding model has changed
    prev_model = meta.get("embedding_model", "")
    if embedding_model and embedding_model != prev_model and prev_model:
        # Record model change in history
        if "model_history" not in meta:
            meta["model_history"] = []
        meta["model_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "previous_model": prev_model,
            "new_model": embedding_model,
            "source": str(Path(source_path).resolve()),
        })
        # Update current model
        meta["embedding_model"] = embedding_model
    elif embedding_model and not prev_model:
        # First time setting model
        meta["embedding_model"] = embedding_model
    
    # Track reranker model too
    if reranker_model:
        meta["reranker_model"] = reranker_model
    
    meta["sources"].append(
        {
            "path": str(Path(source_path).resolve()),
            "added": datetime.now(timezone.utc).isoformat(),
            "documents": doc_count,
            "chunks": chunk_count,
            "embedding_model": embedding_model,  # Track per-source model
        }
    )
    meta["document_count"] = meta.get("document_count", 0) + doc_count
    meta["chunk_count"] = meta.get("chunk_count", 0) + chunk_count
    save_collection_meta(name, meta)


def check_model_compatibility(name: str, target_model: str) -> Dict[str, Any]:
    """Check if collection can be safely used with target embedding model.
    
    Returns:
        Dict with compatibility info:
        - compatible: bool (True if same model or no model set)
        - current_model: str (the model currently used by collection)
        - target_model: str (the model being checked)
        - needs_reindex: bool (True if reindexing is recommended)
        - mixed_models: bool (True if collection has mixed model sources)
        - history: list of model changes
    """
    meta = load_collection_meta(name)
    current_model = meta.get("embedding_model", "")
    model_history = meta.get("model_history", [])
    
    # Check if any sources used different models
    sources = meta.get("sources", [])
    source_models = set(s.get("embedding_model", "") for s in sources if s.get("embedding_model"))
    mixed_models = len(source_models) > 1
    
    compatible = not current_model or current_model == target_model
    
    return {
        "compatible": compatible,
        "current_model": current_model,
        "target_model": target_model,
        "needs_reindex": not compatible,
        "mixed_models": mixed_models,
        "source_models": sorted(source_models),
        "history": model_history,
    }


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


# ── Export / Import ────────────────────────────────────────────────────


def export_collection(name: str, output_path: str) -> Path:
    """Export a collection as a portable tar.gz archive.
    
    Args:
        name: Collection name to export
        output_path: Path for the output archive (should end in .tar.gz)
        
    Returns:
        Path to the created archive
    """
    slug = _sanitize_name(name)
    cdir = _collection_dir(slug)
    
    if not cdir.exists():
        raise FileNotFoundError(f"Collection '{name}' does not exist")
    
    # Ensure output path has correct extension
    out_path = Path(output_path)
    if not out_path.name.endswith('.tar.gz'):
        out_path = out_path.with_suffix('.tar.gz')
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create archive
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(cdir, arcname=slug)
    
    return out_path


def import_collection(archive_path: str, new_name: Optional[str] = None) -> Dict[str, Any]:
    """Import a collection from a tar.gz archive.
    
    Args:
        archive_path: Path to the .tar.gz archive
        new_name: Optional new name for the imported collection
                  (defaults to archive's original name)
                  
    Returns:
        Metadata for the imported collection
    """
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    
    # Extract to temporary location first to inspect
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmpdir)
        
        # Find the collection directory in the archive
        tmp_path = Path(tmpdir)
        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        
        if not subdirs:
            raise ValueError("Invalid archive: no collection directory found")
        
        source_dir = subdirs[0]
        
        # Determine target name
        if new_name:
            target_slug = _sanitize_name(new_name)
        else:
            target_slug = source_dir.name
        
        # Check if collection already exists
        target_dir = _collection_dir(target_slug)
        if target_dir.exists():
            raise FileExistsError(
                f"Collection '{target_slug}' already exists. "
                f"Delete it first or specify a new name."
            )
        
        # Copy to collections root
        shutil.copytree(source_dir, target_dir)
        
        # Update metadata
        meta = load_collection_meta(target_slug)
        meta["imported_from"] = str(archive.resolve())
        meta["imported_at"] = datetime.now(timezone.utc).isoformat()
        if new_name:
            meta["display_name"] = new_name.strip()
        save_collection_meta(target_slug, meta)
        
        return meta


def merge_collections(source_name: str, target_name: str) -> Dict[str, Any]:
    """Merge one collection into another.
    
    Combines embeddings, knowledge graphs, and updates metadata.
    The source collection remains unchanged.
    
    Args:
        source_name: Collection to merge from
        target_name: Collection to merge into
        
    Returns:
        Updated metadata for the target collection
    """
    # Load both collections
    source_meta = load_collection_meta(source_name)
    target_meta = load_collection_meta(target_name)
    
    # Merge embeddings
    source_emb_path = _embeddings_path(source_name)
    target_emb_path = _embeddings_path(target_name)
    
    if Path(source_emb_path).exists():
        # Load source embeddings
        with open(source_emb_path, 'r') as f:
            source_emb = json.load(f)
        
        # Load target embeddings (or create empty)
        if Path(target_emb_path).exists():
            with open(target_emb_path, 'r') as f:
                target_emb = json.load(f)
        else:
            target_emb = {"documents": {}, "embeddings": [], "next_id": 0}
        
        # Merge documents (source takes precedence on ID collision)
        offset = target_emb.get("next_id", 0)
        for doc_id, doc in source_emb.get("documents", {}).items():
            new_id = str(int(doc_id) + offset)
            target_emb["documents"][new_id] = doc
            # Update embedding references
            for emb in source_emb.get("embeddings", []):
                if emb.get("doc_id") == doc_id:
                    new_emb = emb.copy()
                    new_emb["doc_id"] = new_id
                    target_emb["embeddings"].append(new_emb)
        
        target_emb["next_id"] = offset + source_emb.get("next_id", 0)
        
        # Save merged embeddings
        Path(target_emb_path).parent.mkdir(parents=True, exist_ok=True)
        with open(target_emb_path, 'w') as f:
            json.dump(target_emb, f, indent=2)
    
    # Merge knowledge graphs
    source_kg = load_kg(source_name)
    if source_kg:
        target_kg = load_kg(target_name)
        merged_kg = merge_kg(target_kg, source_kg)
        save_kg(target_name, merged_kg)
    
    # Update metadata
    target_meta["sources"].extend([
        {**s, "merged_from": source_name} 
        for s in source_meta.get("sources", [])
    ])
    target_meta["document_count"] = target_meta.get("document_count", 0) + source_meta.get("document_count", 0)
    target_meta["chunk_count"] = target_meta.get("chunk_count", 0) + source_meta.get("chunk_count", 0)
    target_meta["merged_sources"] = target_meta.get("merged_sources", []) + [source_name]
    
    save_collection_meta(target_name, target_meta)
    return target_meta


# ── Tagging & Search ─────────────────────────────────────────────────


def add_tags(name: str, tags: List[str]) -> None:
    """Add tags to a collection.
    
    Args:
        name: Collection name
        tags: List of tag strings to add
    """
    meta = load_collection_meta(name)
    if "tags" not in meta:
        meta["tags"] = []
    
    # Normalize tags: lowercase, no spaces, unique
    normalized = [t.lower().replace(" ", "-").strip() for t in tags if t.strip()]
    existing = set(meta["tags"])
    new_tags = [t for t in normalized if t not in existing]
    
    meta["tags"].extend(new_tags)
    save_collection_meta(name, meta)


def remove_tags(name: str, tags: List[str]) -> None:
    """Remove tags from a collection.
    
    Args:
        name: Collection name
        tags: List of tag strings to remove
    """
    meta = load_collection_meta(name)
    if "tags" not in meta or not meta["tags"]:
        return
    
    # Normalize tags to remove
    to_remove = {t.lower().replace(" ", "-").strip() for t in tags}
    meta["tags"] = [t for t in meta["tags"] if t not in to_remove]
    save_collection_meta(name, meta)


def search_collections(
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    embedding_model: Optional[str] = None,
    min_documents: Optional[int] = None,
    max_documents: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Search/filter collections by various criteria.
    
    Args:
        query: Optional text search on name/description
        tags: Optional list of tags to filter by (any match)
        embedding_model: Filter by embedding model used
        min_documents: Minimum document count
        max_documents: Maximum document count
        
    Returns:
        List of matching collection metadata
    """
    all_collections = list_collections()
    results = []
    
    # Normalize tags filter
    if tags:
        tag_filter = {t.lower().replace(" ", "-").strip() for t in tags}
    else:
        tag_filter = None
    
    query_lower = query.lower() if query else None
    
    for coll in all_collections:
        # Text search
        if query_lower:
            name_match = query_lower in coll.get("name", "").lower()
            desc_match = query_lower in coll.get("description", "").lower()
            display_match = query_lower in coll.get("display_name", "").lower()
            if not (name_match or desc_match or display_match):
                continue
        
        # Tag filter (any tag matches)
        if tag_filter:
            coll_tags = set(t.lower() for t in coll.get("tags", []))
            if not tag_filter & coll_tags:
                continue
        
        # Embedding model filter
        if embedding_model:
            if embedding_model.lower() not in coll.get("embedding_model", "").lower():
                continue
        
        # Document count filters
        doc_count = coll.get("document_count", 0)
        if min_documents is not None and doc_count < min_documents:
            continue
        if max_documents is not None and doc_count > max_documents:
            continue
        
        results.append(coll)
    
    return results


# ── Statistics ──────────────────────────────────────────────────────


def get_collection_stats(name: str) -> Dict[str, Any]:
    """Get comprehensive statistics for a collection.
    
    Returns:
        Dict with collection statistics including:
        - document_count, chunk_count
        - source count, tag count
        - embedding model info
        - knowledge graph size
        - age (days since creation)
        - last update time
    """
    from datetime import datetime
    
    meta = load_collection_meta(name)
    kg = load_kg(name)
    
    # Calculate age
    created = meta.get("created", "")
    age_days = None
    if created:
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - created_dt).days
        except:
            pass
    
    # Embeddings file size
    emb_path = Path(_embeddings_path(name))
    emb_size = emb_path.stat().st_size if emb_path.exists() else 0
    
    return {
        "name": meta.get("name"),
        "display_name": meta.get("display_name"),
        "document_count": meta.get("document_count", 0),
        "chunk_count": meta.get("chunk_count", 0),
        "sources_count": len(meta.get("sources", [])),
        "tags_count": len(meta.get("tags", [])),
        "tags": meta.get("tags", []),
        "embedding_model": meta.get("embedding_model", "unknown"),
        "reranker_model": meta.get("reranker_model", "unknown"),
        "knowledge_graph_size": len(kg),
        "created": created,
        "updated": meta.get("updated"),
        "age_days": age_days,
        "embeddings_file_bytes": emb_size,
        "has_embeddings": emb_path.exists(),
        "has_knowledge_graph": bool(kg),
    }


def get_global_stats() -> Dict[str, Any]:
    """Get global statistics across all collections.
    
    Returns:
        Dict with aggregated statistics:
        - total_collections
        - total_documents, total_chunks
        - model breakdown
        - tag distribution
    """
    all_collections = list_collections()
    
    total_docs = sum(c.get("document_count", 0) for c in all_collections)
    total_chunks = sum(c.get("chunk_count", 0) for c in all_collections)
    
    # Model breakdown
    model_counts = {}
    for c in all_collections:
        model = c.get("embedding_model", "unknown") or "unknown"
        model_counts[model] = model_counts.get(model, 0) + 1
    
    # Tag distribution
    tag_counts = {}
    for c in all_collections:
        for tag in c.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    return {
        "total_collections": len(all_collections),
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "average_documents": total_docs / len(all_collections) if all_collections else 0,
        "average_chunks": total_chunks / len(all_collections) if all_collections else 0,
        "model_distribution": model_counts,
        "tag_distribution": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)),
        "total_unique_tags": len(tag_counts),
    }
