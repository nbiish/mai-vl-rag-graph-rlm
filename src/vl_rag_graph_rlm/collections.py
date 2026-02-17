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


# ── Metadata ─────────────────────────────────────────────────────────


def set_metadata(name: str, key: str, value: Any) -> None:
    """Set custom metadata key-value pair for a collection.
    
    Args:
        name: Collection name
        key: Metadata key
        value: Metadata value (any JSON-serializable type)
    """
    meta = load_collection_meta(name)
    if "custom_metadata" not in meta:
        meta["custom_metadata"] = {}
    meta["custom_metadata"][key] = value
    save_collection_meta(name, meta)


def get_metadata(name: str, key: Optional[str] = None) -> Any:
    """Get custom metadata for a collection.
    
    Args:
        name: Collection name
        key: Optional specific key to retrieve (returns all if None)
        
    Returns:
        Metadata value or dict of all metadata
    """
    meta = load_collection_meta(name)
    custom = meta.get("custom_metadata", {})
    if key:
        return custom.get(key)
    return custom


def add_creation_note(name: str, note: str, author: str = "") -> None:
    """Add a creation note to the collection.
    
    Args:
        name: Collection name
        note: Note text
        author: Optional author identifier
    """
    meta = load_collection_meta(name)
    if "creation_notes" not in meta:
        meta["creation_notes"] = []
    meta["creation_notes"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "author": author,
    })
    save_collection_meta(name, meta)


def record_version(name: str, version: str, changes: str = "") -> None:
    """Record a version checkpoint for the collection.
    
    Args:
        name: Collection name
        version: Version string (e.g., "1.0.0")
        changes: Description of changes in this version
    """
    meta = load_collection_meta(name)
    if "versions" not in meta:
        meta["versions"] = []
    meta["versions"].append({
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "changes": changes,
        "documents": meta.get("document_count", 0),
        "chunks": meta.get("chunk_count", 0),
    })
    meta["current_version"] = version
    save_collection_meta(name, meta)


# ── Snapshots ─────────────────────────────────────────────────────────


def _snapshots_dir(name: str) -> Path:
    """Return the snapshots directory for a collection."""
    return _collection_dir(name) / "snapshots"


def create_snapshot(name: str, snapshot_name: Optional[str] = None) -> Path:
    """Create a point-in-time snapshot of a collection.
    
    Args:
        name: Collection name
        snapshot_name: Optional snapshot name (defaults to timestamp)
        
    Returns:
        Path to the created snapshot directory
    """
    slug = _sanitize_name(name)
    snapshot_id = snapshot_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snap_dir = _snapshots_dir(slug) / snapshot_id
    
    snap_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy embeddings
    emb_src = Path(_embeddings_path(slug))
    if emb_src.exists():
        shutil.copy2(emb_src, snap_dir / "embeddings.json")
    
    # Copy knowledge graph
    kg_src = _kg_path(slug)
    if kg_src.exists():
        shutil.copy2(kg_src, snap_dir / "knowledge_graph.md")
    
    # Copy metadata with snapshot info
    meta = load_collection_meta(slug)
    snapshot_meta = {
        **meta,
        "snapshot_id": snapshot_id,
        "snapshot_created": datetime.now(timezone.utc).isoformat(),
        "snapshot_of": slug,
    }
    (snap_dir / "collection.json").write_text(
        json.dumps(snapshot_meta, indent=2), encoding="utf-8"
    )
    
    # Update collection metadata with snapshot reference
    if "snapshots" not in meta:
        meta["snapshots"] = []
    meta["snapshots"].append({
        "id": snapshot_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "path": str(snap_dir),
    })
    save_collection_meta(slug, meta)
    
    return snap_dir


def list_snapshots(name: str) -> List[Dict[str, Any]]:
    """List all snapshots for a collection.
    
    Args:
        name: Collection name
        
    Returns:
        List of snapshot metadata
    """
    meta = load_collection_meta(name)
    return meta.get("snapshots", [])


def restore_snapshot(name: str, snapshot_id: str) -> None:
    """Restore a collection from a snapshot.
    
    Args:
        name: Collection name
        snapshot_id: Snapshot identifier to restore
    """
    slug = _sanitize_name(name)
    snap_dir = _snapshots_dir(slug) / snapshot_id
    
    if not snap_dir.exists():
        raise FileNotFoundError(f"Snapshot '{snapshot_id}' not found")
    
    # Restore embeddings
    emb_snap = snap_dir / "embeddings.json"
    if emb_snap.exists():
        shutil.copy2(emb_snap, _embeddings_path(slug))
    
    # Restore knowledge graph
    kg_snap = snap_dir / "knowledge_graph.md"
    if kg_snap.exists():
        shutil.copy2(kg_snap, _kg_path(slug))
    
    # Update metadata
    meta = load_collection_meta(slug)
    meta["restored_from"] = snapshot_id
    meta["restored_at"] = datetime.now(timezone.utc).isoformat()
    save_collection_meta(slug, meta)


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
        - embedding distribution
        - KG entity counts
        - query history (if tracked)
    """
    from datetime import datetime
    import numpy as np
    
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
    
    # Load embeddings for distribution analysis
    embedding_distribution = {}
    if emb_path.exists():
        try:
            with open(emb_path, 'r') as f:
                emb_data = json.load(f)
            
            # Analyze embedding dimensions and count
            embeddings = emb_data.get("embeddings", [])
            if embeddings:
                dims = [len(e.get("embedding", [])) for e in embeddings if e.get("embedding")]
                if dims:
                    embedding_distribution = {
                        "count": len(embeddings),
                        "dimensions": max(dims) if dims else 0,
                        "avg_magnitude": 0.0,  # Would need actual computation
                    }
            
            # Document type distribution
            docs = emb_data.get("documents", {})
            doc_types = {}
            for doc in docs.values():
                dtype = doc.get("type", "unknown")
                doc_types[dtype] = doc_types.get(dtype, 0) + 1
            embedding_distribution["document_types"] = doc_types
            
        except Exception:
            pass
    
    # Knowledge graph entity analysis
    kg_stats = {"size_bytes": len(kg), "has_content": bool(kg)}
    if kg:
        # Simple entity extraction from markdown-style KG
        entities = set()
        relationships = 0
        for line in kg.split("\n"):
            # Count lines that look like entity definitions or relationships
            if line.startswith("##") or line.startswith("###"):
                entities.add(line.strip("# "))
            elif "->" in line or "|" in line:
                relationships += 1
        kg_stats["estimated_entities"] = len(entities)
        kg_stats["estimated_relationships"] = relationships
    
    # Query history (if available in metadata)
    query_history = meta.get("query_history", [])
    
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
        "knowledge_graph_stats": kg_stats,
        "embedding_distribution": embedding_distribution,
        "query_history_count": len(query_history),
        "recent_queries": query_history[-5:] if query_history else [],
        "created": created,
        "updated": meta.get("updated"),
        "age_days": age_days,
        "embeddings_file_bytes": emb_size,
        "has_embeddings": emb_path.exists(),
        "has_knowledge_graph": bool(kg),
        "custom_metadata": meta.get("custom_metadata", {}),
        "current_version": meta.get("current_version", "unversioned"),
        "snapshot_count": len(meta.get("snapshots", [])),
    }


def print_collection_dashboard(name: str) -> None:
    """Print a formatted statistics dashboard for a collection."""
    stats = get_collection_stats(name)
    
    print(f"\n{'='*60}")
    print(f"Collection Dashboard: {stats['display_name'] or stats['name']}")
    print(f"{'='*60}")
    
    print(f"\n📊 Documents & Content:")
    print(f"  • Documents: {stats['document_count']:,}")
    print(f"  • Chunks: {stats['chunk_count']:,}")
    print(f"  • Sources: {stats['sources_count']}")
    print(f"  • Age: {stats['age_days']} days" if stats['age_days'] else "  • Age: N/A")
    
    print(f"\n🔧 Models:")
    print(f"  • Embedding: {stats['embedding_model']}")
    print(f"  • Reranker: {stats['reranker_model']}")
    
    print(f"\n📈 Embeddings:")
    emb = stats.get('embedding_distribution', {})
    if emb:
        print(f"  • Count: {emb.get('count', 'N/A'):,}")
        print(f"  • Dimensions: {emb.get('dimensions', 'N/A')}")
        if emb.get('document_types'):
            print(f"  • Document types:")
            for dtype, count in emb['document_types'].items():
                print(f"      - {dtype}: {count}")
    else:
        print(f"  • No embeddings loaded")
    
    print(f"\n🕸️ Knowledge Graph:")
    kg = stats.get('knowledge_graph_stats', {})
    if stats['has_knowledge_graph']:
        print(f"  • Size: {stats['knowledge_graph_size']:,} bytes")
        print(f"  • Estimated entities: {kg.get('estimated_entities', 'N/A')}")
        print(f"  • Estimated relationships: {kg.get('estimated_relationships', 'N/A')}")
    else:
        print(f"  • No knowledge graph")
    
    print(f"\n🏷️ Tags: {', '.join(stats['tags']) if stats['tags'] else 'None'}")
    
    if stats['recent_queries']:
        print(f"\n📝 Recent Queries:")
        for q in stats['recent_queries']:
            print(f"  • {q.get('query', 'N/A')[:50]}...")
    
    print(f"\n📦 Storage:")
    print(f"  • Embeddings file: {stats['embeddings_file_bytes']:,} bytes")
    print(f"  • Snapshots: {stats['snapshot_count']}")
    print(f"  • Version: {stats['current_version']}")
    
    print(f"{'='*60}\n")


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


# ── Collection Suggestions ────────────────────────────────────────────


def suggest_collections_for_query(
    query: str,
    top_k: int = 3,
    min_relevance_score: float = 0.1
) -> List[Dict[str, Any]]:
    """Recommend relevant collections based on query content.
    
    Uses keyword matching against collection names, descriptions, tags,
    and metadata to find the most relevant collections.
    
    Args:
        query: User query string
        top_k: Number of suggestions to return
        min_relevance_score: Minimum relevance threshold
        
    Returns:
        List of suggested collections with relevance scores
    """
    import re
    from collections import Counter
    
    all_collections = list_collections()
    if not all_collections:
        return []
    
    # Extract keywords from query (simple tokenization)
    query_lower = query.lower()
    query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
    
    scored_collections = []
    
    for coll in all_collections:
        score = 0.0
        match_details = []
        
        # Check name matches (high weight)
        name = coll.get("name", "").lower()
        display_name = coll.get("display_name", "").lower()
        for word in query_words:
            if word in name:
                score += 0.3
                match_details.append(f"name:{word}")
            if word in display_name:
                score += 0.25
                match_details.append(f"display:{word}")
        
        # Check description matches (medium weight)
        description = coll.get("description", "").lower()
        for word in query_words:
            if word in description:
                score += 0.2
                match_details.append(f"desc:{word}")
        
        # Check tag matches (high weight for exact matches)
        tags = [t.lower() for t in coll.get("tags", [])]
        for tag in tags:
            if tag in query_lower:
                score += 0.35
                match_details.append(f"tag:{tag}")
            # Partial tag matches
            for word in query_words:
                if word in tag or tag in word:
                    score += 0.15
                    match_details.append(f"tag_partial:{tag}")
        
        # Check custom metadata (low weight but included)
        custom_meta = coll.get("custom_metadata", {})
        for key, value in custom_meta.items():
            value_str = str(value).lower()
            key_str = key.lower()
            for word in query_words:
                if word in value_str or word in key_str:
                    score += 0.1
                    match_details.append(f"meta:{key}")
        
        # Boost score for document count (more content = more useful)
        doc_count = coll.get("document_count", 0)
        if doc_count > 0:
            # Logarithmic boost to avoid overshadowing keyword matches
            import math
            score += min(0.1 * math.log10(doc_count + 1), 0.2)
        
        if score >= min_relevance_score:
            scored_collections.append({
                "name": coll.get("name"),
                "display_name": coll.get("display_name"),
                "description": coll.get("description"),
                "relevance_score": round(score, 3),
                "document_count": doc_count,
                "chunk_count": coll.get("chunk_count", 0),
                "tags": coll.get("tags", []),
                "match_details": match_details[:5],  # Top 5 matches
            })
    
    # Sort by relevance score descending
    scored_collections.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return scored_collections[:top_k]


def print_collection_suggestions(query: str, top_k: int = 3) -> None:
    """Print formatted collection suggestions for a query."""
    suggestions = suggest_collections_for_query(query, top_k)
    
    if not suggestions:
        print(f"\nNo relevant collections found for: '{query[:50]}...'")
        print("Consider creating a new collection or broadening your query.")
        return
    
    print(f"\n📚 Recommended collections for: '{query[:50]}...'")
    print(f"{'='*60}")
    
    for i, sugg in enumerate(suggestions, 1):
        print(f"\n{i}. {sugg['display_name']} (score: {sugg['relevance_score']})")
        if sugg['description']:
            print(f"   Description: {sugg['description'][:80]}...")
        print(f"   Documents: {sugg['document_count']:,} | Chunks: {sugg['chunk_count']:,}")
        if sugg['tags']:
            print(f"   Tags: {', '.join(sugg['tags'])}")
        
        # Show how to use it
        print(f"   💡 Use: vrlmrag -c {sugg['name']} -q \"{query[:40]}...\"")
    
    print(f"{'='*60}\n")
