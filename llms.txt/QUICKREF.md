# Quick Reference — VL-RAG-Graph-RLM MCP Tools

> **For LLMs**: Ultra-simplified tool reference. Default = comprehensive. Use `mode="fast"` only when speed matters.

---

## 3 Core Tools

### 1. `analyze` — Analyze Documents

**When to use**: Any document analysis, research, summarization, extraction.

**Parameters**:
```json
{
  "input_path": "./my-document.pdf",  // Required: file or folder
  "query": "Summarize the key findings",  // Optional: your question
  "mode": "comprehensive",  // Optional: "comprehensive" (default) or "fast"
  "output_path": null  // Optional: save report to file
}
```

**Note**: Provider and model are configured via `.env` file only.

**Examples**:
```json
// Full analysis (default - always use this unless user says "fast")
{"input_path": "./research-papers", "query": "What are the main conclusions?"}

// Fast search only when explicitly requested
{"input_path": "./notes.md", "query": "Quick summary", "mode": "fast"}
```

---

### 2. `query_collection` — Query Knowledge Collections

**When to use**: Querying previously created knowledge bases, blending multiple collections.

**Parameters**:
```json
{
  "collection": "research",  // Required: collection name
  "query": "What did the papers say about X?",  // Required: your question
  "mode": "comprehensive"  // Optional: "comprehensive" (default) or "fast"
}
```

**Note**: Provider and model are configured via `.env` file only.

**Examples**:
```json
// Query single collection
{"collection": "research", "query": "Explain the methodology"}

// Blend multiple collections (repeat -c in CLI, array in API)
{"collection": "research,codebase", "query": "How does the code implement the paper's algorithm?"}
```

---

### 3. `collection_manage` — Manage Collections

**When to use**: Adding documents to collections, listing, exporting, merging collections.

**Parameters**:
```json
{
  "action": "add",  // Required: see actions below
  "collection": "research",  // Required for most actions
  "path": "./new-papers/",  // Required for "add" action
  "target_collection": null,  // Required for "merge" action
  "query": null,  // Optional: for "search" action
  "tags": null  // Optional: for "tag" action
}
```

**Actions**:
| Action | Description | Required Params |
|--------|-------------|-----------------|
| `add` | Add documents to collection | `collection`, `path` |
| `list` | List all collections | none |
| `info` | Show collection details | `collection` |
| `delete` | Remove collection | `collection` |
| `export` | Export to tar.gz | `collection`, `path` (output file) |
| `import` | Import from tar.gz | `path` (input file) |
| `merge` | Merge collections | `collection` (source), `target_collection` |
| `tag` | Add tags | `collection`, `tags` |
| `search` | Find collections | `query` or `tags` |

**Examples**:
```json
// Add documents to collection
{"action": "add", "collection": "research", "path": "./papers/"}

// List all collections
{"action": "list"}

// Export collection
{"action": "export", "collection": "research", "path": "./research-backup.tar.gz"}
```

---

## Mode Selection (Always Comprehensive by Default)

| Mode | Use When | Features |
|------|----------|----------|
| `comprehensive` | **ALWAYS DEFAULT** — use this unless user explicitly says "fast" | Full VL-RAG-Graph-RLM: vision, audio, RAG, reranking, knowledge graph, recursive LLM |
| `fast` | Only when user explicitly requests speed over depth | Quick keyword + dense search, minimal processing |

**Rule**: If user doesn't specify mode, use `comprehensive`. Only use `fast` when user says things like "quick summary", "fast answer", "just search", "don't analyze deeply".

---

## What Works (Feb 12, 2026)

✅ **9/9 providers working**: SambaNova, ModalResearch, Nebius, Ollama, Groq, Cerebras, Z.AI, ZenMux, OpenRouter

✅ **Universal fallback API keys**: All 20+ providers support `{PROVIDER}_API_KEY_FALLBACK`

✅ **Collection operations**: Create, add, query, delete, export, import, merge, tag all working

✅ **Document types**: PPTX, PDF, DOCX, TXT, MD, CSV, XLSX, images, video (API mode), audio (API mode)

✅ **Three-tier resilience**: Primary key → Fallback key → Model fallback → Provider hierarchy

✅ **API embedding**: Working with OpenRouter text-embeddings + ZenMux omni for multimodal

✅ **Knowledge graph**: Extraction, persistence, merging across runs

✅ **Timeout handling**: Dynamic timeouts for reasoning models (120s normal, 300s reasoning, max 600s)

---

## What Needs Attention

⚠️ **flashrank dependency**: Not installed by default. Install with `uv pip install -e ".[reranker]"`

⚠️ **API embedding bug fixed**: Collections now properly embed documents in API mode (was creating 0 embeddings before Feb 12 fix)

⚠️ **Video/audio requires API mode**: Local models blocked for media to prevent OOM crashes

⚠️ **Circuit breaker for provider hierarchy**: Still pending — individual providers have circuit breakers but full hierarchy doesn't yet

⚠️ **Google-generativeai deprecation**: Migration to `google-genai` pending

---

## Quick Commands

```bash
# Analyze document (comprehensive default)
vrlmrag ./document.pdf -q "Summarize"

# Fast search only
vrlmrag ./notes.md -q "Quick lookup" --profile fast

# Create and populate collection
vrlmrag -c research --add ./papers/

# Query collection
vrlmrag -c research -q "What are the findings?"

# Blend collections
vrlmrag -c research -c codebase -q "How does code implement the paper?"
```

---

## Environment (One-Time Setup)

```bash
# Copy example env file
cp .env.example .env

# Add at least one provider API key
OPENROUTER_API_KEY=sk-or-v1-your-key
# Optional: fallback key for multi-account
OPENROUTER_API_KEY_FALLBACK=sk-or-v1-backup-key
```

Hierarchy auto-falls through: ModalResearch → SambaNova → Nebius → Ollama → Groq → Cerebras → Z.AI → ZenMux → OpenRouter → ...
