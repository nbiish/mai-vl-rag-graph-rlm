# YOLO PLAN — Update llms.txt Folder

## Overview
The llms.txt folder contains project documentation (PRD, RULES, TODO, ARCHITECTURE). This plan ensures documentation aligns with the current codebase state (v0.1.0) and includes all 17 providers, the 6-pillar architecture, and recent additions like Cerebras.

## Analysis Findings
- Current files: ARCHITECTURE.md, PRD.md, RULES.md, TODO.md
- New provider added: `provider_cerebras.py` (not yet fully documented)
- ARCHITECTURE.md mentions Cerebras but needs verification of all component mappings
- Documentation has been recently updated but may need further alignment

## Tasks

- [x] Verify and update ARCHITECTURE.md — Component Map section
  - [x] Cross-check all source files against documented component map
  - [x] Verify `src/vl_rag_graph_rlm/core/` modules are documented (parser.py, prompts.py, repl.py)
  - [x] Verify `src/vl_rag_graph_rlm/utils/` modules are documented (parsing.py, prompts.py)
  - [x] Verify `src/vl_rag_graph_rlm/environments/` is documented (repl.py)
  - [x] Add missing components if any
  - **Added components:**
    - Core Parser (`core/parser.py`) — FINAL/FINAL_VAR statement extraction
    - Core Prompts (`core/prompts.py`) — System prompt templates for RLM
    - Core REPL (`core/repl.py`) — REPLExecutor with RestrictedPython sandbox
    - Utils Parsing (`utils/parsing.py`) — LLM response parsing, code block extraction
    - Utils Prompts (`utils/prompts.py`) — Additional prompt utilities
    - RAG Provider (`rag/provider.py`) — RAG provider interface
    - RAG Store (`rag/store.py`) — Vector store base module
    - RAG Reranker (`rag/reranker.py`) — Reranker implementations
    - ERNIE Client (`rag/ernie_client.py`) — Baidu ERNIE / OpenAI-compatible client
    - Environments REPL (`environments/repl.py`) — Alternative safe Python execution sandbox
  - **Updated CLI entry:** Changed "15 providers" to "17 providers"

- [x] Verify and update ARCHITECTURE.md — Templates table
  - [x] Verify all 17 provider templates are listed
  - [x] Confirm `provider_cerebras.py` is documented
  - [x] Verify "Full Pipeline" status for all templates

- [x] Verify and update PRD.md — Supported Providers table
  - [x] Ensure all 17 providers are listed with correct default models
  - [x] Verify Cerebras entry (llama-4-scout-17b-16e-instruct)
  - [x] Check context window specifications

- [x] Update PRD.md — CLI section
  - [x] Verify provider count in documentation (should be 17 providers total)
  - [x] Update "15 providers supported" comments to reflect accurate count

- [x] Verify and update RULES.md
  - [x] Cross-check rules against actual implementation patterns
  - [x] Ensure device detection rules are accurate (MPS/CUDA/CPU)
  - [x] Verify Qwen3-VL fallback patterns
  - **Updated:**
    - Clarified factory function imports (`from vl_rag_graph_rlm.rag`)
    - Corrected "Never" section: device detection should be dynamic (CUDA/CPU), not hardcoded
    - Added new "Device Detection" section documenting actual patterns in codebase
    - Fixed Qwen3-VL fallback pattern to match actual implementation (`try/except ImportError` with `HAS_QWEN3VL` check)
    - Updated MPS detection guidance to match template patterns

- [x] Verify and update TODO.md
  - [x] Mark completed v0.1.0 items
  - [x] Ensure Cerebras provider is listed in completed items
  - [x] Verify roadmap v0.2.0 items are still relevant
  - **Updates made:**
    - Updated provider count from "15 CLI providers" to "15 providers (plus 2 generic compatible templates)"
    - Added Cerebras provider completion entry (llama-4-scout-17b-16e-instruct default)
    - Added OpenAI-compatible and Anthropic-compatible template completion entries
    - Enhanced documentation references to mention 6-pillar component map and device detection patterns

- [x] Add new documentation file: llms.txt/CONTRIBUTING.md
  - [x] Document how to add a new provider
  - [x] Document the 6-pillar requirement for templates
  - [x] Include testing guidelines

- [x] Add new documentation file: llms.txt/CHANGELOG.md
  - [x] Document v0.1.0 release features
  - [x] List all 17 providers
  - [x] Document 6-pillar architecture

- [x] Review and update llms.txt folder structure
  - [x] Ensure cross-references between files are correct
  - [x] Verify all file headers are consistent
  - [x] Add index/README if needed
  - **Changes made:**
    - Created `llms.txt/README.md` as documentation index with navigation
    - Added cross-reference links to ARCHITECTURE.md and CHANGELOG.md
    - Verified all file headers use consistent `# Title — VL-RAG-Graph-RLM` format
    - Cross-references between files are working (PRD → TODO/RULES/ARCHITECTURE, CONTRIBUTING → all docs)

## Notes
- The llms.txt folder now includes a README.md index for navigation
- See llms.txt/README.md for the documentation overview and quick navigation
- All provider templates should demonstrate all 6 pillars
- Version is 0.1.0 — ensure consistency across all docs
