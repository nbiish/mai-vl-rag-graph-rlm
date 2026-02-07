"""Core RLM components.

Provides:
- REPLExecutor: Safe Python code execution with RestrictedPython
- Parser: FINAL() and FINAL_VAR() extraction from LLM responses
- Prompts: System and user prompt builders for RLM
- VLRAGGraphRLM: Main RLM class
"""

# Import from rlm_core.py (main RLM implementation)
from vl_rag_graph_rlm.rlm_core import (
    VLRAGGraphRLM,
    VLRAGGraphRLMError,
    MaxIterationsError,
    MaxDepthError,
    vlraggraphrlm_complete,
)

# Import from core/ subpackage (recursive-llm-src components)
from vl_rag_graph_rlm.core.repl import REPLExecutor, REPLError
from vl_rag_graph_rlm.core.parser import (
    extract_final,
    extract_final_var,
    is_final,
    parse_response
)
from vl_rag_graph_rlm.core.prompts import build_system_prompt, build_user_prompt

__all__ = [
    # From core.py
    "VLRAGGraphRLM",
    "VLRAGGraphRLMError",
    "MaxIterationsError",
    "MaxDepthError",
    "vlraggraphrlm_complete",
    # From core/ subpackage
    "REPLExecutor",
    "REPLError",
    "extract_final",
    "extract_final_var",
    "is_final",
    "parse_response",
    "build_system_prompt",
    "build_user_prompt",
]
