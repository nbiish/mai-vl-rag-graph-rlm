"""Utility functions for parsing LLM responses."""

import re
from typing import Optional


def find_code_blocks(text: str) -> list[str]:
    """
    Extract code blocks from LLM response.
    Looks for ```python or ``` blocks.
    """
    code_blocks = []

    # Python code blocks
    python_pattern = r'```python\s*\n(.*?)\n```'
    for match in re.finditer(python_pattern, text, re.DOTALL):
        code_blocks.append(match.group(1).strip())

    # Generic code blocks (if no python blocks found)
    if not code_blocks:
        generic_pattern = r'```\s*\n(.*?)\n```'
        for match in re.finditer(generic_pattern, text, re.DOTALL):
            code_blocks.append(match.group(1).strip())

    return code_blocks


def find_final_answer(text: str, env: Optional[dict] = None) -> Optional[str]:
    """
    Extract final answer from FINAL() or FINAL_VAR() call.

    Args:
        text: LLM response text
        env: Optional environment dict for FINAL_VAR lookup

    Returns:
        Final answer string or None if not found
    """
    # Try FINAL() patterns
    patterns = [
        r'FINAL\s*\(\s*"""(.*?)"""\s*\)',  # Triple double quotes
        r"FINAL\s*\(\s*'''(.*?)'''\s*\)",  # Triple single quotes
        r'FINAL\s*\(\s*"([^"]*)"\s*\)',  # Double quotes
        r"FINAL\s*\(\s*'([^']*)'\s*\)",  # Single quotes
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try FINAL_VAR() if env provided
    if env is not None:
        var_match = re.search(r'FINAL_VAR\s*\(\s*(\w+)\s*\)', text)
        if var_match:
            var_name = var_match.group(1)
            if var_name in env:
                return str(env[var_name])

    return None


def has_final_answer(text: str) -> bool:
    """Check if text contains FINAL() or FINAL_VAR()."""
    return 'FINAL(' in text or 'FINAL_VAR(' in text


def format_iteration(iteration: Any) -> list[dict[str, str]]:
    """
    Format an iteration for adding to message history.

    Returns:
        List of message dicts with role and content
    """
    messages = []

    # Assistant's response
    messages.append({
        "role": "assistant",
        "content": iteration.response
    })

    # Add code execution results
    for code_block in iteration.code_blocks:
        stdout = code_block.result.stdout
        stderr = code_block.result.stderr

        if stdout:
            messages.append({
                "role": "user",
                "content": f"Code output:\n{stdout}"
            })

        if stderr:
            messages.append({
                "role": "user",
                "content": f"Code error:\n{stderr}"
            })

    return messages


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vl_rag_graph_rlm.types import RLMIteration
