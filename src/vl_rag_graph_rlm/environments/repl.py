"""Safe REPL executor using RestrictedPython."""

import io
import sys
import re as re_module
from typing import Dict, Any, Optional

from RestrictedPython import compile_restricted_exec, safe_globals, limited_builtins, utility_builtins
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
from RestrictedPython.PrintCollector import PrintCollector


class REPLError(Exception):
    """Error during REPL execution."""
    pass


class REPLExecutor:
    """Safe Python code executor using RestrictedPython."""

    def __init__(self, timeout: int = 30, max_output_chars: int = 10000):
        """
        Initialize REPL executor.

        Args:
            timeout: Execution timeout in seconds
            max_output_chars: Maximum characters to return
        """
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def execute(self, code: str, env: Dict[str, Any]) -> str:
        """
        Execute Python code in restricted environment.

        Args:
            code: Python code to execute
            env: Environment with context, query, recursive_llm, etc.

        Returns:
            String result of execution
        """
        # Extract code from markdown blocks if present
        code = self._extract_code(code)

        if not code.strip():
            return "No code to execute"

        # Build restricted globals
        restricted_globals = self._build_globals(env)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            # Compile with RestrictedPython
            byte_code = compile_restricted_exec(code)

            if byte_code.errors:
                raise REPLError(f"Compilation error: {', '.join(byte_code.errors)}")

            # Execute
            exec(byte_code.code, restricted_globals, env)

            # Get output
            output = captured_output.getvalue()

            # Check for PrintCollector output
            if '_print' in env and callable(env['_print']):
                print_collector = env['_print']
                if hasattr(print_collector, 'txt'):
                    output += ''.join(print_collector.txt)

            # Try to get value of last expression
            lines = code.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                if last_line and not any(kw in last_line for kw in ['=', 'import', 'def', 'class', 'if', 'for', 'while', 'with', 'try', 'except']):
                    try:
                        result = eval(last_line, restricted_globals, env)
                        if result is not None:
                            output += str(result) + '\n'
                    except:
                        pass

            if not output:
                return "Code executed successfully (no output)"

            # Truncate if too long
            if len(output) > self.max_output_chars:
                truncated = output[:self.max_output_chars]
                return f"{truncated}\n\n[Output truncated: {len(output)} chars total]"

            return output.strip()

        except Exception as e:
            raise REPLError(f"Execution error: {str(e)}")

        finally:
            sys.stdout = old_stdout

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        # Check for python code blocks
        if '```python' in text:
            start = text.find('```python') + len('```python')
            end = text.find('```', start)
            if end != -1:
                return text[start:end].strip()

        # Check for generic code blocks
        if '```' in text:
            start = text.find('```') + 3
            end = text.find('```', start)
            if end != -1:
                return text[start:end].strip()

        return text

    def _build_globals(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """Build restricted globals for safe execution."""
        restricted_globals = safe_globals.copy()
        restricted_globals.update(limited_builtins)
        restricted_globals.update(utility_builtins)

        # Add guards
        restricted_globals['_iter_unpack_sequence_'] = guarded_iter_unpack_sequence
        restricted_globals['_getattr_'] = safer_getattr
        restricted_globals['_getitem_'] = lambda obj, index: obj[index]
        restricted_globals['_getiter_'] = iter
        restricted_globals['_print_'] = PrintCollector

        # Safe builtins
        restricted_globals.update({
            'len': len, 'str': str, 'int': int, 'float': float,
            'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
            'set': set, 'frozenset': frozenset, 'bytes': bytes, 'bytearray': bytearray,
            'range': range, 'enumerate': enumerate, 'zip': zip, 'map': map,
            'filter': filter, 'reversed': reversed, 'iter': iter, 'next': next,
            'sorted': sorted, 'sum': sum, 'min': min, 'max': max, 'any': any, 'all': all,
            'abs': abs, 'round': round, 'pow': pow, 'divmod': divmod,
            'chr': chr, 'ord': ord, 'hex': hex, 'oct': oct, 'bin': bin,
            'repr': repr, 'ascii': ascii, 'format': format,
            'isinstance': isinstance, 'issubclass': issubclass,
            'callable': callable, 'type': type, 'hasattr': hasattr,
            'True': True, 'False': False, 'None': None,
        })

        # Safe standard library modules
        import json
        import math
        from datetime import datetime, timedelta
        from collections import Counter, defaultdict, namedtuple

        restricted_globals.update({
            're': re_module,
            'json': json,
            'math': math,
            'datetime': datetime,
            'timedelta': timedelta,
            'Counter': Counter,
            'defaultdict': defaultdict,
            'namedtuple': namedtuple,
        })

        return restricted_globals
