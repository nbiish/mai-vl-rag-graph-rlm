#!/usr/bin/env python3
"""Legacy wrapper for benchmark execution.

Use tests/full_matrix_benchmark.py as the canonical orchestrator.
This script delegates to the pre-final phase for backward compatibility.
"""
import sys


def main():
    """Delegate to canonical full-matrix pre-final benchmark."""
    print("[legacy] benchmark_modes.py delegates to full_matrix_benchmark.py --phase pre_final")
    from full_matrix_benchmark import main as full_matrix_main

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], "--phase", "pre_final"]
        full_matrix_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
