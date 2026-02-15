#!/usr/bin/env python3
"""Legacy wrapper for timing checks.

Use tests/full_matrix_benchmark.py as the canonical orchestrator.
This script delegates to final confirmation phase for backward compatibility.
"""

import sys


def main() -> None:
    print("[legacy] timing_test.py delegates to full_matrix_benchmark.py --phase final_confirmation")
    from full_matrix_benchmark import main as full_matrix_main

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], "--phase", "final_confirmation"]
        full_matrix_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
