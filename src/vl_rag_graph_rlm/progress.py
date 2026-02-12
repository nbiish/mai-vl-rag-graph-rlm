"""Progress bar utilities for VL-RAG-Graph-RLM.

Provides tqdm-based progress bars for long-running operations
with graceful fallback if tqdm is not installed.
"""

from typing import Iterator, Optional, TypeVar, Any
from contextlib import contextmanager

T = TypeVar('T')

# Try to import tqdm, fallback to simple iterator if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def get_progress_bar(
    iterable: Iterator[T],
    desc: str = "",
    total: Optional[int] = None,
    unit: str = "it",
    disable: bool = False,
) -> Iterator[T]:
    """Get a progress bar iterator (tqdm if available, else plain).
    
    Args:
        iterable: The iterable to wrap
        desc: Description to show next to progress bar
        total: Total number of items (for calculating percentage)
        unit: Unit name (default: "it" for iterations)
        disable: If True, disable the progress bar entirely
        
    Returns:
        Iterator that yields from iterable with progress display
    """
    if disable or not HAS_TQDM:
        return iterable
    
    return tqdm(iterable, desc=desc, total=total, unit=unit)


@contextmanager
def progress_context(
    desc: str = "Processing",
    total: Optional[int] = None,
    unit: str = "it",
    disable: bool = False,
):
    """Context manager for progress bar operations.
    
    Usage:
        with progress_context("Embedding", total=len(chunks)) as pbar:
            for chunk in chunks:
                process(chunk)
                pbar.update(1)
    """
    if disable or not HAS_TQDM:
        # Dummy progress bar that does nothing
        class DummyPbar:
            def update(self, n: int = 1) -> None:
                pass
            def set_postfix(self, **kwargs: Any) -> None:
                pass
            def close(self) -> None:
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        
        yield DummyPbar()
    else:
        from tqdm import tqdm
        pbar = tqdm(desc=desc, total=total, unit=unit)
        try:
            yield pbar
        finally:
            pbar.close()


def tqdm_available() -> bool:
    """Check if tqdm is available for progress bars."""
    return HAS_TQDM
