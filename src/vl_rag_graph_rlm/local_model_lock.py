"""Cross-process mutex for local model inference — one model in RAM at a time.

Ensures that across ALL processes (CLI sessions, MCP servers, IDE agents,
terminal tabs, etc.) only **one** local model is loaded into RAM at any
given moment.  API-based providers are explicitly excluded from locking.

Mechanism
---------
- A single lock file at ``~/.vrlmrag/local_model.lock`` acts as the
  system-wide mutex.
- The lock file contains JSON with the holder's PID, process name,
  model identifier, timestamp, and a human-readable description.
- Acquisition uses ``fcntl.flock(LOCK_EX | LOCK_NB)`` for atomic,
  OS-enforced advisory locking.  If the lock is held, the caller
  **blocks** (with configurable timeout) until the holder releases.
- On release the lock file is truncated (not deleted) so the path
  remains stable.
- Stale-lock detection: before blocking, the module checks whether the
  PID recorded in the lock file is still alive.  If the process is
  gone (crash, kill -9, etc.) the stale lock is forcibly broken.

Thread safety
-------------
Within a single process, a ``threading.Lock`` serialises access so
multiple threads (e.g. async MCP tool handlers) do not race.

Usage
-----
::

    from vl_rag_graph_rlm.local_model_lock import local_model_lock

    with local_model_lock("Qwen/Qwen3-VL-Embedding-2B", timeout=300):
        embedder = create_qwen3vl_embedder(...)
        result = embedder.embed_text("hello")
    # lock released — model can be unloaded or another process can proceed

The context manager is **reentrant** within the same process (nested
``with`` blocks from the same PID simply increment a counter).

Security notes (AGENTS.md compliance)
--------------------------------------
- Lock file permissions: 0o600 (owner-only read/write).
- PID verification uses ``os.kill(pid, 0)`` — no signals sent.
- No secrets are stored in the lock file.
- Input validation via dataclass fields (no arbitrary deserialization).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import struct
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LOCK_DIR = Path(os.getenv("VRLMRAG_LOCK_DIR", Path.home() / ".vrlmrag"))
_LOCK_FILE = _LOCK_DIR / "local_model.lock"

# Default timeout (seconds) waiting for another process to release.
# 0 = non-blocking (raise immediately), None = wait forever.
DEFAULT_TIMEOUT: Optional[float] = 600.0  # 10 minutes

# How often to poll when waiting for the lock (seconds).
_POLL_INTERVAL: float = 0.5

# How old a lock-holder PID must be before we consider force-breaking
# (only if the PID is confirmed dead).  Prevents race on very fast restarts.
_STALE_GRACE_SECONDS: float = 2.0


# ---------------------------------------------------------------------------
# Lock metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LockInfo:
    """Metadata written into the lock file by the current holder."""

    pid: int
    model_id: str
    acquired_at: str  # ISO-8601 UTC
    process_name: str = ""
    description: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> LockInfo:
        data = json.loads(raw)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Check whether *pid* refers to a running process.

    Uses ``os.kill(pid, 0)`` which checks existence without sending a
    signal.  Returns False for PIDs that belong to zombie/dead processes.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it — still alive.
        return True


def _process_name(pid: Optional[int] = None) -> str:
    """Best-effort human-readable name for a process."""
    pid = pid or os.getpid()
    try:
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if cmdline_path.exists():
            raw = cmdline_path.read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
            return raw[:120]
    except Exception:
        pass

    # macOS fallback
    try:
        import subprocess

        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:120]
    except Exception:
        pass

    return f"pid={pid}"


# ---------------------------------------------------------------------------
# Core lock implementation
# ---------------------------------------------------------------------------

class _LocalModelLock:
    """System-wide advisory lock for local model inference.

    This is a singleton — all threads in a process share one instance.
    Cross-process serialisation is handled by ``fcntl.flock``.
    """

    _instance: Optional[_LocalModelLock] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> _LocalModelLock:
        with cls._init_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._thread_lock = threading.Lock()
                inst._fd: Optional[int] = None
                inst._reentrant_count = 0
                inst._holder_pid: Optional[int] = None
                cls._instance = inst
            return cls._instance

    def _ensure_lock_dir(self) -> None:
        """Create the lock directory with secure permissions."""
        _LOCK_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(_LOCK_DIR), 0o700)
        except OSError:
            pass

    def _read_lock_info(self) -> Optional[LockInfo]:
        """Read metadata from the lock file (best-effort)."""
        try:
            if _LOCK_FILE.exists() and _LOCK_FILE.stat().st_size > 0:
                raw = _LOCK_FILE.read_text(encoding="utf-8").strip()
                if raw:
                    return LockInfo.from_json(raw)
        except (json.JSONDecodeError, OSError, TypeError, KeyError):
            pass
        return None

    def _write_lock_info(self, info: LockInfo) -> None:
        """Write holder metadata into the lock file."""
        try:
            _LOCK_FILE.write_text(info.to_json(), encoding="utf-8")
            os.chmod(str(_LOCK_FILE), 0o600)
        except OSError as exc:
            logger.warning("Could not write lock info: %s", exc)

    def _clear_lock_info(self) -> None:
        """Truncate the lock file (keep the inode for stability)."""
        try:
            if _LOCK_FILE.exists():
                _LOCK_FILE.write_text("", encoding="utf-8")
        except OSError:
            pass

    def _break_stale_lock(self) -> bool:
        """If the recorded holder PID is dead, forcibly release the lock.

        Returns True if a stale lock was broken.
        """
        info = self._read_lock_info()
        if info is None:
            return False

        if _pid_alive(info.pid):
            return False

        # PID is dead — check grace period
        try:
            acquired = datetime.fromisoformat(info.acquired_at)
            age = (datetime.now(timezone.utc) - acquired).total_seconds()
            if age < _STALE_GRACE_SECONDS:
                return False
        except (ValueError, TypeError):
            pass

        logger.warning(
            "Breaking stale lock held by dead PID %d (model=%s, acquired=%s)",
            info.pid,
            info.model_id,
            info.acquired_at,
        )
        self._clear_lock_info()
        return True

    def acquire(
        self,
        model_id: str,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        description: str = "",
    ) -> None:
        """Acquire the system-wide local-model lock.

        Args:
            model_id: Identifier for the model being loaded (for diagnostics).
            timeout: Max seconds to wait.  None = wait forever, 0 = non-blocking.
            description: Human-readable note (e.g. "MCP query_document").

        Raises:
            TimeoutError: If the lock cannot be acquired within *timeout*.
            OSError: On unexpected filesystem errors.
        """
        self._thread_lock.acquire()

        # Reentrant: same process already holds the lock
        if self._fd is not None and self._holder_pid == os.getpid():
            self._reentrant_count += 1
            logger.debug(
                "Reentrant lock acquire (depth=%d, model=%s)",
                self._reentrant_count,
                model_id,
            )
            self._thread_lock.release()
            return

        try:
            self._ensure_lock_dir()

            # Open (or create) the lock file
            fd = os.open(
                str(_LOCK_FILE),
                os.O_RDWR | os.O_CREAT,
                0o600,
            )

            # Try non-blocking first
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                self._holder_pid = os.getpid()
                self._reentrant_count = 1
                self._write_lock_info(
                    LockInfo(
                        pid=os.getpid(),
                        model_id=model_id,
                        acquired_at=datetime.now(timezone.utc).isoformat(),
                        process_name=_process_name(),
                        description=description,
                    )
                )
                logger.info(
                    "Local model lock acquired (pid=%d, model=%s)",
                    os.getpid(),
                    model_id,
                )
                self._thread_lock.release()
                return
            except (BlockingIOError, OSError):
                # Lock is held by another process — fall through to polling
                pass

            # Check for stale lock before waiting
            if self._break_stale_lock():
                # Retry non-blocking after breaking stale lock
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._fd = fd
                    self._holder_pid = os.getpid()
                    self._reentrant_count = 1
                    self._write_lock_info(
                        LockInfo(
                            pid=os.getpid(),
                            model_id=model_id,
                            acquired_at=datetime.now(timezone.utc).isoformat(),
                            process_name=_process_name(),
                            description=description,
                        )
                    )
                    logger.info(
                        "Local model lock acquired after stale-break (pid=%d, model=%s)",
                        os.getpid(),
                        model_id,
                    )
                    self._thread_lock.release()
                    return
                except (BlockingIOError, OSError):
                    pass

            # Log who holds the lock
            holder = self._read_lock_info()
            if holder:
                logger.info(
                    "Waiting for local model lock — held by PID %d (%s, model=%s, since %s)",
                    holder.pid,
                    holder.process_name,
                    holder.model_id,
                    holder.acquired_at,
                )

            # Polling loop with timeout
            if timeout is not None and timeout <= 0:
                os.close(fd)
                self._thread_lock.release()
                raise TimeoutError(
                    f"Local model lock is held by another process "
                    f"(holder={holder}). Non-blocking acquire failed."
                )

            deadline = time.monotonic() + timeout if timeout is not None else None

            while True:
                # Check for stale locks periodically
                self._break_stale_lock()

                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Got it!
                    self._fd = fd
                    self._holder_pid = os.getpid()
                    self._reentrant_count = 1
                    self._write_lock_info(
                        LockInfo(
                            pid=os.getpid(),
                            model_id=model_id,
                            acquired_at=datetime.now(timezone.utc).isoformat(),
                            process_name=_process_name(),
                            description=description,
                        )
                    )
                    logger.info(
                        "Local model lock acquired after wait (pid=%d, model=%s)",
                        os.getpid(),
                        model_id,
                    )
                    self._thread_lock.release()
                    return
                except (BlockingIOError, OSError):
                    pass

                if deadline is not None and time.monotonic() >= deadline:
                    os.close(fd)
                    self._thread_lock.release()
                    raise TimeoutError(
                        f"Timed out ({timeout}s) waiting for local model lock. "
                        f"Holder: PID {holder.pid if holder else '?'} "
                        f"({holder.model_id if holder else '?'})"
                    )

                time.sleep(_POLL_INTERVAL)

        except (TimeoutError, OSError):
            raise
        except Exception:
            self._thread_lock.release()
            raise

    def release(self) -> None:
        """Release the system-wide local-model lock."""
        if self._fd is None:
            return

        if self._reentrant_count > 1:
            self._reentrant_count -= 1
            logger.debug("Reentrant lock release (depth=%d)", self._reentrant_count)
            return

        try:
            self._clear_lock_info()
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            logger.info("Local model lock released (pid=%d)", os.getpid())
        except OSError as exc:
            logger.warning("Error releasing lock: %s", exc)
        finally:
            self._fd = None
            self._holder_pid = None
            self._reentrant_count = 0

    @property
    def is_held(self) -> bool:
        """Whether this process currently holds the lock."""
        return self._fd is not None and self._holder_pid == os.getpid()

    def status(self) -> dict:
        """Return diagnostic info about the current lock state.

        Useful for CLI ``--lock-status`` or MCP ``lock_status`` tool.
        """
        info = self._read_lock_info()
        if info is None:
            return {
                "locked": False,
                "holder_pid": None,
                "holder_alive": None,
                "model_id": None,
                "acquired_at": None,
                "process_name": None,
                "description": None,
                "this_process_holds": self.is_held,
            }

        alive = _pid_alive(info.pid)
        return {
            "locked": alive,  # only truly locked if holder is alive
            "holder_pid": info.pid,
            "holder_alive": alive,
            "model_id": info.model_id,
            "acquired_at": info.acquired_at,
            "process_name": info.process_name,
            "description": info.description,
            "this_process_holds": self.is_held,
        }


# ---------------------------------------------------------------------------
# Module-level singleton + context manager
# ---------------------------------------------------------------------------

_lock = _LocalModelLock()


@contextmanager
def local_model_lock(
    model_id: str,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    description: str = "",
) -> Generator[LockInfo, None, None]:
    """Context manager: acquire the system-wide local model lock.

    Ensures only one local model is loaded in RAM across all processes.
    API-based providers should NOT use this lock.

    Args:
        model_id: Identifier for the model (e.g. "Qwen/Qwen3-VL-Embedding-2B").
        timeout: Max seconds to wait for the lock.
        description: Human-readable context (e.g. "CLI run_analysis").

    Yields:
        LockInfo for the current acquisition.

    Raises:
        TimeoutError: If the lock cannot be acquired within *timeout*.

    Example::

        with local_model_lock("Qwen/Qwen3-VL-Embedding-2B") as info:
            embedder = create_qwen3vl_embedder(...)
            result = embedder.embed_text("hello")
    """
    _lock.acquire(model_id, timeout=timeout, description=description)
    try:
        info = LockInfo(
            pid=os.getpid(),
            model_id=model_id,
            acquired_at=datetime.now(timezone.utc).isoformat(),
            process_name=_process_name(),
            description=description,
        )
        yield info
    finally:
        _lock.release()


def lock_status() -> dict:
    """Query the current state of the local model lock.

    Returns a dict with keys: locked, holder_pid, holder_alive,
    model_id, acquired_at, process_name, description, this_process_holds.
    """
    return _lock.status()


def is_local_provider(provider: Optional[str] = None, use_api: bool = False) -> bool:
    """Determine whether the current configuration uses local models.

    Returns True if local GPU/CPU models will be loaded (and thus the
    lock should be acquired).  Returns False for API-only configurations.

    Args:
        provider: The LLM provider name (irrelevant — LLM is always API).
        use_api: Whether API-based embeddings are enabled.

    Note:
        The LLM provider (sambanova, openrouter, etc.) is always remote
        and never needs locking.  Only local embedding/reranking models
        (Qwen3-VL, Qwen3-Embedding, FlashRank) consume significant RAM.
        FlashRank is ~34 MB and coexists fine, so we only lock for the
        heavy transformer models.
    """
    if use_api:
        return False
    return True
