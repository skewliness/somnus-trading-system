from __future__ import annotations

from contextlib import contextmanager, nullcontext
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator


STATE_SCHEMA_VERSION = 1


def _lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    lock_path = _lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def timed_file_lock(
    path: Path,
    *,
    timeout_seconds: float = 60.0,
    poll_seconds: float = 0.25,
) -> Iterator[None]:
    lock_path = _lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    with lock_path.open("a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for lock {lock_path}")
                time.sleep(max(0.01, float(poll_seconds)))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8", lock: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    lock_ctx = file_lock(path) if lock else nullcontext()
    with lock_ctx:
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding=encoding,
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_name = handle.name
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
            _fsync_dir(path.parent)
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)


def atomic_write_json(path: Path, payload: Any, *, lock: bool = True) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2), lock=lock)


def load_json(path: Path, default: Any | None = None, *, lock: bool = True) -> Any:
    if not path.exists():
        return {} if default is None else default
    lock_ctx = file_lock(path) if lock else nullcontext()
    with lock_ctx:
        raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {} if default is None else default
    return json.loads(raw)


def load_state(path: Path) -> dict[str, Any]:
    payload = load_json(path, {"positions": [], "journal": [], "learning_journal": []})
    if not isinstance(payload, dict):
        payload = {}
    return normalize_state(payload)


def normalize_state(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.setdefault("schema_version", STATE_SCHEMA_VERSION)
    normalized.setdefault("positions", [])
    normalized.setdefault("journal", [])
    normalized.setdefault("learning_journal", list(normalized.get("journal") or []))
    return normalized


def atomic_write_state(path: Path, payload: dict[str, Any], *, lock: bool = True) -> None:
    atomic_write_json(path, normalize_state(payload), lock=lock)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return len(rows)


def append_text_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
