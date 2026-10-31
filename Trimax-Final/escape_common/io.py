"""Crash-safe file writing shared by Stream A and Stream C.

Every artifact goes to a temp file in the destination directory, is fsync'd, and
is moved into place with os.replace. A reader, or a resumed run, therefore never
sees a half-written file: a path either does not exist or holds a complete file.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path


@contextlib.contextmanager
def atomic_open(path, mode: str = "wb"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        kwargs = {} if "b" in mode else {"encoding": "utf-8"}
        with os.fdopen(fd, mode, **kwargs) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def atomic_write_bytes(path, data: bytes) -> None:
    with atomic_open(path, "wb") as f:
        f.write(data)


def atomic_write_text(path, text: str) -> None:
    with atomic_open(path, "w") as f:
        f.write(text)


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_save_npy(path, array) -> None:
    import numpy as np

    with atomic_open(path, "wb") as f:
        np.save(f, array, allow_pickle=False)


def atomic_to_parquet(df, path) -> None:
    with atomic_open(path, "wb") as f:
        df.to_parquet(f, index=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def file_lock(path):
    """Advisory exclusive lock on `{path}.lock` (POSIX flock). Serialises
    read-modify-write updates of small shared files such as runs.json."""
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def update_json(path, update_fn):
    """Locked read-modify-write of a JSON object file. `update_fn` receives the
    current dict ({} if the file does not exist) and returns the new dict."""
    path = Path(path)
    with file_lock(path):
        current = json.loads(path.read_text()) if path.exists() else {}
        new = update_fn(current)
        atomic_write_json(path, new)
        return new
