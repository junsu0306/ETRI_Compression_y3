"""Small file helpers shared by every stage (standard library only)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data) -> None:
    """Write JSON atomically so an interrupted run never leaves a partial file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def claim_lock(stream) -> bool:
    """Take the exclusive lock on an open file and write this PID into it; False if it is taken."""
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    stream.seek(0)
    stream.truncate()
    stream.write(f"{os.getpid()}\n")
    stream.flush()
    return True


def lock_owner(path) -> int | None:
    """PID of the live process that holds the lock on this file, or None if nobody does."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            text = stream.read().strip()
            if not text.isdigit() or int(text) <= 1:
                raise RuntimeError(f"{path} is locked but does not name its owner")
            return int(text)
    return None
