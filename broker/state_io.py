"""
state_io.py — crash-safe writes for trading state files.

Every position store was written with `path.write_text(json.dumps(...))`, which
truncates the file before the new bytes land. A crash, a kill, or a disk-full
between those two steps leaves a truncated or empty file — and the loaders all
swallow the resulting parse error and return `{}`, so the bot silently forgets
every open position and starts managing nothing.

The cron bot also overlaps with manual scripts, so two writers can interleave on
the same file. `os.replace` is atomic on POSIX, so a reader sees either the old
file or the new one, never a half-written one.
"""

import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(path: Path, data, indent: int = 2) -> None:
    """
    Serialise `data` to `path` atomically.

    Serialising before opening the target means a non-serialisable value raises
    without touching the existing file. The temp file is created in the same
    directory so `os.replace` stays on one filesystem, and fsync forces the
    bytes to disk before the rename is published.
    """
    path = Path(path)
    payload = json.dumps(data, indent=indent)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path, default=None):
    """
    Load JSON, returning `default` when the file is missing.

    A corrupt file is NOT silently treated as empty: for a position store that
    would mean "no open positions", which reads as a safe answer and is the most
    dangerous possible one. The caller is made to handle it.
    """
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())
