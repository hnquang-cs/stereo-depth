"""Cache expensive dataset indexing.

Indexing is a one-off scan that produces a list of samples, and it is slow on the
read-only network mounts these datasets usually live on: measured on Kaggle,
43,552 KITTI odometry pairs took 2756 s to enumerate and a 50,400-pair HDF5
container 1062 s. The notebook then builds each dataset more than once -- the
diagnostic cell, the preview cell and the trainer each construct their own -- so
that cost was being paid two or three times per run, and again on every re-run.

The result depends only on the dataset root and the options that select samples
from it, so it is memoised in the process and written to disk. Dataset mounts are
read-only and immutable, which is what makes this safe. A directory is identified
by its path and mtime; a file -- an HDF5 container -- by its size and the bytes
at its start and end, so that its entry stays valid wherever the dataset is
mounted, and two files that merely share a name and a size do not collide.

A new Kaggle session starts with an empty ``/kaggle/working``, so the disk cache
alone helps only within a session. Entries can therefore also be read from
*seed* directories: the prepared dataset ships its FlyingThings3D container's
entry (see :mod:`stereo.data.prepare`), and the notebook points
``STEREO_INDEX_SEEDS`` at it, so no GPU session waits ten minutes to list it.

Set ``STEREO_INDEX_CACHE=off`` to disable, or to a path to choose the location.
``STEREO_INDEX_SEEDS`` takes read-only directories, separated by ``os.pathsep``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

_MEMORY: Dict[str, Any] = {}


def cache_dir() -> Optional[str]:
    """Where indices are stored, or ``None`` when caching is off."""
    setting = os.environ.get("STEREO_INDEX_CACHE", "").strip()
    if setting.lower() in ("off", "0", "false", "no"):
        return None
    if setting:
        return setting
    # Survives re-running cells in one Kaggle session; seeds carry an index
    # across sessions.
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/.stereo_index_cache"
    return os.path.join(tempfile.gettempdir(), "stereo-index-cache")


def seed_dirs() -> List[str]:
    """Read-only directories that may already hold entries."""
    setting = os.environ.get("STEREO_INDEX_SEEDS", "")
    return [path for path in setting.split(os.pathsep) if path and os.path.isdir(path)]


#: Bytes read from each end of a file to fingerprint it.
FINGERPRINT_BYTES = 64 * 1024


def fingerprint(path: str) -> str:
    """A file's identity wherever it is mounted: its size, first and last 64 KB.

    128 KB costs milliseconds to read even on a network mount, against minutes
    to index the file it identifies.
    """
    size = os.path.getsize(path)
    digest = hashlib.sha1(str(size).encode())
    with open(path, "rb") as handle:
        digest.update(handle.read(FINGERPRINT_BYTES))
        if size > FINGERPRINT_BYTES:
            handle.seek(max(size - FINGERPRINT_BYTES, FINGERPRINT_BYTES))
            digest.update(handle.read(FINGERPRINT_BYTES))
    return digest.hexdigest()


def _key(root: str, kind: str, options: Dict[str, Any]) -> str:
    try:
        if os.path.isfile(root):
            identity: Dict[str, Any] = {"file": fingerprint(root)}
        else:
            identity = {"directory": os.path.abspath(root), "mtime": os.stat(root).st_mtime_ns}
    except OSError:
        identity = {"missing": os.path.abspath(root)}
    payload = json.dumps({**identity, "kind": kind, "options": options},
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


def entry_name(root: str, kind: str, options: Dict[str, Any]) -> str:
    """The file an entry is stored under, in a cache or a seed directory."""
    return f"{kind}-{_key(root, kind, options)}.json"


def cached_index(root: str, kind: str, options: Dict[str, Any],
                 build: Callable[[], Any], quiet: bool = False) -> Any:
    """Return ``build()``'s result, from cache when one is available.

    ``build`` must return something JSON-serialisable. If it does not, the value
    is still returned and simply not cached.
    """
    key = _key(root, kind, options)
    if key in _MEMORY:
        return _MEMORY[key]

    directory = cache_dir()
    name = entry_name(root, kind, options)
    path = os.path.join(directory, name) if directory else None
    for candidate in ([path] if path else []) + [os.path.join(d, name) for d in seed_dirs()]:
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate) as handle:
                value = json.load(handle)
        except (OSError, ValueError):
            continue                              # a corrupt entry is just rebuilt
        _MEMORY[key] = value
        if not quiet:
            print(f"  index: reusing {len(value) if hasattr(value, '__len__') else 1} "
                  f"cached entries for {kind}")
        return value

    value = build()
    _MEMORY[key] = value
    if path:
        try:
            os.makedirs(directory, exist_ok=True)
            with open(path + ".tmp", "w") as handle:
                json.dump(value, handle)
            os.replace(path + ".tmp", path)       # atomic, so a killed run leaves no half file
        except (OSError, TypeError, ValueError):
            pass                                  # caching is an optimisation, never a requirement
    return value
