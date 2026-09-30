"""Cache expensive dataset indexing.

Indexing is a one-off scan that produces a list of samples, and it is slow on the
read-only network mounts these datasets usually live on: measured on Kaggle,
43,552 KITTI odometry pairs took 2756 s to enumerate and a 50,400-pair HDF5
container 1062 s. The notebook then builds each dataset more than once -- the
diagnostic cell, the preview cell and the trainer each construct their own -- so
that cost was being paid two or three times per run, and again on every re-run.

The result depends only on the dataset root and the options that select samples
from it, so it is memoised in the process and written to disk. Dataset mounts are
read-only and immutable, which is what makes this safe; a cache entry also
records the root's own mtime and is discarded if that changes.

Set ``STEREO_INDEX_CACHE=off`` to disable, or to a path to choose the location.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Callable, Dict, Optional

_MEMORY: Dict[str, Any] = {}


def cache_dir() -> Optional[str]:
    """Where indices are stored, or ``None`` when caching is off."""
    setting = os.environ.get("STEREO_INDEX_CACHE", "").strip()
    if setting.lower() in ("off", "0", "false", "no"):
        return None
    if setting:
        return setting
    # Kaggle keeps /kaggle/working between sessions, so an index survives a
    # re-run there; elsewhere the temp directory is enough to fix the repeated
    # builds within one run.
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/.stereo_index_cache"
    return os.path.join(tempfile.gettempdir(), "stereo-index-cache")


def _key(root: str, kind: str, options: Dict[str, Any]) -> str:
    try:
        stamp = str(os.stat(root).st_mtime_ns)
    except OSError:
        stamp = "missing"
    payload = json.dumps({"root": os.path.abspath(root), "kind": kind,
                          "options": options, "mtime": stamp}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


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
    path = os.path.join(directory, f"{kind}-{key}.json") if directory else None
    if path and os.path.isfile(path):
        try:
            with open(path) as handle:
                value = json.load(handle)
            _MEMORY[key] = value
            if not quiet:
                print(f"  index: reusing {len(value) if hasattr(value, '__len__') else 1} "
                      f"cached entries for {kind}")
            return value
        except (OSError, ValueError):
            pass                                  # a corrupt entry is just rebuilt

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
