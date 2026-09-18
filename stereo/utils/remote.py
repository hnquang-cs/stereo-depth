"""Fetching a previous run's checkpoints so training can continue.

Kaggle sessions are time-limited, so a long run is done in stages: each session
writes an archive of its checkpoints, which is uploaded somewhere durable and
pulled back at the start of the next one.

Handles Google Drive share links (the usual reason this exists), plain HTTP(S)
URLs, and local paths -- the last covering an archive attached as a Kaggle
Dataset, which needs no network at all.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from typing import Dict, List, Optional

#: Files a resumable run consists of. ``last.pt`` continues training;
#: ``best.pt`` is the best label-free checkpoint; ``history.json`` keeps the
#: curves continuous across sessions.
RUN_FILES = ("last.pt", "best.pt", "history.json")

#: The shapes a Google Drive link comes in.
DRIVE_ID_PATTERNS = (
    r"/file/d/([A-Za-z0-9_-]{10,})",        # /file/d/<id>/view
    r"[?&]id=([A-Za-z0-9_-]{10,})",         # /open?id=<id>, /uc?id=<id>
    r"/d/([A-Za-z0-9_-]{10,})",             # short forms
)


def parse_drive_file_id(url: str) -> Optional[str]:
    """Extract the file id from a Drive URL, or ``None`` if it is not one.

    A bare id is accepted as-is, which is what people often paste.
    """
    if not url:
        return None
    if "drive.google.com" not in url and "docs.google.com" not in url:
        # A bare file id: no scheme, no separators, Drive's alphabet.
        return url if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url.strip()) else None
    if "/folders/" in url:
        raise ValueError(
            "that is a Drive FOLDER link. Share the zip file itself instead "
            "(right-click the file -> Share -> Anyone with the link -> Copy link).")
    for pattern in DRIVE_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"could not find a file id in the Drive link: {url}")


def _ensure_gdown():
    """Import gdown, installing it if absent. Returns the module or None.

    gdown exists to handle Drive's large-file confirmation page, which a plain
    GET does not get past.
    """
    import importlib
    try:
        return importlib.import_module("gdown")
    except ImportError:
        pass
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"],
                       check=True, capture_output=True)
        return importlib.import_module("gdown")
    except Exception:
        return None


def _looks_like_an_archive(path: str) -> bool:
    """Drive serves an HTML page for permission or quota errors; catch that here
    rather than letting it surface as a baffling 'not a zip file'."""
    with open(path, "rb") as handle:
        magic = handle.read(4)
    return magic[:2] == b"PK" or magic[:2] == b"\x1f\x8b" or magic[:1] == b"B"


def _check_is_archive(path: str) -> None:
    """Fail early and legibly on something that is not an archive.

    Drive serves an HTML permission or quota page instead of the file when a
    link is not shared publicly; without this the failure surfaces much later as
    a baffling "not a zip file".
    """
    if _looks_like_an_archive(path):
        return
    with open(path, "rb") as handle:
        head = handle.read(200).decode("utf-8", "replace")
    raise RuntimeError(
        f"this is not an archive. Its first bytes are:\n  {head!r}\n"
        "From a Drive link that is usually the permission or quota page rather than "
        "your file -- check it is shared with 'Anyone with the link'.")


def download(source: str, destination: str) -> str:
    """Fetch ``source`` to ``destination``. Returns the local path.

    ``source`` may be a local path (copied), a Drive link, or any HTTP(S) URL.
    """
    if os.path.exists(source):
        if os.path.abspath(source) != os.path.abspath(destination):
            shutil.copy(source, destination)
        _check_is_archive(destination)
        return destination

    os.makedirs(os.path.dirname(os.path.abspath(destination)) or ".", exist_ok=True)
    file_id = parse_drive_file_id(source)

    if file_id:
        gdown = _ensure_gdown()
        if gdown is None:
            raise RuntimeError(
                "downloading from Google Drive needs gdown, which could not be installed. "
                "Either 'pip install gdown', or attach the archive as a Kaggle Dataset and "
                "give its path instead of a link.")
        result = gdown.download(id=file_id, output=destination, quiet=False)
        if result is None or not os.path.exists(destination):
            raise RuntimeError(
                f"Drive refused the download of file id {file_id}.\n"
                "Most often the file is not shared publicly: open it in Drive -> Share -> "
                "General access -> 'Anyone with the link'. Drive also rate-limits files "
                "that have been downloaded heavily.")
    else:
        if not source.lower().startswith(("http://", "https://")):
            raise FileNotFoundError(f"not a local path, a Drive link or an HTTP URL: {source}")
        command = (["curl", "-L", "--fail", "-o", destination, source] if shutil.which("curl")
                   else ["wget", "-O", destination, source])
        subprocess.run(command, check=True)

    _check_is_archive(destination)
    return destination


def extract(archive_path: str, directory: str) -> List[str]:
    """Extract a zip or tar into ``directory``; returns the member names."""
    os.makedirs(directory, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(directory)
            return archive.namelist()
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(directory)
            return archive.getnames()
    raise RuntimeError(f"{archive_path} is neither a zip nor a tar archive")


def restore_run(source: str, output_dir: str, workspace: Optional[str] = None) -> Dict[str, str]:
    """Fetch and unpack a previous run's checkpoints into ``output_dir``.

    The archive's internal layout does not matter: the run files are located
    wherever they sit inside it and copied to the top of ``output_dir``, which is
    where the trainer looks.

    Returns ``{filename: path}`` for each run file found.
    """
    os.makedirs(output_dir, exist_ok=True)
    workspace = workspace or os.path.join(output_dir, "_restore")
    if os.path.isdir(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace)

    archive_path = os.path.join(workspace, "checkpoints_archive")
    download(source, archive_path)
    extract(archive_path, workspace)

    found: Dict[str, str] = {}
    for directory, _, filenames in os.walk(workspace):
        for filename in filenames:
            if filename in RUN_FILES and filename not in found:
                source_path = os.path.join(directory, filename)
                target = os.path.join(output_dir, filename)
                if os.path.abspath(source_path) != os.path.abspath(target):
                    shutil.copy(source_path, target)
                found[filename] = target

    if not found:
        raise FileNotFoundError(
            f"the archive contains none of {RUN_FILES}.\n"
            f"What it does contain:\n  "
            + "\n  ".join(sorted(os.path.relpath(os.path.join(d, f), workspace)
                                 for d, _, fs in os.walk(workspace) for f in fs)[:25]))

    shutil.rmtree(workspace, ignore_errors=True)
    return found


def package_run(output_dir: str, archive_path: Optional[str] = None) -> Optional[str]:
    """Zip this run's checkpoints, ready to upload for the next session.

    Returns the archive path, or ``None`` if there is nothing to package.
    """
    present = [name for name in RUN_FILES if os.path.isfile(os.path.join(output_dir, name))]
    if not present:
        return None
    archive_path = archive_path or os.path.join(output_dir, "checkpoints.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in present:
            archive.write(os.path.join(output_dir, name), arcname=name)
    return archive_path
