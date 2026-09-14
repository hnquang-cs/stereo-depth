"""Scene Flow FlyingThings3D, with layout auto-discovery.

Scene Flow is 132 GB officially, so on Kaggle it is normally *attached* as a
community mirror rather than downloaded.  Those mirrors do not agree on a
directory layout, so this module discovers the layout instead of assuming one.

Two layouts are recognised.

**official** -- the structure of the upstream archives::

    <root>/frames_finalpass/TRAIN|TEST/<A|B|C>/<scene>/left|right/*.png
    <root>/disparity/       TRAIN|TEST/<A|B|C>/<scene>/left|right/*.pfm

**subset** -- the smaller ``FlyingThings3D_subset`` release, which is flat
(no ``A/B/C`` scene directories) and uses lowercase split names::

    <root>/train|val/image_clean|image_final/left|right/*.png
    <root>/train|val/disparity/               left|right/*.pfm

In both cases the relevant root may be buried some directories deep inside an
attached Kaggle dataset (mirrors commonly nest a folder of the same name, or
wrap everything in ``FlyingThings3D/``), so :func:`discover_sceneflow` searches
downward for it rather than requiring an exact path.

Disparity sign: the official PFM holds positive left-referenced disparity, which
is this repository's convention, so it is read as-is.  (The reference
implementation negates it because it decodes PFM through OpenCV, which applies
the signed scale factor from the header; :func:`stereo.data.io.read_pfm` uses
the sign only to pick the byte order, as the format specifies.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from .base import DatasetMode, StereoDataset
from .io import read_image, read_middlebury_disparity

#: Scene Flow renders with a virtual camera of focal length 1050 px (35 mm lens)
#: and a 1 m baseline.  Used only for optional depth conversion.
SCENEFLOW_FOCAL_LENGTH = 1050.0
SCENEFLOW_BASELINE = 1.0

#: Image-pass directory names, most preferred first.
OFFICIAL_PASS_DIRS = ("frames_finalpass", "frames_cleanpass")
SUBSET_PASS_DIRS = ("image_clean", "image_final")

#: How a requested split maps onto each layout's own naming.
SPLIT_ALIASES = {
    "TRAIN": {"official": ("TRAIN",), "subset": ("train",)},
    "TEST": {"official": ("TEST",), "subset": ("val", "test")},
}

#: How deep to search inside an attached dataset before giving up.
MAX_SEARCH_DEPTH = 5


@dataclass
class SceneFlowLayout:
    """Where the images and (optionally) the disparity actually live."""
    kind: str                       # "official" or "subset"
    root: str                       # the directory the layout is rooted at
    frames_dir: str                 # directory holding the split's image tree
    disparity_dir: Optional[str]    # None when the mirror ships images only
    pass_name: str
    split: str                      # the split directory name actually used

    def describe(self) -> str:
        lines = [f"layout      : {self.kind}",
                 f"root        : {self.root}",
                 f"images      : {self.frames_dir}  (pass: {self.pass_name})",
                 f"disparity   : {self.disparity_dir or 'NOT PRESENT -- images only'}",
                 f"split       : {self.split}"]
        if self.disparity_dir is None:
            lines.append("              label-free training works; benchmarking needs disparity")
        return "\n".join(lines)


def _subdirs(path: str) -> List[str]:
    try:
        return sorted(entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry)))
    except OSError:
        return []


def _candidate_roots(root: str, max_depth: int = MAX_SEARCH_DEPTH):
    """``root`` first, then every directory beneath it, breadth-first."""
    frontier = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        yield current
        if depth < max_depth:
            frontier.extend((os.path.join(current, child), depth + 1) for child in _subdirs(current))


def _match_official(base: str, split: str, pass_name: Optional[str]) -> Optional[SceneFlowLayout]:
    passes = (pass_name if isinstance(pass_name, str) else None,) if pass_name else OFFICIAL_PASS_DIRS
    for candidate_pass in passes:
        name = candidate_pass if candidate_pass.startswith("frames_") else f"frames_{candidate_pass}"
        frames_root = os.path.join(base, name)
        if not os.path.isdir(frames_root):
            continue
        for split_name in SPLIT_ALIASES[split]["official"]:
            frames_dir = os.path.join(frames_root, split_name)
            if not os.path.isdir(frames_dir):
                continue
            disparity_dir = os.path.join(base, "disparity", split_name)
            return SceneFlowLayout(
                kind="official", root=base, frames_dir=frames_dir,
                disparity_dir=disparity_dir if os.path.isdir(disparity_dir) else None,
                pass_name=name.replace("frames_", ""), split=split_name)
    return None


def _match_subset(base: str, split: str, pass_name: Optional[str]) -> Optional[SceneFlowLayout]:
    for split_name in SPLIT_ALIASES[split]["subset"]:
        split_dir = os.path.join(base, split_name)
        if not os.path.isdir(split_dir):
            continue
        passes = SUBSET_PASS_DIRS
        if pass_name:
            preferred = "image_final" if "final" in pass_name else "image_clean"
            passes = (preferred,) + tuple(p for p in SUBSET_PASS_DIRS if p != preferred)
        for candidate_pass in passes:
            frames_dir = os.path.join(split_dir, candidate_pass)
            if not os.path.isdir(os.path.join(frames_dir, "left")):
                continue
            disparity_dir = os.path.join(split_dir, "disparity")
            return SceneFlowLayout(
                kind="subset", root=base, frames_dir=frames_dir,
                disparity_dir=disparity_dir if os.path.isdir(os.path.join(disparity_dir, "left")) else None,
                pass_name=candidate_pass, split=split_name)
    return None


def discover_sceneflow(root: str, split: str = "TRAIN",
                       pass_name: Optional[str] = None) -> SceneFlowLayout:
    """Find a usable Scene Flow layout at or beneath ``root``.

    Args:
        root: an attached Kaggle dataset directory, or a downloaded Scene Flow root.
        split: ``"TRAIN"`` or ``"TEST"``.  ``"TEST"`` also accepts the subset
            release's ``val`` directory, which is its equivalent.
        pass_name: force ``"finalpass"``/``"cleanpass"`` (official) or
            ``"image_clean"``/``"image_final"`` (subset). ``None`` auto-selects.

    Raises:
        FileNotFoundError: with a listing of what *was* found, so an unsupported
            mirror can be reported without guessing.
    """
    split = split.upper()
    if split not in SPLIT_ALIASES:
        raise ValueError(f"split must be 'TRAIN' or 'TEST', got {split!r}")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no such directory: {root}")

    for base in _candidate_roots(root):
        for matcher in (_match_official, _match_subset):
            layout = matcher(base, split, pass_name)
            if layout is not None:
                return layout

    raise FileNotFoundError(
        f"could not find a Scene Flow layout for split {split!r} under {root}.\n"
        f"Looked for either:\n"
        f"  official : <dir>/frames_finalpass|frames_cleanpass/{SPLIT_ALIASES[split]['official'][0]}/...\n"
        f"  subset   : <dir>/{SPLIT_ALIASES[split]['subset'][0]}/image_clean|image_final/left/...\n"
        f"searching up to {MAX_SEARCH_DEPTH} directories deep.\n\n"
        f"What is actually there:\n{describe_tree(root)}")


def describe_tree(root: str, max_depth: int = 3, max_entries: int = 12) -> str:
    """A short directory tree, for diagnosing an unrecognised mirror."""
    lines = []

    def walk(path: str, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        children = _subdirs(path)
        files = []
        try:
            files = sorted(f for f in os.listdir(path) if not os.path.isdir(os.path.join(path, f)))
        except OSError:
            pass
        for child in children[:max_entries]:
            lines.append(f"{prefix}{child}/")
            walk(os.path.join(path, child), depth + 1, prefix + "  ")
        if len(children) > max_entries:
            lines.append(f"{prefix}... and {len(children) - max_entries} more directories")
        if files:
            shown = ", ".join(files[:3])
            more = f" ... and {len(files) - 3} more" if len(files) > 3 else ""
            lines.append(f"{prefix}[{len(files)} files: {shown}{more}]")

    lines.append(f"{root}/")
    walk(root, 1, "  ")
    return "\n".join(lines)


class SceneFlowEntry(NamedTuple):
    """One frame, addressed the same way in both layouts."""
    relative: str      # "" for the flat subset layout, "<letter>/<scene>" for official
    filename: str

    @property
    def sample_id(self) -> str:
        stem = os.path.splitext(self.filename)[0]
        return f"{self.relative.replace(os.sep, '_')}_{stem}" if self.relative else stem


class SceneFlowDataset(StereoDataset):
    """FlyingThings3D, official or subset layout.

    Args:
        root: Scene Flow root, or any attached directory containing one.
        split: ``"TRAIN"`` or ``"TEST"``.  The paper's Table IV reports **TEST**.
        pass_name: ``None`` auto-selects; otherwise forces a specific image pass.
        require_disparity: raise if no disparity tree is present.  Defaults to
            ``True`` only in ``BENCHMARK`` mode -- label-free training needs
            images alone, so an images-only mirror is perfectly usable for it.
    """

    def __init__(self, root: str, split: str = "TRAIN", mode: DatasetMode = DatasetMode.TRAIN,
                 transform=None, pass_name: Optional[str] = None, name: Optional[str] = None,
                 require_disparity: Optional[bool] = None):
        super().__init__(mode=mode, transform=transform, name=name or f"sceneflow_{split.lower()}")
        self.layout = discover_sceneflow(root, split, pass_name)
        self.root = self.layout.root
        self.split = self.layout.split

        if require_disparity is None:
            require_disparity = mode == DatasetMode.BENCHMARK
        if require_disparity and self.layout.disparity_dir is None:
            raise FileNotFoundError(
                f"this Scene Flow copy has images but no disparity tree, so it cannot be "
                f"benchmarked:\n{self.layout.describe()}")

        self.entries = self._index()
        if not self.entries:
            raise RuntimeError(f"no Scene Flow frames found:\n{self.layout.describe()}")

    # -- indexing ----------------------------------------------------------- #

    def _index(self) -> List[SceneFlowEntry]:
        if self.layout.kind == "subset":
            return _index_flat(self.layout.frames_dir)
        return _index_official(self.layout.frames_dir)

    def _image_path(self, entry: SceneFlowEntry, view: str) -> str:
        return os.path.join(self.layout.frames_dir, entry.relative, view, entry.filename)

    def _disparity_path(self, entry: SceneFlowEntry, view: str) -> str:
        stem = os.path.splitext(entry.filename)[0]
        return os.path.join(self.layout.disparity_dir, entry.relative, view, stem + ".pfm")

    # -- dataset hooks ------------------------------------------------------- #

    def _num_samples(self) -> int:
        return len(self.entries)

    def _load_images(self, index: int) -> Tuple:
        entry = self.entries[index]
        return read_image(self._image_path(entry, "left")), read_image(self._image_path(entry, "right"))

    def _sample_metadata(self, index: int) -> Dict[str, Any]:
        return {
            "sample_id": self.entries[index].sample_id,
            "focal_length": SCENEFLOW_FOCAL_LENGTH,
            "baseline": SCENEFLOW_BASELINE,
        }

    def _load_ground_truth(self, index: int) -> Dict[str, Any]:
        disparity, valid = read_middlebury_disparity(self._disparity_path(self.entries[index], "left"))
        return {"disparity_gt": disparity, "valid_gt_mask": valid.astype("float32")}


def _index_official(frames_dir: str) -> List[SceneFlowEntry]:
    """``<letter>/<scene>/left|right/*.png``, keeping only frames present in both views."""
    entries: List[SceneFlowEntry] = []
    for letter in _subdirs(frames_dir):
        letter_dir = os.path.join(frames_dir, letter)
        for scene in _subdirs(letter_dir):
            relative = os.path.join(letter, scene)
            entries.extend(_pair_views(os.path.join(frames_dir, relative), relative))
    return entries


def _index_flat(frames_dir: str) -> List[SceneFlowEntry]:
    """``left|right/*.png`` with no scene directories."""
    return _pair_views(frames_dir, "")


def _pair_views(directory: str, relative: str) -> List[SceneFlowEntry]:
    left_dir = os.path.join(directory, "left")
    right_dir = os.path.join(directory, "right")
    if not (os.path.isdir(left_dir) and os.path.isdir(right_dir)):
        return []
    try:
        right_names = set(os.listdir(right_dir))
        left_names = sorted(os.listdir(left_dir))
    except OSError:
        return []
    return [SceneFlowEntry(relative, name) for name in left_names
            if name.lower().endswith((".png", ".jpg", ".jpeg")) and name in right_names]
