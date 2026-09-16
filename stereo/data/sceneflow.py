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
from .discovery import (common_ancestor, describe_missing_stereo, describe_tree,
                        find_view_dir_pairs)
from .io import read_image, read_middlebury_disparity

#: Scene Flow renders with a virtual camera of focal length 1050 px (35 mm lens)
#: and a 1 m baseline.  Used only for optional depth conversion.
SCENEFLOW_FOCAL_LENGTH = 1050.0
SCENEFLOW_BASELINE = 1.0

#: Directory holding one image pass. Official Scene Flow uses ``frames_*``; the
#: FlyingThings3D_subset release uses ``image_clean`` / ``image_final``.
SUBSET_PASS_DIRS = ("image_clean", "image_final")

#: Directory names that denote a dataset split, in any of the releases' spellings.
KNOWN_SPLIT_DIRS = {"train", "test", "val", "validation"}

#: How a requested split maps onto each release's own naming.
SPLIT_ALIASES = {
    "TRAIN": ("TRAIN", "train"),
    "TEST": ("TEST", "test", "val", "validation"),
}

#: How deep to search for an image-pass directory inside an attached dataset.
MAX_SEARCH_DEPTH = 6


@dataclass
class SceneFlowLayout:
    """Where the images and (optionally) the disparity actually live."""
    root: str                       # the directory the pass tree is rooted at
    frames_dir: str                 # directory to index for left/right pairs
    disparity_dir: Optional[str]    # None when the mirror ships images only
    pass_name: str
    split: str                      # the split directory used, or "(no split)"
    subset: str                     # FlyingThings3D / Driving / Monkaa / unknown

    #: Sentinel used when the tree carries no train/test division at all.
    NO_SPLIT = "(no split)"

    @property
    def has_split(self) -> bool:
        """False when TRAIN and TEST would return exactly the same frames."""
        return self.split != self.NO_SPLIT

    def describe(self) -> str:
        lines = [f"subset      : {self.subset}",
                 f"pass        : {self.pass_name}",
                 f"split       : {self.split}",
                 f"images      : {self.frames_dir}",
                 f"disparity   : {self.disparity_dir or 'NOT PRESENT -- images only'}"]
        if self.disparity_dir is None:
            lines.append("              label-free training works; benchmarking needs disparity")
        if not self.has_split:
            lines.append("              NO train/test division: TRAIN and TEST are the same "
                         "frames, so this copy must not be used for benchmarking")
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


def _pass_candidates(root: str, pass_name: Optional[str]):
    """Yield ``(frames_dir, disparity_dir, pass_name)`` for every image-pass tree found.

    An image-pass directory is recognised by *name* (``frames_*``, ``image_clean``,
    ``image_final``); its sibling ``disparity`` directory, if present, holds the
    matching ground truth. This single rule covers both releases, because the
    subset layout puts ``image_clean`` and ``disparity`` side by side inside its
    split directory exactly as the official one does inside the dataset root.
    """
    for base in _candidate_roots(root):
        for child in _subdirs(base):
            is_pass = child.startswith("frames_") or child in SUBSET_PASS_DIRS
            if not is_pass:
                continue
            if pass_name and pass_name.lower().replace("frames_", "") not in child.lower():
                continue
            disparity = os.path.join(base, "disparity")
            yield (os.path.join(base, child),
                   disparity if os.path.isdir(disparity) else None,
                   child)


def _fallback_candidates(root: str):
    """Last resort: a mirror with no ``frames_*`` / ``image_*`` directory at all.

    Some mirrors strip the pass level entirely and ship ``TRAIN/A/0000/left``
    directly. The image tree is then inferred from wherever ``left``/``right``
    pairs actually are -- ground-truth trees are skipped by the shared discovery
    helper, so a parallel ``disparity/`` tree is not mistaken for imagery -- and
    a sibling ``disparity`` directory is paired with it if one exists.
    """
    pair_dirs = [directory for directory, _ in
                 find_view_dir_pairs(root, [("left", "right")], max_depth=MAX_SEARCH_DEPTH)]
    if not pair_dirs:
        return

    ancestor = common_ancestor(pair_dirs)
    if ancestor is None:
        return
    # Walk up while the parent still contains only this one tree, so that a
    # TRAIN/TEST level above the scenes is included rather than cut off.
    for candidate in (ancestor, os.path.dirname(ancestor)):
        if not candidate or not os.path.isdir(candidate) or candidate == os.path.dirname(candidate):
            continue
        disparity = None
        for sibling in ("disparity", "disparities", "disp"):
            path = os.path.join(os.path.dirname(candidate), sibling)
            if os.path.isdir(path):
                disparity = path
                break
        yield candidate, disparity, "(no pass directory)"


def _resolve_split(frames_dir: str, disparity_dir: Optional[str], split: str):
    """Descend into a split directory if the tree has one.

    Returns ``(frames_dir, disparity_dir, split_name)``, or ``None`` if this tree
    belongs to a different split than the one requested.

    Three shapes occur in the wild:
      * split *inside* the pass tree  -- ``frames_finalpass/TRAIN/...`` (official)
      * split *above* the pass tree   -- ``train/image_clean/...``      (subset)
      * no split at all               -- ``frames_cleanpass/<scene>/...``
        (Monkaa and Driving have no official split, and some mirrors flatten
        FlyingThings3D the same way)
    """
    wanted = SPLIT_ALIASES[split]
    children = _subdirs(frames_dir)

    if any(child.lower() in KNOWN_SPLIT_DIRS for child in children):
        for name in children:
            if name in wanted:
                return (os.path.join(frames_dir, name),
                        os.path.join(disparity_dir, name) if disparity_dir else None,
                        name)
        return None   # the tree is split, but not into the split we were asked for

    parent = os.path.basename(os.path.dirname(frames_dir))
    if parent.lower() in KNOWN_SPLIT_DIRS:
        if parent in wanted:
            return frames_dir, disparity_dir, parent
        return None   # this is a different split's tree

    return frames_dir, disparity_dir, "(no split)"


#: Scene Flow's three subsets, matched on directory names.
SUBSET_NAMES = {"flyingthings3d": "FlyingThings3D", "driving": "Driving", "monkaa": "Monkaa"}


def _subset_of(frames_dir: str, search_root: str) -> str:
    """Identify the subset from directory names *below the search root*.

    Only the portion of the path inside the dataset matters: matching the whole
    absolute path would let the user's own directory names (a working directory
    called ``driving-project``, say) decide which subset is selected.
    """
    try:
        relative = os.path.relpath(frames_dir, search_root)
    except ValueError:
        relative = frames_dir
    for component in relative.lower().split(os.sep):
        if component in SUBSET_NAMES:
            return SUBSET_NAMES[component]
    return "unknown"


def _rank(layout: "SceneFlowLayout") -> tuple:
    """Prefer FlyingThings3D (the paper's benchmark), then the final pass, then shallower.

    Matched on ``pass_name`` rather than the full path, for the same reason as
    :func:`_subset_of`: a containing directory whose name happens to include
    "final" must not change which image pass is chosen.
    """
    pass_lower = layout.pass_name.lower()
    return (0 if layout.subset == "FlyingThings3D" else 1,
            0 if ("finalpass" in pass_lower or pass_lower == "image_final") else 1,
            layout.frames_dir.count(os.sep))


def discover_sceneflow(root: str, split: str = "TRAIN", pass_name: Optional[str] = None,
                       subset: Optional[str] = None) -> SceneFlowLayout:
    """Find a usable Scene Flow layout at or beneath ``root``.

    Discovery is structural rather than path-name based: it locates an image-pass
    directory, works out how (or whether) that tree is split, and then indexes
    every directory beneath it that holds a ``left``/``right`` pair -- at any
    depth. That covers official FlyingThings3D (``TRAIN/<A|B|C>/<scene>/left``),
    Monkaa (``<scene>/left``), Driving
    (``<focallength>/<direction>/<speed>/left``), the flat subset release, and
    mirrors that nest or flatten any of them.

    Args:
        root: an attached Kaggle dataset directory, or a downloaded Scene Flow root.
        split: ``"TRAIN"`` or ``"TEST"``. Trees with no split are accepted for either.
        pass_name: force e.g. ``"finalpass"``/``"cleanpass"``. ``None`` auto-selects.
        subset: restrict to ``"FlyingThings3D"``, ``"Driving"`` or ``"Monkaa"``.
            ``None`` prefers FlyingThings3D but accepts whatever is present.

    Raises:
        FileNotFoundError: with the directory tree that *was* found, so an
            unsupported mirror can be reported rather than guessed at.
    """
    split = split.upper()
    if split not in SPLIT_ALIASES:
        raise ValueError(f"split must be 'TRAIN' or 'TEST', got {split!r}")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no such directory: {root}")

    layouts = []
    candidates = list(_pass_candidates(root, pass_name))
    if not candidates and not pass_name:
        candidates = list(_fallback_candidates(root))
    for frames_dir, disparity_dir, found_pass in candidates:
        found_subset = _subset_of(frames_dir, root)
        if subset and found_subset.lower() != subset.lower():
            continue
        resolved = _resolve_split(frames_dir, disparity_dir, split)
        if resolved is None:
            continue
        split_frames, split_disparity, split_name = resolved
        if not _has_stereo_pair(split_frames):
            continue
        layouts.append(SceneFlowLayout(
            root=os.path.dirname(frames_dir), frames_dir=split_frames,
            disparity_dir=split_disparity, pass_name=found_pass,
            split=split_name, subset=found_subset))

    if layouts:
        return sorted(layouts, key=_rank)[0]

    explanation = describe_missing_stereo(root)
    raise FileNotFoundError(
        (explanation + "\n\n" if explanation else "")
        + f"could not find a Scene Flow layout for split {split!r} under {root}.\n"
        f"Looked for a directory named frames_* / image_clean / image_final containing "
        f"left+right image folders at any depth"
        f"{f' (restricted to subset {subset!r})' if subset else ''}"
        f"{f' (restricted to pass {pass_name!r})' if pass_name else ''}, "
        f"searching up to {MAX_SEARCH_DEPTH} directories deep.\n\n"
        f"What is actually there:\n{describe_tree(root)}")


def _has_stereo_pair(frames_dir: str, max_probe_depth: int = 6) -> bool:
    """True if some directory beneath ``frames_dir`` holds both ``left`` and ``right``."""
    for current in _candidate_roots(frames_dir, max_probe_depth):
        children = set(_subdirs(current))
        if {"left", "right"} <= children:
            return True
    return False


def find_scene_dirs(frames_dir: str) -> List[str]:
    """Relative paths of every directory under ``frames_dir`` holding a left/right pair.

    ``""`` means the pair is directly in ``frames_dir`` (the flat subset layout).
    """
    scenes: List[str] = []
    for current, dirnames, _ in os.walk(frames_dir):
        if "left" in dirnames and "right" in dirnames:
            relative = os.path.relpath(current, frames_dir)
            scenes.append("" if relative == "." else relative)
            # Do not descend into the view directories themselves.
            dirnames[:] = [d for d in dirnames if d not in ("left", "right")]
        dirnames.sort()
    return sorted(scenes)


class SceneFlowEntry(NamedTuple):
    """One frame. ``relative`` is the scene directory under the pass root."""
    relative: str
    filename: str

    @property
    def sample_id(self) -> str:
        stem = os.path.splitext(self.filename)[0]
        scene = self.relative.replace(os.sep, "_")
        return f"{scene}_{stem}" if scene else stem


class SceneFlowDataset(StereoDataset):
    """Scene Flow, any release and any mirror layout.

    Args:
        root: a Scene Flow root, or any attached directory containing one.
        split: ``"TRAIN"`` or ``"TEST"``. Trees with no split (Monkaa, Driving,
            and mirrors that flatten FlyingThings3D) are accepted for either.
        pass_name: ``None`` auto-selects; otherwise forces a specific image pass.
        subset: restrict to ``"FlyingThings3D"``, ``"Driving"`` or ``"Monkaa"``.
        require_disparity: raise if no disparity tree is present. Defaults to
            ``True`` only in ``BENCHMARK`` mode -- label-free training needs
            images alone, so an images-only mirror is fine for it.
        allow_unsplit_benchmark: permit ``BENCHMARK`` mode on a tree with no
            train/test division. Off by default, because on such a copy the TEST
            split *is* the training data and any metric from it is contaminated.
    """

    def __init__(self, root: str, split: str = "TRAIN", mode: DatasetMode = DatasetMode.TRAIN,
                 transform=None, pass_name: Optional[str] = None, name: Optional[str] = None,
                 subset: Optional[str] = None, require_disparity: Optional[bool] = None,
                 allow_unsplit_benchmark: bool = False):
        super().__init__(mode=mode, transform=transform, name=name or f"sceneflow_{split.lower()}")
        self.layout = discover_sceneflow(root, split, pass_name, subset)
        self.root = self.layout.root
        self.split = self.layout.split

        if require_disparity is None:
            require_disparity = mode == DatasetMode.BENCHMARK
        if require_disparity and self.layout.disparity_dir is None:
            raise FileNotFoundError(
                f"this Scene Flow copy has images but no disparity tree, so it cannot be "
                f"benchmarked:\n{self.layout.describe()}")

        if mode is DatasetMode.BENCHMARK and not self.layout.has_split \
                and not allow_unsplit_benchmark:
            raise RuntimeError(
                "this Scene Flow copy has no train/test division, so the TEST split would "
                "be the very frames the model trained on. Benchmarking it would report a "
                "number contaminated by training data, which is not a benchmark.\n"
                f"{self.layout.describe()}\n\n"
                "Options:\n"
                "  * benchmark on a dataset that does have a held-out split "
                "(middlebury2014, eth3d, kitti2015), or\n"
                "  * use a Scene Flow mirror that keeps the official TRAIN/TEST directories, or\n"
                "  * exclude these frames from training and pass "
                "allow_unsplit_benchmark=True, having verified the model never saw them.")

        self.entries = self._index()
        if not self.entries:
            raise RuntimeError(f"no Scene Flow frames found:\n{self.layout.describe()}")

    # -- indexing ----------------------------------------------------------- #

    def _index(self) -> List[SceneFlowEntry]:
        entries: List[SceneFlowEntry] = []
        for scene in find_scene_dirs(self.layout.frames_dir):
            entries.extend(_pair_views(os.path.join(self.layout.frames_dir, scene), scene))
        return entries

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
        path = self._disparity_path(self.entries[index], "left")
        disparity, valid = read_middlebury_disparity(path)
        return {"disparity_gt": disparity, "valid_gt_mask": valid.astype("float32")}


def _pair_views(directory: str, relative: str) -> List[SceneFlowEntry]:
    """Frames present in *both* views; anything unpaired is skipped, not fatal."""
    left_dir = os.path.join(directory, "left")
    right_dir = os.path.join(directory, "right")
    try:
        right_names = set(os.listdir(right_dir))
        left_names = sorted(os.listdir(left_dir))
    except OSError:
        return []
    return [SceneFlowEntry(relative, name) for name in left_names
            if name.lower().endswith((".png", ".jpg", ".jpeg")) and name in right_names]
