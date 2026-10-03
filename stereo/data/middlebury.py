"""Middlebury stereo: every release on vision.middlebury.edu, and MiddEval3.

The releases disagree on file names, on where the views sit and on how the
disparity is stored, so a loader written for one finds nothing in another --
or, worse, reads labels that are wrong by a constant factor:

    release     left / right                disparity: left, right        to pixels
    2001        im2.ppm      im6.ppm        disp2.pgm      disp6.pgm      / 8
    2003        im2.png|ppm  im6.png|ppm    disp2.png|pgm  disp6.png|pgm  x width / 1800
    2005, 2006  view1.png    view5.png      disp1.png      disp5.png      / 1, 2, 3 (full, half, third size)
    2014, 2021  im0.png      im1.png        disp0.pfm      disp1.pfm      as is
    MiddEval3   im0.png      im1.png        disp0GT.pfm    disp1GT.pfm    as is

The right view's disparity is what makes the horizontal flip possible: a
flipped pair's left view is the old right view, mirrored. It is loaded in
training only, the one mode that flips.

Each rule was confirmed on downloaded data: the right view, shifted by the
converted label, reconstructs the left better than at any other scale (see
:func:`stereo.data.check_label_scale`). The 2003 page documents only its
quarter size; the half and full sizes were measured, and all three store the
full-size disparity.

Files that look like disparity and are not:

    disp0-n.pgm, disp0-sd.pfm   2014 perfect: sample count, standard deviation
    disp0y.pfm                  2014 imperfect: the VERTICAL disparity
    orig/disp0.pfm              2021: superseded by the scene's own disp0.pfm

2005 and 2006 photograph each scene under 3 illuminations x 3 exposures. The
release's default pair is Illum1/Exp1 for 2005 and Illum1/Exp2 for 2006 (the
single-illumination archives move it up into the scene directory); the other
exposures are as much as 9x darker.

Scenes without public ground truth are skipped: the MiddEval3 test set, and
Computer, Drumsticks and Dwarves of 2005. A scene present more than once -- in
several sizes, as 2014 perfect and imperfect, or in both MiddEval3 and the
release it came from -- is used once, from the smallest copy at least
``min_width`` wide. Resized to the training width the copies are the same
image, and the small one decodes many times faster.

Depth from disparity uses the dataset's own formula, including the
principal-point offset::

    Z = baseline * f / (d + doffs)
"""

from __future__ import annotations

import os
import re
import zlib
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .base import DatasetMode, StereoDataset
from .discovery import GROUND_TRUTH_DIR_NAMES, describe_tree, walk_dirs
from .io import (image_width, read_image, read_middlebury_calib, read_middlebury_disparity,
                 read_nonocc_mask)

#: 2005 shares the 2006 layout but not its default exposure.
SCENES_2005 = {"art", "books", "computer", "dolls", "drumsticks", "dwarves",
               "laundry", "moebius", "reindeer"}
#: 2021 shares the 2014 layout. Each scene is imaged from 1-3 viewpoints: artroom1, ...
SCENES_2021 = {"artroom", "bandsaw", "chess", "curule", "ladder", "octogons",
               "pendulum", "podium", "skates", "skiboots", "traproom"}
#: 2003 comes in exactly these widths (quarter, half, full); 2001 is 430-435 wide.
WIDTHS_2003 = (450, 900, 1800)

RELEASES = ("2001", "2003", "2005", "2006", "2014", "2021", "MiddEval3")

#: View pairs, to recognise a scene that has views but no ground truth.
VIEW_PAIRS = (("im0.png", "im1.png"), ("im2.png", "im6.png"), ("im2.ppm", "im6.ppm"),
              ("view1.png", "view5.png"))


@dataclass(frozen=True)
class Scene:
    """One labelled stereo pair."""
    release: str
    left: str
    right: str
    disparity: str
    width: int
    disparity_right: Optional[str] = None

    @property
    def directory(self) -> str:
        return os.path.dirname(self.disparity)


def pixels_per_unit(release: str, width: int) -> float:
    """What one stored disparity unit is, in pixels, at this image width."""
    if release == "2001":
        return 1 / 8
    if release == "2003":
        return width / 1800
    if release in ("2005", "2006"):        # full 1240-1396 wide, half 620-698, third 413-465
        return 1.0 if width > 1000 else 1 / 2 if width > 550 else 1 / 3
    return 1.0                             # PFM is in pixels


def match_scene(directory: str, files: Sequence[str]) -> Optional[Scene]:
    """The labelled scene ``directory`` holds, given the names of its files."""
    files = set(files)
    name = os.path.basename(directory).lower()

    def scene(release: str, left: str, right: str, disparity: str, disparity_right: str) -> Scene:
        left = os.path.join(directory, left)
        return Scene(release, left, os.path.join(directory, right),
                     os.path.join(directory, disparity), image_width(left),
                     os.path.join(directory, disparity_right) if disparity_right in files else None)

    if {"im0.png", "im1.png"} <= files:
        if "disp0GT.pfm" in files:
            return scene("MiddEval3", "im0.png", "im1.png", "disp0GT.pfm", "disp1GT.pfm")
        if "disp0.pfm" in files:
            release = "2021" if name.rstrip("0123456789") in SCENES_2021 else "2014"
            return scene(release, "im0.png", "im1.png", "disp0.pfm", "disp1.pfm")

    for image, disparity in ((".png", ".png"), (".ppm", ".pgm")):
        if {"im2" + image, "im6" + image, "disp2" + disparity} <= files:
            found = scene("2001", "im2" + image, "im6" + image, "disp2" + disparity,
                          "disp6" + disparity)
            return replace(found, release="2003") if found.width in WIDTHS_2003 else found

    if "disp1.png" in files:
        release = "2005" if name in SCENES_2005 else "2006"
        for views in ("", "Illum1/Exp1" if release == "2005" else "Illum1/Exp2"):
            left, right = os.path.join(views, "view1.png"), os.path.join(views, "view5.png")
            if all(os.path.isfile(os.path.join(directory, v)) for v in (left, right)):
                return scene(release, left, right, "disp1.png", "disp5.png")
    return None


def scene_identity(scene: Scene) -> str:
    """A name shared by every copy of one scene, across sizes and releases."""
    name = os.path.basename(scene.directory).lower()
    name = re.sub(r"-(im)?perfect$", "", name)                 # 2014
    return re.sub(r"^(cones|teddy)[qhf]$", r"\1", name)        # 2003 per-size archives


#: MiddEval3 entries that re-use another scene with its right view changed:
#: exposure (E), lighting (L), or rectified perfectly (P).
MIDDEVAL3_VARIANTS = {"artl": "art", "motorcyclee": "motorcycle", "pianol": "piano",
                      "playtablep": "playtable"}


def scene_group(scene: Scene) -> str:
    """Scenes sharing geometry, which a train/validation division keeps on one side.

    Copies of a scene, MiddEval3's variants of it, and numbered siblings: 2021
    images each scene from up to 3 viewpoints (artroom1, artroom2), and the
    numbered sets of other years (Cloth1-4) share objects and materials.
    """
    name = scene_identity(scene)
    return MIDDEVAL3_VARIANTS.get(name, name).rstrip("0123456789")


def pick_copy(copies: Sequence[Scene], min_width: int = 0) -> Scene:
    """The smallest copy at least ``min_width`` wide, else the widest.

    Ties go to 2014's perfect rectification over its imperfect one.
    """
    def rank(scene: Scene) -> Tuple[bool, int, bool, str]:
        too_small = scene.width < min_width
        return (too_small, -scene.width if too_small else scene.width,
                scene.directory.lower().endswith("-imperfect"), scene.directory)

    return min(copies, key=rank)


def index_scenes(root: str, releases: Optional[Sequence[str]] = None,
                 min_width: int = 0) -> Tuple[List[Scene], List[str], int]:
    """Labelled scenes under ``root``, at any depth.

    Mirrors wrap the scenes in extra folders (``MiddEval3/trainingQ/``,
    ``ThirdSize/``, the dataset's own name), so scenes are found by their files
    rather than by assuming where they sit.

    Returns ``(scenes, unlabelled, duplicates)``: the scenes, one per identity;
    the directories that have a view pair but no ground truth; and how many
    extra copies were dropped.
    """
    wanted = set(releases or RELEASES)
    unknown = wanted - set(RELEASES)
    if unknown:
        raise ValueError(f"unknown Middlebury release(s) {sorted(unknown)}; known: {RELEASES}")

    copies: Dict[str, List[Scene]] = {}
    unlabelled = []
    for directory in walk_dirs(root, skip_names=GROUND_TRUTH_DIR_NAMES):
        try:
            with os.scandir(directory) as entries:
                files = [entry.name for entry in entries if entry.is_file()]
        except OSError:
            continue
        scene = match_scene(directory, files)
        if scene is None:
            in_exposure_dir = re.match(r"Exp\d+$", os.path.basename(directory))
            if not in_exposure_dir and any({a, b} <= set(files) for a, b in VIEW_PAIRS):
                unlabelled.append(os.path.relpath(directory, root))
        elif scene.release in wanted:
            copies.setdefault(scene_identity(scene), []).append(scene)

    scenes = sorted((pick_copy(group, min_width) for group in copies.values()),
                    key=lambda scene: scene.directory)
    duplicates = sum(len(group) - 1 for group in copies.values())
    return scenes, sorted(unlabelled), duplicates


class MiddleburyDataset(StereoDataset):
    """Middlebury scene directories, every release (and ETH3D, which shares the MiddEval3 layout)."""

    nonocc_name = "mask0nocc.png"

    def __init__(self, root: str, mode: DatasetMode = DatasetMode.TRAIN, transform=None,
                 name: Optional[str] = None, scenes: Optional[List[str]] = None,
                 releases: Optional[Sequence[str]] = None, min_width: int = 0):
        """Args:
            scenes: scene directories relative to ``root``, instead of searching.
            releases: keep only these (see :data:`RELEASES`); ``None`` keeps all.
            min_width: of several copies of a scene, use the smallest at least
                this wide. Set it to the training width.
        """
        super().__init__(mode=mode, transform=transform, name=name or "middlebury")
        self.root = root
        self.unlabelled: List[str] = []
        self.duplicates = 0
        if scenes is not None:
            self.entries = [self._explicit_scene(scene) for scene in scenes]
        else:
            self.entries, self.unlabelled, self.duplicates = index_scenes(root, releases, min_width)

        if not self.entries:
            skipped = (f"{len(self.unlabelled)} scene(s) have views but no ground truth, e.g. "
                       f"{', '.join(self.unlabelled[:3])}.\n" if self.unlabelled else "")
            raise RuntimeError(
                f"no labelled {self.name} scenes under {root}"
                + (f" for release(s) {', '.join(releases)}" if releases else "") + ".\n"
                + skipped
                + "Looked for, at any depth: im0.png + im1.png + disp0GT.pfm or disp0.pfm; "
                  "im2 + im6 + disp2 (.png or .ppm/.pgm); "
                  "view1.png + view5.png (or Illum1/Exp*/) + disp1.png.\n\n"
                  f"What is actually there:\n{describe_tree(root)}")

        self.releases: Dict[str, int] = {}
        for scene in self.entries:
            self.releases[scene.release] = self.releases.get(scene.release, 0) + 1

    def _explicit_scene(self, relative: str) -> Scene:
        directory = os.path.join(self.root, relative)
        scene = match_scene(directory, os.listdir(directory))
        if scene is None:
            raise FileNotFoundError(f"{directory} is not a labelled Middlebury scene. "
                                    f"Present: {', '.join(sorted(os.listdir(directory))[:8])}")
        return scene

    def summary(self) -> str:
        """Scenes per release, and what was left out."""
        parts = [", ".join(f"{release} {count}" for release, count in sorted(self.releases.items()))]
        if self.unlabelled:
            parts.append(f"{len(self.unlabelled)} without ground truth skipped")
        if self.duplicates:
            parts.append(f"{self.duplicates} duplicate copies skipped")
        return "  |  ".join(parts)

    def holdout_indices(self, fraction: float) -> List[int]:
        """Indices on the validation side of a division by :func:`scene_group`.

        A stable hash of the group decides its side, so adding or dropping a
        release moves no other scene across, and a resumed run validates on the
        same scenes. A set so small that no group falls under ``fraction`` holds
        out one group regardless, and at least one is always left to train on.
        """
        groups = sorted({scene_group(scene) for scene in self.entries})
        if len(groups) < 2:
            return []
        share = {group: zlib.crc32(group.encode()) / 2 ** 32 for group in groups}
        held = {group for group in groups if share[group] < fraction}
        if not held:                                   # a small set: hold out one
            held = {min(groups, key=share.get)}
        if len(held) == len(groups):                   # and train on at least one
            held.discard(max(held, key=share.get))
        return [i for i, scene in enumerate(self.entries) if scene_group(scene) in held]

    def _num_samples(self) -> int:
        return len(self.entries)

    def _load_images(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        scene = self.entries[index]
        return read_image(scene.left), read_image(scene.right)

    def _sample_metadata(self, index: int) -> Dict[str, Any]:
        scene = self.entries[index]
        metadata: Dict[str, Any] = {"sample_id": os.path.relpath(scene.directory, self.root),
                                    "release": scene.release}
        calib_path = os.path.join(scene.directory, "calib.txt")
        if os.path.exists(calib_path):
            metadata.update(read_middlebury_calib(calib_path))
        return metadata

    def _load_ground_truth(self, index: int) -> Dict[str, np.ndarray]:
        scene = self.entries[index]
        disparity, valid = read_scene_disparity(scene, scene.disparity)
        ground_truth = {"disparity_gt": disparity, "valid_gt_mask": valid.astype(np.float32)}
        if self.mode is DatasetMode.TRAIN and scene.disparity_right:
            right, right_valid = read_scene_disparity(scene, scene.disparity_right)
            ground_truth["disparity_gt_right"] = right
            ground_truth["valid_gt_mask_right"] = right_valid.astype(np.float32)

        nonocc_path = os.path.join(scene.directory, self.nonocc_name)
        if os.path.exists(nonocc_path):
            nonocc = read_nonocc_mask(nonocc_path)
            ground_truth["nonocc_mask"] = (nonocc & valid).astype(np.float32)
        return ground_truth


def read_scene_disparity(scene: Scene, path: str) -> Tuple[np.ndarray, np.ndarray]:
    """A disparity file of ``scene`` in pixels, and where it is known."""
    if path.endswith(".pfm"):
        disparity, valid = read_middlebury_disparity(path)
    else:
        stored = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if stored is None:
            raise FileNotFoundError(f"could not read disparity: {path}")
        if stored.ndim == 3:
            stored = stored[..., 0]
        valid = stored > 0                                         # 0 = unknown
        disparity = stored.astype(np.float32)
    scale = pixels_per_unit(scene.release, disparity.shape[1])
    return (disparity * scale).astype(np.float32), valid
