"""Finding stereo data inside an arbitrarily-shaped directory.

Public datasets are normally *attached* on Kaggle rather than downloaded, and
community mirrors reshape them freely: extra wrapper folders, renamed splits,
dropped levels. Matching fixed path names therefore breaks constantly, so every
loader in this package locates its data **structurally** -- by looking for the
pair of views, wherever it sits -- and reports the real directory tree when it
cannot.

Two shapes cover every dataset here:

* a directory holding two *view directories* (``left``/``right``,
  ``image_02``/``image_03``, ``image_2``/``image_3``, ``colored_0``/``colored_1``)
* a directory holding two *view files* (Middlebury's ``im0.png``/``im1.png``)
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".ppm", ".pgm", ".bmp", ".tif", ".tiff")

#: View-directory spellings, most specific first. KITTI raw nests the frames one
#: level further down, in ``image_02/data``; that is handled by
#: :func:`view_image_dir`.
VIEW_DIR_PAIRS: Sequence[Tuple[str, str]] = (
    ("left", "right"),
    ("image_02", "image_03"),   # KITTI raw / Eigen split (colour cameras)
    ("image_2", "image_3"),     # KITTI 2015
    ("colored_0", "colored_1"), # KITTI 2012
    ("cam0", "cam1"),
)

#: Directory names that hold ground truth, never input images. Skipped when
#: searching, because a disparity tree mirrors the image tree's view folders and
#: would otherwise be indexed as if it were imagery.
GROUND_TRUTH_DIR_NAMES = {"disparity", "disparities", "disp", "disp_occ", "disp_noc",
                          "disp_occ_0", "disp_noc_0", "disp_occ_1", "disp_noc_1",
                          "depth", "velodyne", "velodyne_points", "proj_depth",
                          "groundtruth", "gt"}

MAX_SEARCH_DEPTH = 8


def subdirs(path: str) -> List[str]:
    try:
        return sorted(entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry)))
    except OSError:
        return []


def image_files(path: str) -> List[str]:
    try:
        return sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTENSIONS))
    except OSError:
        return []


def has_images(path: str) -> bool:
    return bool(image_files(path))


def view_image_dir(pair_dir: str, view: str) -> str:
    """Where a view's frames actually live.

    KITTI raw puts them in ``image_02/data``; everything else puts them directly
    in the view directory.
    """
    direct = os.path.join(pair_dir, view)
    nested = os.path.join(direct, "data")
    if not has_images(direct) and has_images(nested):
        return nested
    return direct


def walk_dirs(root: str, max_depth: int = MAX_SEARCH_DEPTH,
              skip_names: Iterable[str] = ()) -> Iterable[str]:
    """Breadth-first directory walk, skipping ground-truth trees."""
    skip = {name.lower() for name in skip_names}
    frontier = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        yield current
        if depth >= max_depth:
            continue
        for child in subdirs(current):
            if child.lower() in skip:
                continue
            frontier.append((os.path.join(current, child), depth + 1))


def find_view_dir_pairs(root: str, view_pairs: Sequence[Tuple[str, str]] = VIEW_DIR_PAIRS,
                        max_depth: int = MAX_SEARCH_DEPTH,
                        skip_ground_truth: bool = True) -> List[Tuple[str, Tuple[str, str]]]:
    """Every directory under ``root`` holding a pair of view directories with images.

    Returns ``[(pair_dir, (left_view, right_view)), ...]`` sorted by path.
    """
    skip = GROUND_TRUTH_DIR_NAMES if skip_ground_truth else set()
    found = []
    for current in walk_dirs(root, max_depth, skip):
        children = set(subdirs(current))
        for left, right in view_pairs:
            if {left, right} <= children and has_images(view_image_dir(current, left)):
                found.append((current, (left, right)))
                break
    return sorted(found)


def find_view_file_pairs(root: str, left_name: str, right_name: str,
                         max_depth: int = MAX_SEARCH_DEPTH) -> List[str]:
    """Every directory under ``root`` holding both named image files (Middlebury scenes)."""
    found = []
    for current in walk_dirs(root, max_depth, GROUND_TRUTH_DIR_NAMES):
        if (os.path.isfile(os.path.join(current, left_name))
                and os.path.isfile(os.path.join(current, right_name))):
            found.append(current)
    return sorted(found)


def paired_filenames(left_dir: str, right_dir: str) -> List[str]:
    """Filenames present in both view directories; unpaired frames are skipped."""
    right = set(image_files(right_dir))
    return [name for name in image_files(left_dir) if name in right]


def common_ancestor(paths: Sequence[str]) -> Optional[str]:
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    try:
        return os.path.commonpath(list(paths))
    except ValueError:
        return None


def describe_tree(root: str, max_depth: int = 4, max_entries: int = 10) -> str:
    """A short directory tree, for diagnosing an unrecognised mirror."""
    lines = [f"{root}/"]

    def walk(path: str, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        children = subdirs(path)
        try:
            files = sorted(f for f in os.listdir(path) if not os.path.isdir(os.path.join(path, f)))
        except OSError:
            files = []
        for child in children[:max_entries]:
            lines.append(f"{prefix}{child}/")
            walk(os.path.join(path, child), depth + 1, prefix + "  ")
        if len(children) > max_entries:
            lines.append(f"{prefix}... and {len(children) - max_entries} more directories")
        if files:
            shown = ", ".join(files[:3])
            more = f" ... and {len(files) - 3} more" if len(files) > 3 else ""
            lines.append(f"{prefix}[{len(files)} files: {shown}{more}]")

    walk(root, 1, "  ")
    return "\n".join(lines)
