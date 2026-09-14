"""File readers for images and benchmark ground truth.

Ground-truth readers live here but are only ever called from a dataset in
``BENCHMARK`` mode -- see :mod:`stereo.data.base`.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Tuple

import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """Read an 8-bit colour image as float32 RGB in ``[0, 1]``, shape ``(H, W, 3)``.

    Note on channel order: the reference implementation feeds OpenCV's BGR
    straight into the network.  Channel order is a convention the network learns
    either way; RGB is used here because every other tool in the pipeline
    (Pillow, matplotlib) assumes it, and mixing the two silently is a classic
    source of bugs.  A checkpoint trained here therefore expects RGB.
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def read_pfm(path: str) -> np.ndarray:
    """Read a PFM file as float32, top-to-bottom, shape ``(H, W)`` or ``(H, W, 3)``.

    PFM stores rows bottom-to-top, so the array is flipped vertically on load.
    The sign of the scale line encodes endianness only; its magnitude is the
    scale factor, which for every disparity file used here is 1.
    """
    with open(path, "rb") as handle:
        header = handle.readline().rstrip()
        if header == b"PF":
            channels = 3
        elif header == b"Pf":
            channels = 1
        else:
            raise ValueError(f"not a PFM file: {path}")

        dim_line = handle.readline().decode("latin-1")
        match = re.match(r"^\s*(\d+)\s+(\d+)\s*$", dim_line)
        if not match:
            raise ValueError(f"malformed PFM dimensions in {path}: {dim_line!r}")
        width, height = int(match.group(1)), int(match.group(2))

        scale = float(handle.readline().rstrip())
        endian = "<" if scale < 0 else ">"

        data = np.frombuffer(handle.read(width * height * channels * 4), dtype=endian + "f4")

    data = data.reshape((height, width, channels) if channels == 3 else (height, width))
    data = np.flipud(data).astype(np.float32)
    return np.ascontiguousarray(data)


def read_kitti_disparity(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a KITTI 16-bit disparity PNG.

    KITTI stores ``disparity * 256`` as uint16 with 0 marking "no ground truth".

    Returns ``(disparity, valid)`` as float32 ``(H, W)`` and bool ``(H, W)``.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"could not read KITTI disparity: {path}")
    if raw.dtype != np.uint16:
        raise ValueError(f"expected uint16 KITTI disparity, got {raw.dtype} for {path}")
    valid = raw > 0
    disparity = raw.astype(np.float32) / 256.0
    disparity[~valid] = 0.0
    return disparity, valid


def read_middlebury_disparity(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Middlebury/ETH3D ``disp0GT.pfm``.

    Invalid pixels are stored as ``inf``.  Returns ``(disparity, valid)`` with
    the invalid entries zeroed.
    """
    disparity = read_pfm(path)
    if disparity.ndim != 2:
        raise ValueError(f"expected single-channel disparity, got shape {disparity.shape} for {path}")
    valid = np.isfinite(disparity) & (disparity > 0)
    disparity = np.where(valid, disparity, 0.0).astype(np.float32)
    return disparity, valid


def read_nonocc_mask(path: str) -> np.ndarray:
    """Read a Middlebury/ETH3D ``mask0nocc.png``: 255 marks non-occluded valid pixels."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    return mask == 255


def read_middlebury_calib(path: str) -> Dict[str, float]:
    """Parse a Middlebury/ETH3D ``calib.txt``.

    Returns a dict with at least ``focal_length`` (fx of cam0, pixels),
    ``baseline`` (metres; the file gives millimetres), ``doffs``, ``ndisp``.
    Depth follows the dataset's own formula ``Z = baseline * f / (d + doffs)``.
    """
    values: Dict[str, str] = {}
    with open(path, "r") as handle:
        for line in handle:
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()

    calib: Dict[str, float] = {}
    if "cam0" in values:
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", values["cam0"])
        if len(numbers) >= 5:
            calib["focal_length"] = float(numbers[0])
            calib["principal_point_x"] = float(numbers[2])
            calib["principal_point_y"] = float(numbers[4])
    if "baseline" in values:
        calib["baseline"] = float(values["baseline"]) / 1000.0  # mm -> m
    if "doffs" in values:
        calib["doffs"] = float(values["doffs"])
    if "ndisp" in values:
        calib["ndisp"] = float(values["ndisp"])
    return calib


def read_kitti_calib(path: str) -> Dict[str, float]:
    """Parse a KITTI stereo ``calib.txt`` / ``calib_cam_to_cam.txt`` style file.

    Handles the 2012/2015 ``calib_cam_to_cam`` form (``P_rect_0x``) and the
    simpler ``calib.txt`` form (``P0:``/``P2:``).  The rectified projection
    matrices give ``P = [f, 0, cx, -f * B_x; ...]``, so the baseline between the
    two colour cameras is ``(P_right[0, 3] - P_left[0, 3]) / -f``.
    """
    matrices: Dict[str, np.ndarray] = {}
    with open(path, "r") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            numbers = value.split()
            if len(numbers) == 12:
                matrices[key.strip()] = np.array([float(n) for n in numbers]).reshape(3, 4)

    left_key = next((k for k in ("P_rect_02", "P2", "P_rect_00", "P0") if k in matrices), None)
    right_key = next((k for k in ("P_rect_03", "P3", "P_rect_01", "P1") if k in matrices), None)
    if left_key is None or right_key is None:
        return {}

    left, right = matrices[left_key], matrices[right_key]
    focal = float(left[0, 0])
    baseline = float(right[0, 3] - left[0, 3]) / -focal if focal != 0 else 0.0
    return {
        "focal_length": focal,
        "principal_point_x": float(left[0, 2]),
        "principal_point_y": float(left[1, 2]),
        "baseline": abs(baseline),
    }


def list_images(directory: str) -> list:
    """Sorted list of image filenames in ``directory``."""
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm", ".tif", ".tiff")
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"not a directory: {directory}")
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(extensions))
