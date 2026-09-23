"""Checkpoint save/load, including the architecture config so a checkpoint is self-describing."""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional

import torch

from ..model.stereo_net import StereoNet, StereoNetConfig


def save_checkpoint(path: str, model: StereoNet, optimizer=None, scheduler=None,
                    epoch: int = 0, iteration: int = 0, extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "model_config": model.config.to_dict(),
        "epoch": epoch,
        "iteration": iteration,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def build_model_from_checkpoint(path: str, map_location="cpu") -> StereoNet:
    """Reconstruct the exact architecture the checkpoint was trained with."""
    payload = load_checkpoint(path, map_location)
    if "model_config" not in payload:
        raise RuntimeError(f"{path} has no model_config; cannot reconstruct the architecture")
    model = StereoNet(StereoNetConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    return model


def checkpoint_hash(path: str) -> str:
    """SHA-256 of the checkpoint file, recorded in evaluation output."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
