"""The training loop. Plain PyTorch, one readable function per stage.

Training is supervised, on the paper's objective and schedule: Adam at 1e-3 with
polynomial decay, 20 epochs, batch 16, AMP. See
:class:`~stereo.losses.PaperObjective` for the loss.

``train`` starts from random initialisation or a checkpoint; ``adapt`` is the
same code with a smaller learning rate and your own data (see
``configs/adapt.yaml``).

Checkpoint selection uses validation EPE -- the quantity the benchmark reports,
measured on data the optimiser never saw.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..data import (BatchGeometricAugment, DatasetMode, build_loader,
                    build_training_datasets)
from ..data.registry import spec_name
from ..losses import labels_from_batch
from ..model import StereoNet
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.seed import set_seed
from .objective import ObjectiveState, SupervisedObjective


def build_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    if config.name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                weight_decay=config.weight_decay)
    if config.name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    if config.name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                               momentum=config.momentum, weight_decay=config.weight_decay)
    raise ValueError(f"unknown optimizer {config.name!r}")


#: A warm-up may never consume more than this fraction of the run. Warm-up
#: lengths are configured in absolute iterations, which silently becomes a
#: disaster on a small dataset: with 2 steps/epoch a 500-iteration warm-up
#: outlasts a 30-epoch run, pinning the learning rate at a few percent of its
#: configured value and leaving the model effectively untrained.
MAX_WARMUP_FRACTION = 0.1


def effective_warmup(configured: int, total_iterations: int) -> int:
    """Clamp a warm-up to a sensible share of the actual run length."""
    ceiling = max(1, int(total_iterations * MAX_WARMUP_FRACTION))
    return max(0, min(configured, ceiling))


def build_scheduler(optimizer, config, total_iterations: int):
    """Warm-up followed by the configured decay, as a per-iteration LambdaLR."""
    warmup = effective_warmup(config.warmup_iterations, total_iterations)

    def factor(iteration: int) -> float:
        if warmup > 0 and iteration < warmup:
            return (iteration + 1) / warmup
        progress = (iteration - warmup) / max(total_iterations - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        if config.schedule == "poly":
            return (1.0 - progress) ** config.poly_exponent
        if config.schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


class Trainer:
    """Label-free stereo trainer."""

    def __init__(self, config: Config, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(config.training.seed)

        self.model = StereoNet(config.model).to(self.device)
        if config.training.init_checkpoint:
            payload = load_checkpoint(config.training.init_checkpoint, map_location=self.device)
            self.model.load_state_dict(payload["model"])
            print(f"initialised from {config.training.init_checkpoint}")

        self.objective = SupervisedObjective(config.loss, downsample=config.model.downsample)
        self.geometric_augment = BatchGeometricAugment(config.data.geometric_augmentation,
                                                       seed=config.training.seed)

        self.train_loader, self.val_loaders = self._build_loaders()
        steps_per_epoch = self._steps_per_epoch()
        self.total_iterations = steps_per_epoch * config.training.epochs
        self.steps_per_epoch = steps_per_epoch
        # Both warm-ups are clamped to the run length; see effective_warmup().
        self.loss_warmup = effective_warmup(config.training.warmup_iterations,
                                            self.total_iterations)
        self.lr_warmup = effective_warmup(config.optimizer.warmup_iterations,
                                          self.total_iterations)
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler = build_scheduler(self.optimizer, config.optimizer,
                                         self.total_iterations)
        self.scaler = torch.amp.GradScaler(self.device.type,
                                           enabled=config.training.use_amp and self.device.type == "cuda")

        self.iteration = 0
        self.start_epoch = 0
        self.best_metric = float("inf")
        self.history: List[Dict[str, Any]] = []
        os.makedirs(config.training.output_dir, exist_ok=True)

        if config.training.resume:
            self._resume(config.training.resume)
            self._restore_history()

    # -- setup -------------------------------------------------------------- #

    def _build_loaders(self):
        cfg = self.config
        train_dataset, weights, summary = build_training_datasets(
            cfg.data.train, DatasetMode.TRAIN, cfg.data.resize,
            cfg.data.photometric_augmentation, seed=cfg.training.seed,
            with_labels=True, flip=cfg.data.horizontal_flip)
        print("training datasets:")
        for entry in summary:
            print(f"  {entry['name']:12s} n={entry['size']:7d} weight={entry['weight']} root={entry['root']}")

        train_loader = build_loader(train_dataset, cfg.training.batch_size, shuffle=weights is None,
                                    num_workers=cfg.training.num_workers, sample_weights=weights,
                                    samples_per_epoch=cfg.training.samples_per_epoch,
                                    seed=cfg.training.seed)

        # One loader per dataset, over its held-out part in a fixed order: a
        # sampled mixture scored different images every epoch, so the number
        # that picks the checkpoint moved for reasons other than the model.
        val_loaders = []
        specs = [spec for spec in cfg.data.validation if spec.enabled]
        if specs:
            print("validation datasets:")
        for index, spec in enumerate(specs):
            val_dataset, _, _ = build_training_datasets(
                [spec], DatasetMode.VALIDATION, cfg.data.resize, None,
                seed=cfg.training.seed + 1, with_labels=True)
            name = spec_name(spec)
            if any(name == existing for existing, _, _ in val_loaders):
                name = f"{name}_{index}"
            if len(val_dataset) == 0:
                print(f"  {name:12s} nothing held out -- not validated")
                continue
            print(f"  {name:12s} n={len(val_dataset):7d} part={spec.part} root={spec.root}")
            # Validation runs for at most max_validation_steps batches an epoch,
            # so it does not need the training loader's worker pool. Keeping
            # them persistent doubled the live worker processes -- at
            # NUM_WORKERS=8 that is 16, each holding a pinned prefetch queue,
            # alive for the whole run to serve a brief pass.
            loader = build_loader(val_dataset, cfg.training.batch_size, shuffle=False,
                                  num_workers=min(cfg.training.num_workers, 2),
                                  seed=cfg.training.seed + 1, drop_last=False,
                                  persistent=False, pin=False)
            val_loaders.append((name, loader, spec.weight))
        return train_loader, val_loaders

    def _steps_per_epoch(self) -> int:
        steps = len(self.train_loader)
        if self.config.training.max_steps_per_epoch:
            steps = min(steps, self.config.training.max_steps_per_epoch)
        return max(steps, 1)

    def _resume(self, path: str) -> None:
        payload = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.start_epoch = payload.get("epoch", 0) + 1
        self.iteration = payload.get("iteration", 0)
        print(f"resumed from {path} at epoch {self.start_epoch}")

    def _restore_history(self) -> None:
        """Carry the previous session's curves forward.

        Without this a resumed run writes a history.json containing only the new
        epochs, so the plotted curves restart at the resume point and the earlier
        training appears to have vanished. Also recovers the best
        score, so resuming cannot overwrite a better checkpoint with a worse one.
        """
        path = os.path.join(self.config.training.output_dir, "history.json")
        if not os.path.isfile(path):
            return
        try:
            with open(path) as handle:
                previous = json.load(handle)
        except (OSError, ValueError) as error:
            print(f"  could not read {path}: {error}")
            return

        self.history = [record for record in previous
                        if record.get("epoch", -1) < self.start_epoch]
        metric = self.config.training.selection_metric
        scores = [record[metric] for record in self.history if metric in record]
        if scores:
            self.best_metric = min(scores)
        print(f"  restored {len(self.history)} earlier epochs from history.json"
              + (f"; best {metric} so far {self.best_metric:.5f}" if scores else ""))

    # -- batch preparation --------------------------------------------------- #

    def _prepare(self, batch: Dict[str, Any], augment: bool) -> Dict[str, Any]:
        """Move to device, apply the per-batch geometric augmentation, split the views.

        Returns ``student_left/right`` (possibly colour-jittered) and
        ``clean_left/right`` (never jittered).  The photometric loss reconstructs the clean pair
        -- weak augmentation -- and the student the jittered one; both share the
        *same* geometry, so no disparity rescaling is needed between them.
        """

        moved = {key: value.to(self.device, non_blocking=True)
                 for key, value in batch.items() if torch.is_tensor(value)}
        moved["metadata"] = batch.get("metadata", {})
        if augment:
            moved, _ = self.geometric_augment(moved)

        clean_left = moved.get("left_clean", moved["left"])
        clean_right = moved.get("right_clean", moved["right"])
        return {
            "student_left": moved["left"],
            "student_right": moved["right"],
            "clean_left": clean_left,
            "clean_right": clean_right,
            "valid_mask": moved.get("valid_mask"),
            "labels": labels_from_batch(moved),
            "metadata": moved["metadata"],
        }

    # -- epochs -------------------------------------------------------------- #

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        cfg = self.config
        totals: Dict[str, float] = {}
        count = 0
        steps = self._steps_per_epoch()
        started = time.time()

        for step, batch in enumerate(self.train_loader):
            if step >= steps:
                break
            views = self._prepare(batch, augment=True)

            state = ObjectiveState(
                iteration=self.iteration,
                epoch=epoch,
                warmup_scale=0.0 if self.iteration < self.loss_warmup else 1.0)

            with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                # Left-referenced only: the loss is supervised against the left
                # view's ground truth, so the mirrored right pass costs a forward
                # and contributes nothing.
                student_outputs = self.model(views["student_left"], views["student_right"],
                                             directions=("left",))
                result = self.objective(student_outputs,
                                        {"left": views["clean_left"], "right": views["clean_right"]},
                                        state, labels=views.get("labels"),
                                        valid_mask=views.get("valid_mask"))
                loss = result["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if cfg.optimizer.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optimizer.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()


            self.iteration += 1
            count += 1
            for key, value in result["logs"].items():
                totals[key] = totals.get(key, 0.0) + value

            if self.iteration % cfg.training.log_every == 0:
                self._log_iteration(epoch, step, steps, result["logs"])

        averages = {key: value / max(count, 1) for key, value in totals.items()}
        averages["lr"] = self.scheduler.get_last_lr()[0]
        averages["seconds"] = time.time() - started
        self._check_collapse(averages)
        return averages

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validation metrics, used only to choose a checkpoint.

        Each dataset's held-out part is scored on its own: ``val/<dataset>/<m>``
        for each, and ``val/<m>`` their mean weighted as in training. Runs under
        no_grad -- without it every batch builds an autograd graph for a result
        whose gradient is never used -- and stops after max_validation_steps
        batches per dataset.
        """
        if not self.val_loaders:
            return {}
        self.model.eval()
        limit = self.config.training.max_validation_steps
        logs: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        for name, loader, weight in self.val_loaders:
            totals: Dict[str, float] = {}
            count = 0
            for batch in loader:
                if limit is not None and count >= limit:
                    break
                views = self._prepare(batch, augment=False)
                outputs = self.model(views["clean_left"], views["clean_right"], directions=("left",))
                state = ObjectiveState(iteration=self.iteration, epoch=epoch, warmup_scale=1.0)
                result = self.objective(outputs, {"left": views["clean_left"], "right": views["clean_right"]},
                                        state, labels=views.get("labels"),
                                        valid_mask=views.get("valid_mask"))
                for key, value in result["logs"].items():
                    totals[key] = totals.get(key, 0.0) + value
                count += 1
            if count:
                weights[name] = weight
                logs.update({f"val/{name}/{key}": value / count for key, value in totals.items()})

        for key in {key.split("/", 2)[2] for key in logs}:
            parts = [(logs[f"val/{name}/{key}"], weight) for name, weight in weights.items()
                     if f"val/{name}/{key}" in logs]
            logs[f"val/{key}"] = sum(v * w for v, w in parts) / max(sum(w for _, w in parts), 1e-12)
        return logs

    # -- driver -------------------------------------------------------------- #

    def fit(self) -> str:
        cfg = self.config.training
        print(f"device={self.device}  parameters={self.model.num_parameters():,}  "
              f"num_disparities={self.model.num_disparities} (downsample={self.model.scale})")
        self._report_schedule()

        last_path = os.path.join(cfg.output_dir, "last.pt")
        best_path = os.path.join(cfg.output_dir, "best.pt")

        for epoch in range(self.start_epoch, cfg.epochs):
            train_logs = self.train_epoch(epoch)
            val_logs = self.validate(epoch)
            record = {"epoch": epoch, **{f"train/{k}": v for k, v in train_logs.items()}, **val_logs}
            self.history.append(record)
            self._print_epoch(epoch, train_logs, val_logs)

            save_checkpoint(last_path, self.model, self.optimizer, self.scheduler,
                            epoch, self.iteration,
                            extra={"history_tail": self.history[-1]})

            if cfg.visualize_every and epoch % cfg.visualize_every == 0:
                self.save_visualization(epoch)

            # Checkpoint selection on held-out data; the training EPE only when
            # nothing is held out.
            selection = val_logs.get(cfg.selection_metric, train_logs.get("epe"))
            if selection is not None and selection < self.best_metric:
                self.best_metric = selection
                save_checkpoint(best_path, self.model, self.optimizer, self.scheduler,
                                epoch, self.iteration,
                                extra={"selection_metric": cfg.selection_metric,
                                       "selected_on": "validation" if cfg.selection_metric in val_logs
                                       else "training",
                                       "selection_value": selection})
                print(f"  new best by {cfg.selection_metric} = {selection:.5f} -> {best_path}")

            with open(os.path.join(cfg.output_dir, "history.json"), "w") as handle:
                json.dump(self.history, handle, indent=2)

        return best_path if os.path.exists(best_path) else last_path

    # -- visualisation ---------------------------------------------------------- #

    @torch.no_grad()
    def save_visualization(self, epoch: int) -> Optional[str]:
        """Write one row per sample: left, right, predicted disparity, warped right.

        The fourth panel is the *reconstruction* -- the right view warped into the
        left by the predicted disparity. Beside the left view it makes the
        prediction readable: where the warp looks like the left image the
        disparity is right, and where it smears or doubles it is wrong.

        Rows come from the training loader and from each dataset's held-out
        validation part, always from the *clean* (un-jittered) images. Purely a
        monitoring artefact: no ground truth is involved and nothing here feeds
        back into the objective.
        """
        try:
            from ..utils.visualization import HAVE_MATPLOTLIB, colorize, to_numpy_image
        except Exception as error:                      # pragma: no cover
            print(f"  visualisation unavailable: {error}")
            return None
        if not HAVE_MATPLOTLIB:
            print("  visualisation skipped: matplotlib is not installed")
            return None
        import matplotlib.pyplot as plt

        from ..geometry import warp_right_to_left

        # Both splits, so the figure shows generalisation and not only fit: rows
        # from the training loader above held-out rows, which the validation
        # datasets share between them.
        per_split = max(self.config.training.visualize_samples, 1)
        sources = [("train", self.train_loader, per_split)]
        count = len(self.val_loaders)
        for position, (name, loader, _) in enumerate(self.val_loaders):
            rows = per_split // count + (1 if position < per_split % count else 0)
            if rows:
                sources.append((f"val {name}", loader, rows))

        was_training = self.model.training
        self.model.eval()
        blocks = []
        for name, loader, rows in sources:
            try:
                batch = next(iter(loader))
            except StopIteration:                        # pragma: no cover
                continue
            views = self._prepare(batch, augment=False)
            take = min(rows, views["clean_left"].shape[0])
            left = views["clean_left"][:take]
            right = views["clean_right"][:take]
            with torch.no_grad():
                out = self.model(left, right, directions=("left",))["left"]
                disparity = out["disparity"]
                warped, valid = warp_right_to_left(right, disparity)
                residual = (warped - left).abs().mean(dim=1, keepdim=True) * valid
                coarse = torch.nn.functional.interpolate(
                    out["disparity_small"], size=disparity.shape[-2:],
                    mode="bilinear", align_corners=False) * self.model.scale
            mask = views.get("valid_mask")
            blocks.append({"name": name, "left": left, "right": right,
                           "disparity": disparity, "warped": warped, "valid": valid,
                           "residual": residual, "coarse": coarse,
                           "mask": mask[:take] if mask is not None else None})
        if was_training:
            self.model.train()
        if not blocks:
            return None

        directory = os.path.join(self.config.training.output_dir, "visualizations")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"epoch_{epoch:04d}.png")

        # Flatten the blocks into rows, remembering which split each came from.
        entries = []
        for block in blocks:
            for index in range(block["left"].shape[0]):
                entries.append((block, index))

        # A ragged (aspect-preserving) batch is padded to its tallest member, so
        # each sample is cropped back to its own real rows and given a figure row
        # sized to its own aspect ratio.
        heights = []
        for block, index in entries:
            rows_here = block["left"].shape[-2]
            if block["mask"] is not None:
                rows_here = max(int(block["mask"][index, 0, :, 0].sum().item()), 1)
            heights.append(rows_here)

        width = entries[0][0]["left"].shape[-1]
        panel_w = 4.6
        figure_heights = [panel_w * r / max(width, 1) for r in heights]
        fig = plt.figure(figsize=(4 * panel_w, sum(figure_heights) + 0.75), facecolor="white")
        grid = fig.add_gridspec(len(entries), 4, height_ratios=figure_heights,
                                wspace=0.03, hspace=0.12)

        columns = ("left", "right", "predicted disparity", "right warped into left")
        seen = {}
        for row, (block, index) in enumerate(entries):
            keep = slice(0, heights[row])
            valid_here = block["valid"][index][:, keep]
            residual_here = block["residual"][index][:, keep]
            per_pixel = residual_here[valid_here > 0.5]
            disparity_here = block["disparity"][index][:, keep]
            images = (to_numpy_image(block["left"][index:index + 1, :, keep]),
                      to_numpy_image(block["right"][index:index + 1, :, keep]),
                      colorize(disparity_here),
                      to_numpy_image(block["warped"][index:index + 1, :, keep]))
            notes = (None, None,
                     f"{float(disparity_here.min()):.1f} - {float(disparity_here.max()):.1f} px"
                     f"   (range 0-{self.model.max_disparity})",
                     "photometric residual "
                     f"{float(per_pixel.mean()) if per_pixel.numel() else float('nan'):.4f}")

            seen[block["name"]] = seen.get(block["name"], 0) + 1
            label = f"{block['name']} {seen[block['name']]}"
            for position, (image, note, column) in enumerate(zip(images, notes, columns)):
                axis = fig.add_subplot(grid[row, position])
                axis.imshow(image, aspect="auto")
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_edgecolor("#d9d9d9")
                if row == 0:
                    axis.set_title(column, fontsize=12, color="#222222", pad=8)
                if position == 0:
                    colour = "#1a6fb5" if block["name"] == "train" else "#b5541a"
                    axis.set_ylabel(label, fontsize=10, color=colour, labelpad=6)
                if note:
                    axis.text(0.015, 0.04, note, transform=axis.transAxes, fontsize=9,
                              color="white", va="bottom", ha="left",
                              bbox=dict(boxstyle="round,pad=0.32", facecolor="#000000",
                                        alpha=0.58, edgecolor="none"))

        # Summary statistics are aggregated as SCALARS, not by concatenating the
        # blocks: with an aspect-preserving resize the train and val batches have
        # different heights, so they do not stack.
        def across(key, reduce):
            return reduce([reduce(block[key]).item() for block in blocks])

        low = min(float(block["disparity"].min()) for block in blocks)
        high = max(float(block["disparity"].max()) for block in blocks)
        mean = sum(float(block["disparity"].mean()) for block in blocks) / len(blocks)
        refine = sum(float((block["disparity"] - block["coarse"]).abs().mean())
                     for block in blocks) / len(blocks)
        warp = sum(float(block["valid"].mean()) for block in blocks) / len(blocks)
        splits = " + ".join(f"{sum(1 for b, _ in entries if b['name'] == name)} {name}"
                            for name in dict.fromkeys(b["name"] for b in blocks))
        fig.suptitle(
            f"epoch {epoch}   |   {splits}   |   disparity {low:.1f}-{high:.1f} px, "
            f"mean {mean:.1f}   |   refinement moves it {refine:+.2f} px   |   "
            f"valid warp {warp * 100:.0f}%",
            fontsize=13, color="#222222", y=0.995)
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  visualisation -> {path}")
        return path

    # -- logging -------------------------------------------------------------- #

    def _report_schedule(self) -> None:
        """State the actual schedule, and say so when the run is too small to train."""
        samples = len(self.train_loader.dataset)
        print(f"dataset={samples} pairs  steps/epoch={self.steps_per_epoch}  "
              f"epochs={self.config.training.epochs}  total_iterations={self.total_iterations}")
        for label, configured, effective in (
                ("learning-rate warm-up", self.config.optimizer.warmup_iterations, self.lr_warmup),
                ("loss warm-up", self.config.training.warmup_iterations, self.loss_warmup)):
            note = ""
            if effective != configured:
                note = (f"  <- clamped from {configured}; it would otherwise outlast "
                        f"{100 * MAX_WARMUP_FRACTION:.0f}% of the run")
            print(f"  {label}: {effective} iterations{note}")

        if self.total_iterations < 200:
            print(f"\n  WARNING: only {self.total_iterations} optimiser steps in this entire run. "
                  f"A {self.model.num_parameters():,}-parameter network trained from random "
                  f"initialisation\n"
                  f"  needs orders of magnitude more. With {samples} training pairs, raise "
                  f"epochs, lower batch_size, or -- far better -- attach more data.\n"
                  f"  Expect the result to be close to its initialisation, not a trained model.")

    @staticmethod
    def _breakdown(logs: Dict[str, float]) -> str:
        """The loss written as what it is made of, largest term first.

        Weighted contributions, not raw values: raw numbers answer "how big is
        this residual", which does not say what the optimiser is following. A
        term can be numerically large and contribute almost nothing, or small and
        dominate -- and reading the wrong one is how a term at 99% of the
        objective went unnoticed.
        """
        parts = {name[5:]: value for name, value in logs.items() if name.startswith("part/")}
        parts = {name: value for name, value in parts.items() if abs(value) > 1e-9}
        if not parts:
            return ""
        total = sum(parts.values()) or 1.0
        ordered = sorted(parts.items(), key=lambda item: -abs(item[1]))
        return "  ".join(f"{name} {value:.4f}({100 * value / total:.0f}%)"
                         for name, value in ordered)

    def _log_iteration(self, epoch: int, step: int, steps: int, logs: Dict[str, float]) -> None:
        head = f"ep {epoch} [{step + 1}/{steps}]  loss {logs['total']:.4f}"
        breakdown = self._breakdown(logs)
        detail = [f"d[{logs['disparity_min']:.1f},{logs['disparity_max']:.1f}]"
                  f" mean {logs['disparity_mean']:.1f}",
                  f"warp {100 * logs['valid_warp_ratio']:.0f}%"]
        if "epe" in logs:
            detail.append(f"epe {logs['epe']:.2f}px")
        if "labelled_ratio" in logs:
            detail.append(f"GT pixels {100 * logs['labelled_ratio']:.0f}%")
        if "refine_delta" in logs:
            detail.append(f"refine{logs['refine_delta']:+.1f}")
        detail.append(f"lr {self.scheduler.get_last_lr()[0]:.1e}")

        print(f"  {head}  =  {breakdown}" if breakdown else f"  {head}")
        print(f"       {'  '.join(detail)}")

    def _print_epoch(self, epoch: int, train_logs, val_logs) -> None:
        line = f"epoch {epoch:3d}  loss {train_logs['total']:.4f}"
        breakdown = self._breakdown(train_logs)
        if breakdown:
            line += f"  =  {breakdown}"
        print(line)

        detail = [f"{train_logs['seconds']:.0f}s"]
        if "epe" in train_logs:
            detail.append(f"epe {train_logs['epe']:.2f}px")
        if "labelled_ratio" in train_logs:
            detail.append(f"GT pixels {100 * train_logs['labelled_ratio']:.0f}%")
        if val_logs:
            detail.append(f"val loss {val_logs.get('val/total', float('nan')):.4f}")
            if "val/epe" in val_logs:
                each = ", ".join(f"{name} {val_logs[f'val/{name}/epe']:.2f}"
                                 for name, _, _ in self.val_loaders if f"val/{name}/epe" in val_logs)
                detail.append(f"val epe {val_logs['val/epe']:.2f}px ({each})")
        print(f"           {'  '.join(detail)}")

    def _check_collapse(self, averages: Dict[str, float]) -> None:
        """Warn when training looks like it is degenerating."""
        if averages.get("disparity_max", 1.0) < 1e-3:
            print("  WARNING: disparity collapsed to zero. Lower the smoothness weight or "
                  "check that left/right are not swapped.")
        if averages.get("mean_confidence", 0.0) > 0.99:
            print("  WARNING: mean confidence > 0.99; the matchability target may have "
                  "collapsed to 'confident everywhere'. Reduce loss.confidence.")
        spread = averages.get("disparity_max", 0.0) - averages.get("disparity_min", 0.0)
        if 0.0 < spread < 1.0:
            print(f"  WARNING: disparity spread is only {spread:.3f} px -- the prediction is "
                  "nearly constant. A flat field is exactly left-right consistent, so check "
                  "that loss.left_right and loss.smoothness are not dominating.")
