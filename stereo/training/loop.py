"""The training loop.  Plain PyTorch, one readable function per stage.

There is no ground-truth tensor in this file.  ``assert_label_free`` is called on
every batch, so a dataset that leaked labels would raise on the first iteration.

Stages
------
Stage 1  photometric + smoothness + left-right consistency, from random init.
Stage 2  the same, plus EMA-teacher pseudo-labels, after ``teacher.start_epoch``.
Stage 3  identical code, started from a checkpoint with a smaller learning rate
         (that is the only difference; see ``configs/adapt_unlabeled.yaml``).

Checkpoint selection uses a **label-free** criterion: the validation photometric
reconstruction loss.  Ground-truth metrics are never consulted during training.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..config import Config
from ..data import (BatchGeometricAugment, DatasetMode, assert_label_free, build_loader,
                    build_training_datasets)
from ..model import StereoNet
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.seed import set_seed
from .objective import LabelFreeObjective, ObjectiveState
from .teacher import EmaTeacher, pseudo_label_weight


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

        self.objective = LabelFreeObjective(config.loss, config.teacher)
        self.geometric_augment = BatchGeometricAugment(config.data.geometric_augmentation,
                                                       seed=config.training.seed)

        self.train_loader, self.val_loader = self._build_loaders()
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

        self.teacher: Optional[EmaTeacher] = None
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
            cfg.data.photometric_augmentation, seed=cfg.training.seed)
        print("training datasets (images only, no labels):")
        for entry in summary:
            print(f"  {entry['name']:12s} n={entry['size']:7d} weight={entry['weight']} root={entry['root']}")

        train_loader = build_loader(train_dataset, cfg.training.batch_size, shuffle=weights is None,
                                    num_workers=cfg.training.num_workers, sample_weights=weights,
                                    samples_per_epoch=cfg.training.samples_per_epoch,
                                    seed=cfg.training.seed)

        val_loader = None
        if cfg.data.validation:
            val_dataset, val_weights, val_summary = build_training_datasets(
                cfg.data.validation, DatasetMode.VALIDATION, cfg.data.resize, None,
                seed=cfg.training.seed + 1)
            print("label-free validation datasets (images only):")
            for entry in val_summary:
                print(f"  {entry['name']:12s} n={entry['size']:7d} root={entry['root']}")
            val_loader = build_loader(val_dataset, cfg.training.batch_size, shuffle=False,
                                      num_workers=cfg.training.num_workers,
                                      sample_weights=val_weights, seed=cfg.training.seed + 1,
                                      drop_last=False)
        return train_loader, val_loader

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
        if "teacher" in payload:
            self.teacher = EmaTeacher(self.model, self.config.teacher.ema_decay)
            self.teacher.load_state_dict(payload["teacher"])
            self.teacher.to(self.device)
        print(f"resumed from {path} at epoch {self.start_epoch}")

    def _restore_history(self) -> None:
        """Carry the previous session's curves forward.

        Without this a resumed run writes a history.json containing only the new
        epochs, so the plotted curves restart at the resume point and the earlier
        training appears to have vanished. Also recovers the best label-free
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
        ``clean_left/right`` (never jittered).  The teacher sees the clean pair
        -- weak augmentation -- and the student the jittered one; both share the
        *same* geometry, so no disparity rescaling is needed between them.
        """
        assert_label_free(batch, context="training batch")

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
            "metadata": moved["metadata"],
        }

    # -- epochs -------------------------------------------------------------- #

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        cfg = self.config
        teacher_active = cfg.teacher.enabled and epoch >= cfg.teacher.start_epoch
        if teacher_active and self.teacher is None:
            print(f"epoch {epoch}: starting EMA teacher (decay={cfg.teacher.ema_decay})")
            self.teacher = EmaTeacher(self.model, cfg.teacher.ema_decay).to(self.device)

        pseudo_scale = pseudo_label_weight(epoch, cfg.teacher.start_epoch, cfg.teacher.ramp_epochs) \
            if teacher_active else 0.0

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
                warmup_scale=0.0 if self.iteration < self.loss_warmup else 1.0,
                pseudo_scale=pseudo_scale)

            teacher_outputs = None
            if self.teacher is not None and pseudo_scale > 0.0:
                with torch.no_grad():
                    teacher_outputs = self.teacher.predict(views["clean_left"], views["clean_right"])

            with torch.amp.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                student_outputs = self.model(views["student_left"], views["student_right"],
                                             directions=("left", "right"))
                result = self.objective(student_outputs,
                                        {"left": views["clean_left"], "right": views["clean_right"]},
                                        state, teacher_outputs, max_disparity=self.model.max_disparity,
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

            if self.teacher is not None:
                self.teacher.update(self.model)

            self.iteration += 1
            count += 1
            for key, value in result["logs"].items():
                totals[key] = totals.get(key, 0.0) + value

            if self.iteration % cfg.training.log_every == 0:
                self._log_iteration(epoch, step, steps, result["logs"])

        averages = {key: value / max(count, 1) for key, value in totals.items()}
        averages["lr"] = self.scheduler.get_last_lr()[0]
        averages["seconds"] = time.time() - started
        self._check_collapse(averages, pseudo_scale)
        return averages

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Label-free validation: photometric, consistency and coverage only."""
        if self.val_loader is None:
            return {}
        self.model.eval()
        totals: Dict[str, float] = {}
        count = 0
        for batch in self.val_loader:
            views = self._prepare(batch, augment=False)
            outputs = self.model(views["clean_left"], views["clean_right"], directions=("left", "right"))
            state = ObjectiveState(iteration=self.iteration, epoch=epoch, warmup_scale=1.0, pseudo_scale=0.0)
            # diagnostics=True also evaluates the cost-volume loss when its
            # weight is 0. It is the clearest read on whether the cost volume is
            # matching at all, and without this it disappears from the logs
            # exactly when it is switched off. Affordable here because validation
            # runs once per epoch and is already no_grad; it costs ~89% of a
            # forward pass, so it is never done per training step.
            result = self.objective(outputs, {"left": views["clean_left"], "right": views["clean_right"]},
                                    state, None, max_disparity=self.model.max_disparity,
                                    valid_mask=views.get("valid_mask"), diagnostics=True)
            for key, value in result["logs"].items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        return {f"val/{key}": value / max(count, 1) for key, value in totals.items()}

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
                            self.teacher, epoch, self.iteration,
                            extra={"history_tail": self.history[-1]})

            if cfg.visualize_every and epoch % cfg.visualize_every == 0:
                self.save_visualization(epoch)

            # Checkpoint selection on a LABEL-FREE criterion only.
            selection = val_logs.get(cfg.selection_metric, train_logs.get("photometric"))
            if selection is not None and selection < self.best_metric:
                self.best_metric = selection
                save_checkpoint(best_path, self.model, self.optimizer, self.scheduler,
                                self.teacher, epoch, self.iteration,
                                extra={"selection_metric": cfg.selection_metric,
                                       "selection_value": selection,
                                       "selection_is_label_free": True})
                print(f"  new best by {cfg.selection_metric} = {selection:.5f} -> {best_path}")

            with open(os.path.join(cfg.output_dir, "history.json"), "w") as handle:
                json.dump(self.history, handle, indent=2)

        return best_path if os.path.exists(best_path) else last_path

    # -- visualisation ---------------------------------------------------------- #

    @torch.no_grad()
    def save_visualization(self, epoch: int) -> Optional[str]:
        """Write a left / right / predicted-disparity figure for this epoch.

        Drawn from the validation loader when there is one, otherwise from the
        training loader, and always from the *clean* (un-jittered) images so the
        panel shows what the model actually sees geometrically. Purely a
        monitoring artefact: no ground truth is involved, and nothing here feeds
        back into the objective.
        """
        try:
            from ..utils.visualization import HAVE_MATPLOTLIB, colorize, save_evaluation_figure, \
                to_numpy_image
        except Exception as error:                      # pragma: no cover
            print(f"  visualisation unavailable: {error}")
            return None
        if not HAVE_MATPLOTLIB:
            print("  visualisation skipped: matplotlib is not installed")
            return None

        loader = self.val_loader or self.train_loader
        try:
            batch = next(iter(loader))
        except StopIteration:                            # pragma: no cover
            return None

        was_training = self.model.training
        self.model.eval()
        views = self._prepare(batch, augment=False)
        outputs = self.model(views["clean_left"], views["clean_right"], directions=("left",))
        disparity = outputs["left"]["disparity"]
        confidence = outputs["left"]["confidence"]
        small = outputs["left"]["disparity_small"]
        coarse = torch.nn.functional.interpolate(
            small, size=disparity.shape[-2:], mode="bilinear", align_corners=False) * self.model.scale
        if was_training:
            self.model.train()

        directory = os.path.join(self.config.training.output_dir, "visualizations")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"epoch_{epoch:04d}.png")

        count = min(self.config.training.visualize_samples, disparity.shape[0])
        panels = {}
        for index in range(count):
            suffix = f" [{index}]" if count > 1 else ""
            panels[f"left{suffix}"] = to_numpy_image(views["clean_left"][index:index + 1])
            panels[f"right{suffix}"] = to_numpy_image(views["clean_right"][index:index + 1])
            panels[f"disparity{suffix}"] = colorize(disparity[index])
            # The cost volume's own estimate, before refinement. If this is flat
            # or noise while the refined map looks detailed, the stereo matching
            # is not working and the refinement is inventing the detail.
            panels[f"cost volume{suffix}"] = colorize(coarse[index])
            # Confidence comes free from the same forward pass and is the clearest
            # early warning of the matchability head collapsing to "certain everywhere".
            panels[f"confidence{suffix}"] = colorize(confidence[index], 0.0, 1.0, cmap="viridis")
        save_evaluation_figure(
            path, panels,
            title=f"epoch {epoch}  |  disparity min {float(disparity.min()):.1f} "
                  f"max {float(disparity.max()):.1f} mean {float(disparity.mean()):.1f} px "
                  f"(model range 0..{self.model.max_disparity})")
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

    def _log_iteration(self, epoch: int, step: int, steps: int, logs: Dict[str, float]) -> None:
        parts = [f"ep {epoch} [{step + 1}/{steps}]",
                 f"loss {logs['total']:.4f}",
                 f"photo {logs['photometric']:.4f}",
                 f"(ssim {logs['photometric_ssim']:.3f} l1 {logs['photometric_l1']:.3f})",
                 f"smooth {logs['smoothness']:.4f}",
                 f"lr_cons {logs['left_right']:.4f}"]
        if logs.get("pseudo_scale", 0.0) > 0:
            parts.append(f"pseudo {logs.get('pseudo', 0.0):.4f} (cov {logs['pseudo_valid_ratio']:.3f})")
        if "mean_confidence" in logs:
            parts.append(f"conf {logs['mean_confidence']:.3f}")
        parts.append(f"d[{logs['disparity_min']:.1f},{logs['disparity_max']:.1f}] "
                     f"mean {logs['disparity_mean']:.2f}")
        parts.append(f"warp {logs['valid_warp_ratio']:.3f}")
        if "refine_delta" in logs:
            parts.append(f"cv {logs['cost_volume_mean']:.1f} refine{logs['refine_delta']:+.1f}")
        if "cost_volume_loss" in logs:
            # The cost volume is the only part doing real stereo matching, so its
            # loss and how much of the frame it supervises are the numbers that
            # say whether matching is happening at all.
            parts.append(f"cvloss {logs['cost_volume_loss']:.3f} "
                         f"(sup {logs['cost_supervised_ratio']:.2f})")
        parts.append(f"lr {self.scheduler.get_last_lr()[0]:.2e}")
        print("  " + "  ".join(parts))

    def _print_epoch(self, epoch: int, train_logs, val_logs) -> None:
        # cvloss is shown because it is the term that does the actual stereo
        # matching and it dominates the total: cross-entropy over D candidates
        # starts near log(D), so at 32 bins the run begins around 3.47 and most
        # of train_loss is this one number. Watching `total` alone mostly watches
        # the cost volume, and `photo` alone misses it entirely.
        # left_right carries its weight in the label: it is logged whatever the
        # weight is, and at w=0 (the default, docs/REPORT.md 16b) it does not
        # enter the objective at all.
        parts = [f"epoch {epoch:3d}",
                 f"train_loss {train_logs['total']:.4f}",
                 f"photo {train_logs['photometric']:.4f}"]
        bins = max(self.model.num_disparities // self.model.scale, 2)
        reference = math.log(bins)
        if "cost_volume_loss" in train_logs:
            parts.append(f"cvloss {train_logs['cost_volume_loss']:.4f}/{reference:.2f}@init")
            parts.append(f"sup {train_logs.get('cost_supervised_ratio', 0.0):.2f}")
        elif val_logs and "val/cost_volume_loss" in val_logs:
            # Weight is 0, so it is measured on validation only. Marked (diag) to
            # make clear it is being watched, not optimised.
            parts.append(f"cvloss {val_logs['val/cost_volume_loss']:.4f}"
                         f"/{reference:.2f}@init(diag)")
            parts.append(f"sup {val_logs.get('val/cost_supervised_ratio', 0.0):.2f}")
        # The raw left-right error is in PIXELS and is logged whatever its weight,
        # so the weight is shown next to it: at w=0 it is a diagnostic, not a term.
        parts.append(f"lr_cons {train_logs['left_right']:.4f}(w={self.config.loss.left_right:g})")
        parts.append(f"pseudo_cov {train_logs['pseudo_valid_ratio']:.3f}")
        parts.append(f"{train_logs['seconds']:.0f}s")
        line = "  ".join(parts)
        if val_logs:
            line += f"  | val photo {val_logs.get('val/photometric', float('nan')):.4f}"
            if "cost_volume_loss" in train_logs and "val/cost_volume_loss" in val_logs:
                line += f" cvloss {val_logs['val/cost_volume_loss']:.4f}"
        print(line)

    def _check_collapse(self, averages: Dict[str, float], pseudo_scale: float) -> None:
        """Warn when self-training looks like it is degenerating."""
        cfg = self.config.teacher
        if pseudo_scale > 0.0:
            ratio = averages.get("pseudo_valid_ratio", 0.0)
            if ratio < cfg.min_valid_ratio:
                print(f"  WARNING: pseudo-label coverage {ratio:.3f} < {cfg.min_valid_ratio}: "
                      "the filter is rejecting almost everything; loosen the thresholds or "
                      "extend the warm-up.")
            elif ratio > cfg.max_valid_ratio:
                print(f"  WARNING: pseudo-label coverage {ratio:.3f} > {cfg.max_valid_ratio}: "
                      "the filter is accepting almost everything; the teacher's errors are "
                      "being copied wholesale.")
        if averages.get("disparity_max", 1.0) < 1e-3:
            print("  WARNING: disparity collapsed to zero. Lower the smoothness weight or "
                  "check that left/right are not swapped.")
        if averages.get("mean_confidence", 0.0) > 0.99:
            print("  WARNING: mean confidence > 0.99; the matchability target may have "
                  "collapsed to 'confident everywhere'. Reduce loss.confidence.")
