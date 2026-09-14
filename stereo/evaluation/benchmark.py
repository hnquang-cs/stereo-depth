"""Frozen-checkpoint benchmark runner.

    load checkpoint -> model.eval() -> no_grad -> inference -> load ground truth
    -> apply the protocol's mask -> metrics -> summary.json + per_image.csv

This is the only place in the repository where a model and ground truth meet,
and it happens after the weights are frozen, so no optimisation can be
influenced by labels.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch

from ..data import DatasetMode, DatasetSpec, build_benchmark_dataset, collate_samples
from ..geometry import disparity_to_depth
from ..model import StereoNet
from ..postprocess import PostProcessConfig, postprocess_disparity
from .confidence_metrics import confidence_metrics
from .depth_metrics import DepthAccumulator, depth_metrics, depth_valid_mask, median_scale_factor
from .disparity_metrics import DisparityAccumulator, disparity_valid_mask
from .protocols import PUBLISHED_RESULTS, EvaluationProtocol


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                                       ).decode().strip()
    except Exception:
        return None


@torch.no_grad()
def evaluate_checkpoint(model: StereoNet,
                        protocol: EvaluationProtocol,
                        dataset_root: str,
                        device: torch.device,
                        postprocess: Optional[PostProcessConfig] = None,
                        max_samples: Optional[int] = None,
                        compute_confidence_metrics: bool = True,
                        progress: bool = True) -> Dict[str, Any]:
    """Run one protocol against one frozen model.

    Args:
        model: already loaded; set to ``eval()`` here.
        protocol: see :mod:`stereo.evaluation.protocols`.
        dataset_root: path to the benchmark data.
        device: inference device.
        postprocess: post-processing settings; ``None`` uses the protocol's flag.
        max_samples: evaluate only the first N images (smoke runs).

    Returns:
        The summary dictionary that :func:`write_results` serialises.
    """
    model.eval().to(device)

    spec = DatasetSpec(type=protocol.dataset_type, root=dataset_root, options=dict(protocol.dataset_options))
    dataset = build_benchmark_dataset(spec)
    if dataset.mode is not DatasetMode.BENCHMARK:
        raise RuntimeError("benchmark dataset was not constructed in BENCHMARK mode")

    post_config = postprocess or PostProcessConfig(enabled=protocol.postprocess)

    accumulators = {"all": DisparityAccumulator()}
    if protocol.evaluate_nonocc:
        accumulators["nonocc"] = DisparityAccumulator()
    depth_accumulator = DepthAccumulator() if protocol.depth_metrics else None
    confidence_rows: List[Dict[str, float]] = []
    gt_pixels_before_postprocess = 0
    gt_pixels_after_postprocess = 0

    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for index in range(total):
        batch = collate_samples([dataset[index]])
        left = batch["left"].to(device)
        right = batch["right"].to(device)

        output = model.forward_left(left, right)
        disparity = output["disparity"]
        confidence = output["confidence"]

        if post_config.enabled:
            disparity, post_valid = postprocess_disparity(disparity, confidence, post_config)
        else:
            post_valid = None

        disparity_gt = batch["disparity_gt"].to(device)
        gt_mask = batch.get("valid_gt_mask")
        gt_mask = gt_mask.to(device) if gt_mask is not None else None

        base_valid = disparity_valid_mask(disparity_gt, protocol.max_disparity, protocol.min_disparity,
                                          gt_mask, protocol.ignore_edge)
        gt_pixels_before_postprocess += int(base_valid.sum())
        if post_valid is not None:
            # Post-processing removes predictions; those pixels are excluded from the
            # metric, so the coverage is reported alongside to keep the row honest --
            # a metric computed on 40% of the pixels is not comparable to one on 100%.
            base_valid = base_valid & (post_valid > 0.5)
        gt_pixels_after_postprocess += int(base_valid.sum())

        sample_id = str(batch["metadata"].get("sample_id", [index])[0])
        accumulators["all"].update(disparity, disparity_gt, base_valid, sample_id)

        if "nonocc" in accumulators:
            nonocc = batch.get("nonocc_mask")
            if nonocc is not None:
                nonocc_valid = base_valid & (nonocc.to(device) > 0.5)
                accumulators["nonocc"].update(disparity, disparity_gt, nonocc_valid, sample_id)

        if depth_accumulator is not None:
            row = _depth_row(batch, disparity, disparity_gt, base_valid, protocol, device)
            if row:
                depth_accumulator.update(row)

        if compute_confidence_metrics:
            from ..postprocess import upsample_confidence
            full_confidence = upsample_confidence(confidence, disparity.shape[-2:])
            row = confidence_metrics(full_confidence, disparity, disparity_gt, base_valid)
            if row:
                confidence_rows.append(row)

        if progress and (index + 1) % 25 == 0:
            print(f"  [{protocol.name}] {index + 1}/{total}")

    summary: Dict[str, Any] = {
        "protocol": protocol.to_dict(),
        "dataset_root": dataset_root,
        "num_images_evaluated": total,
        "num_images_available": len(dataset),
        "model": {
            "num_disparities": model.num_disparities,
            "downsample": model.scale,
            "max_disparity_mask_bound_of_model": model.max_disparity,
            "num_parameters": model.num_parameters(),
        },
        "postprocess": asdict(post_config),
        "ground_truth_pixel_coverage": (gt_pixels_after_postprocess / gt_pixels_before_postprocess
                                       if gt_pixels_before_postprocess else 0.0),
        "median_scaling_applied": protocol.median_scaling,
        "primary_metric_source": protocol.primary_source,
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": _git_commit(),
        },
    }

    disparity_results = {key: accumulator.compute() for key, accumulator in accumulators.items()}
    summary["disparity_metrics"] = disparity_results
    summary["primary"] = _extract_primary(disparity_results, protocol)
    if depth_accumulator is not None:
        summary["depth_metrics_secondary"] = depth_accumulator.compute()
    if confidence_rows:
        summary["confidence_metrics_secondary"] = _mean_rows(confidence_rows)
    if protocol.name in PUBLISHED_RESULTS:
        summary["published_reference"] = PUBLISHED_RESULTS[protocol.name]

    summary["per_image"] = accumulators["all"].per_image
    return summary


def _depth_row(batch, disparity, disparity_gt, valid, protocol, device) -> Dict[str, float]:
    """Convert predicted and ground-truth disparity to depth and score it."""
    from ..data import metadata_tensor
    metadata = batch["metadata"]
    focal = metadata_tensor(metadata, "focal_length", None, device)
    baseline = metadata_tensor(metadata, "baseline", None, device)
    if focal is None or baseline is None:
        return {}

    doffs = metadata_tensor(metadata, "doffs", 0.0, device)
    offset = doffs.reshape(-1, 1, 1, 1) if doffs is not None else 0.0

    depth_pred, pred_valid = disparity_to_depth(disparity + offset, focal, baseline,
                                                max_depth=protocol.depth_range[1])
    depth_gt, gt_valid = disparity_to_depth(disparity_gt + offset, focal, baseline,
                                            max_depth=protocol.depth_range[1])
    depth_mask = (valid & (pred_valid > 0.5) & (gt_valid > 0.5)
                  & depth_valid_mask(depth_gt, protocol.depth_range[0], protocol.depth_range[1]))

    scale = 1.0
    if protocol.median_scaling:
        scale = median_scale_factor(depth_pred, depth_gt, depth_mask)
    return depth_metrics(depth_pred, depth_gt, depth_mask, scale)


def _extract_primary(disparity_results: Dict[str, Dict[str, float]],
                     protocol: EvaluationProtocol) -> Dict[str, Any]:
    """Pull out the protocol's primary metrics, one entry per occlusion variant."""
    primary: Dict[str, Any] = {}
    for variant, results in disparity_results.items():
        primary[variant] = {key: results.get(key) for key in protocol.primary_metrics}
    return primary


def _mean_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = {key for row in rows for key in row if key != "num_valid"}
    out = {}
    for key in sorted(keys):
        values = [row[key] for row in rows if key in row and row[key] == row[key]]  # drop NaN
        if values:
            out[key] = float(sum(values) / len(values))
    out["num_images"] = float(len(rows))
    return out


def write_results(summary: Dict[str, Any], output_dir: str, config_snapshot: Optional[str] = None) -> None:
    """Write ``summary.json``, ``per_image.csv`` and an optional config copy."""
    os.makedirs(output_dir, exist_ok=True)
    per_image = summary.pop("per_image", [])

    with open(os.path.join(output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)

    if per_image:
        keys = sorted({key for row in per_image for key in row})
        with open(os.path.join(output_dir, "per_image.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_image)

    if config_snapshot is not None:
        with open(os.path.join(output_dir, "config.yaml"), "w") as handle:
            handle.write(config_snapshot)

    summary["per_image"] = per_image


def format_summary(summary: Dict[str, Any]) -> str:
    """Human-readable report with primary and secondary metrics clearly separated."""
    protocol = summary["protocol"]
    lines = [
        "=" * 78,
        f"Protocol      : {protocol['name']}  (dataset={protocol['dataset_type']}, split={protocol['split']})",
        f"Dataset root  : {summary['dataset_root']}",
        f"Images        : {summary['num_images_evaluated']} of {summary['num_images_available']}",
        "Resolution    : native (no resize)" if not protocol.get("resize")
        else f"Resolution    : {protocol['resize']}",
        f"Valid mask    : d_gt > {protocol['min_disparity']}"
        + (f" and d_gt < {protocol['max_disparity']}" if protocol["max_disparity"] else "")
        + (" and dataset validity mask"),
        f"               ({protocol['max_disparity_source']})",
        f"Post-process  : {'ON' if summary['postprocess']['enabled'] else 'OFF (raw network output)'}"
        + (f"   -- scored on {summary.get('ground_truth_pixel_coverage', 1.0) * 100:.1f}% of the "
           f"ground-truth pixels" if summary['postprocess']['enabled'] else ""),
        f"Scale align   : {'MEDIAN SCALING APPLIED' if summary['median_scaling_applied'] else 'none (metric prediction)'}",
        "",
        f"PRIMARY METRICS -- {summary['primary_metric_source']}",
    ]
    for variant, metrics in summary["primary"].items():
        rendered = "  ".join(f"{key}={_fmt(value)}" for key, value in metrics.items())
        lines.append(f"  [{variant:7s}] {rendered}")

    lines.append("")
    lines.append("SECONDARY METRICS -- diagnostic, not from the paper")
    for variant, metrics in summary["disparity_metrics"].items():
        extra = {k: v for k, v in metrics.items() if k not in summary["primary"][variant]}
        lines.append(f"  [{variant}] " + "  ".join(f"{k}={_fmt(v)}" for k, v in sorted(extra.items())))
    if "depth_metrics_secondary" in summary:
        lines.append("  [depth]  " + "  ".join(
            f"{k}={_fmt(v)}" for k, v in sorted(summary["depth_metrics_secondary"].items())))
    if "confidence_metrics_secondary" in summary:
        lines.append("  [conf]   " + "  ".join(
            f"{k}={_fmt(v)}" for k, v in sorted(summary["confidence_metrics_secondary"].items())))

    if "published_reference" in summary:
        reference = summary["published_reference"]
        lines += ["", f"PUBLISHED REFERENCE ({reference['status']}) -- {reference['source']}",
                  "  " + "  ".join(f"{k}={v}" for k, v in reference["metrics"].items())]
        if "caveat" in reference:
            lines.append(f"  CAVEAT: {reference['caveat']}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
