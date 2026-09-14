#!/usr/bin/env python3
"""Ground-truth benchmark evaluation of a frozen checkpoint.

    python evaluate.py --checkpoint outputs/train/best.pt \
                       --protocol sceneflow --dataset-root datasets/sceneflow

This is the ONLY entry point that reads ground truth.  The model is loaded,
frozen, and evaluated; no optimiser is constructed.
"""

from __future__ import annotations

import argparse
import os

import torch

from stereo.evaluation import evaluate_checkpoint, format_summary, get_protocol, write_results
from stereo.evaluation.protocols import PROTOCOLS
from stereo.postprocess import PostProcessConfig
from stereo.utils.checkpoint import build_model_from_checkpoint, checkpoint_hash
from stereo.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", default="sceneflow", choices=sorted(PROTOCOLS))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--postprocess", action="store_true",
                        help="apply the matchability post-processing (the paper's tables do NOT)")
    parser.add_argument("--postprocess-confidence", type=float, default=0.25)
    parser.add_argument("--postprocess-min-region", type=int, default=2000)
    parser.add_argument("--both", action="store_true",
                        help="evaluate raw AND post-processed output, into two subdirectories")
    parser.add_argument("--no-confidence-metrics", action="store_true")
    parser.add_argument("--save-visualizations", type=int, default=0)
    return parser.parse_args()


def run_once(model, protocol, args, device, postprocess, label):
    summary = evaluate_checkpoint(
        model=model, protocol=protocol, dataset_root=args.dataset_root, device=device,
        postprocess=postprocess, max_samples=args.max_samples,
        compute_confidence_metrics=not args.no_confidence_metrics)
    summary["checkpoint"] = os.path.abspath(args.checkpoint)
    summary["checkpoint_sha256"] = checkpoint_hash(args.checkpoint)
    summary["run_label"] = label

    # --both writes two runs, so the label always disambiguates the directory.
    base_dir = args.output_dir or os.path.join("outputs", "evaluation", protocol.name)
    output_dir = os.path.join(base_dir, label) if args.both else (
        args.output_dir or os.path.join("outputs", "evaluation", f"{protocol.name}_{label}"))
    write_results(summary, output_dir)
    print(format_summary(summary))
    print(f"written to {output_dir}/summary.json and per_image.csv")

    if args.save_visualizations:
        save_visualizations(model, protocol, args, device, output_dir)
    return summary


def save_visualizations(model, protocol, args, device, output_dir):
    """Left/right, prediction, ground truth, error, confidence -- for the first N images."""
    from stereo.data import DatasetSpec, build_benchmark_dataset, collate_samples
    from stereo.evaluation.disparity_metrics import disparity_valid_mask
    from stereo.utils.visualization import colorize, save_evaluation_figure, to_numpy_image

    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    dataset = build_benchmark_dataset(DatasetSpec(type=protocol.dataset_type, root=args.dataset_root,
                                                  options=dict(protocol.dataset_options)))
    count = min(args.save_visualizations, len(dataset))
    for index in range(count):
        batch = collate_samples([dataset[index]])
        left = batch["left"].to(device)
        right = batch["right"].to(device)
        with torch.no_grad():
            output = model.forward_left(left, right)
        disparity = output["disparity"]
        disparity_gt = batch["disparity_gt"].to(device)
        gt_mask = batch.get("valid_gt_mask")
        valid = disparity_valid_mask(disparity_gt, protocol.max_disparity, protocol.min_disparity,
                                     gt_mask.to(device) if gt_mask is not None else None)
        error = torch.abs(disparity - disparity_gt)

        vmax = float(disparity_gt[valid].max()) if valid.any() else None
        panels = {
            "left image": to_numpy_image(left),
            "right image": to_numpy_image(right),
            "predicted disparity": colorize(disparity, 0.0, vmax),
            "ground-truth disparity": colorize(disparity_gt, 0.0, vmax, mask=valid),
            "absolute error (px)": colorize(error, 0.0, 5.0, mask=valid, cmap="inferno"),
            "confidence exp(matchability)": colorize(output["confidence"], 0.0, 1.0, cmap="viridis"),
        }
        sample_id = str(batch["metadata"].get("sample_id", [index])[0])
        save_evaluation_figure(os.path.join(vis_dir, f"{index:04d}_{sample_id}.png"), panels,
                               title=f"{protocol.name} / {sample_id}")
    print(f"wrote {count} visualisations to {vis_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_checkpoint(args.checkpoint, map_location=device)
    protocol = get_protocol(args.protocol)

    print(f"checkpoint        : {args.checkpoint}")
    print(f"model             : num_disparities={model.num_disparities} downsample={model.scale}")
    print(f"protocol          : {protocol.name}")
    if protocol.notes:
        print(f"protocol notes    : {protocol.notes}")

    if args.both:
        run_once(model, protocol, args, device, PostProcessConfig(enabled=False), "raw")
        run_once(model, protocol, args, device,
                 PostProcessConfig(enabled=True, confidence_threshold=args.postprocess_confidence,
                                   min_region_pixels=args.postprocess_min_region), "postprocessed")
    else:
        postprocess = PostProcessConfig(
            enabled=args.postprocess, confidence_threshold=args.postprocess_confidence,
            min_region_pixels=args.postprocess_min_region)
        run_once(model, protocol, args, device, postprocess,
                 "postprocessed" if args.postprocess else "raw")


if __name__ == "__main__":
    main()
