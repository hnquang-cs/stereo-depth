#!/usr/bin/env python3
"""Run a trained model on a stereo pair and write disparity / depth.

    python inference.py --checkpoint outputs/train/best.pt --left a.png --right b.png
    python inference.py --checkpoint m.pt --left a.png --right b.png \
                        --focal-length 1075 --baseline 0.12 --postprocess
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from stereo.data.io import read_image
from stereo.geometry import disparity_to_depth
from stereo.postprocess import PostProcessConfig, postprocess_disparity, upsample_confidence
from stereo.utils.checkpoint import build_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--device", default=None)
    parser.add_argument("--focal-length", type=float, default=None, help="pixels, for metric depth")
    parser.add_argument("--baseline", type=float, default=None, help="metres, for metric depth")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--min-region-pixels", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_checkpoint(args.checkpoint, map_location=device).to(device).eval()

    left = torch.from_numpy(read_image(args.left)).permute(2, 0, 1).unsqueeze(0).to(device)
    right = torch.from_numpy(read_image(args.right)).permute(2, 0, 1).unsqueeze(0).to(device)
    if left.shape != right.shape:
        raise SystemExit(f"left {tuple(left.shape)} and right {tuple(right.shape)} must match")

    with torch.no_grad():
        output = model.forward_left(left, right)

    disparity = output["disparity"]
    confidence = upsample_confidence(output["confidence"], disparity.shape[-2:])
    valid = torch.ones_like(disparity)
    if args.postprocess:
        disparity, valid = postprocess_disparity(
            disparity, output["confidence"],
            PostProcessConfig(enabled=True, confidence_threshold=args.confidence_threshold,
                              min_region_pixels=args.min_region_pixels))

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.left))[0]
    np.save(os.path.join(args.output_dir, f"{stem}_disparity.npy"),
            disparity[0, 0].cpu().numpy())
    np.save(os.path.join(args.output_dir, f"{stem}_confidence.npy"),
            confidence[0, 0].cpu().numpy())

    print(f"disparity: min={disparity.min():.2f} max={disparity.max():.2f} mean={disparity.mean():.2f} px")
    print(f"model search range: 0 .. {model.max_disparity} px")
    print(f"valid fraction: {valid.mean():.3f}")

    if args.focal_length is not None and args.baseline is not None:
        depth, depth_valid = disparity_to_depth(disparity, args.focal_length, args.baseline)
        depth = torch.where(depth_valid > 0.5, depth, torch.full_like(depth, float("nan")))
        np.save(os.path.join(args.output_dir, f"{stem}_depth.npy"), depth[0, 0].cpu().numpy())
        finite = depth[torch.isfinite(depth)]
        if finite.numel():
            print(f"depth: min={finite.min():.3f} max={finite.max():.3f} median={finite.median():.3f} m")
    else:
        print("no --focal-length/--baseline given; depth not computed")

    try:
        from stereo.utils.visualization import colorize, save_evaluation_figure, to_numpy_image
        save_evaluation_figure(
            os.path.join(args.output_dir, f"{stem}_visualization.png"),
            {"left": to_numpy_image(left), "right": to_numpy_image(right),
             "disparity": colorize(disparity, 0.0, None),
             "confidence": colorize(confidence, 0.0, 1.0, cmap="viridis")})
    except Exception as error:
        print(f"visualisation skipped: {error}")

    print(f"written to {args.output_dir}")


if __name__ == "__main__":
    main()
