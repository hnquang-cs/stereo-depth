#!/usr/bin/env python3
"""Label-free stereo training.

    python train.py --config configs/train_unlabeled.yaml
    python train.py --config configs/adapt_unlabeled.yaml --init checkpoints/best.pt

No ground-truth disparity or depth is read anywhere in this path.
"""

from __future__ import annotations

import argparse
import os

import torch

from stereo.config import config_to_yaml, load_config
from stereo.training import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--output", default=None, help="override training.output_dir")
    parser.add_argument("--init", default=None, help="checkpoint to initialise from (Stage 3)")
    parser.add_argument("--resume", default=None, help="checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu or mps")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="dotted config overrides, e.g. teacher.start_epoch=5")
    return parser.parse_args()


def main():
    args = parse_args()

    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = yaml_value(value)

    config = load_config(args.config, overrides)
    if args.output:
        config.training.output_dir = args.output
    if args.init:
        config.training.init_checkpoint = args.init
    if args.resume:
        config.training.resume = args.resume
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.optimizer.learning_rate = args.lr
    if args.seed is not None:
        config.training.seed = args.seed

    device = torch.device(args.device) if args.device else None

    os.makedirs(config.training.output_dir, exist_ok=True)
    snapshot = config_to_yaml(config)
    with open(os.path.join(config.training.output_dir, "config.yaml"), "w") as handle:
        handle.write(snapshot)
    print(snapshot)

    trainer = Trainer(config, device)
    best = trainer.fit()
    print(f"\nbest label-free checkpoint: {best}")
    print("Evaluate it against ground truth with:")
    print(f"  python evaluate.py --checkpoint {best} --protocol sceneflow "
          f"--dataset-root datasets/sceneflow")


def yaml_value(text: str):
    import yaml
    return yaml.safe_load(text)


if __name__ == "__main__":
    main()
