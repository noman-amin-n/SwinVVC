#!/usr/bin/env python3
"""
Stage 4 - Train the Swin Transformer CU-partition classifier.

Reproduces the training procedure reported in the paper: ImageNet-pretrained
``swin_base_patch4_window7_224``, AdamW (lr 3e-5, weight decay 0.05), cosine
annealing to 1e-6, MixUp (alpha 0.8), EMA weight averaging, gradient clipping
at 1.0, batch size 16, up to 12 epochs with early stopping.

The best **EMA** weights are written to ``<output-dir>/best_model.pth``, and the
resolved configuration and per-epoch history are saved alongside them for
provenance.

Example:
    python scripts/train.py \
        --train-path datasets/balanced \
        --val-path datasets/validation \
        --output-dir runs/swin_base
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

from swinvvc.config import Config
from swinvvc.data.datasets import build_dataloaders
from swinvvc.models.swin import build_model_from_config, get_device
from swinvvc.training.engine import train
from swinvvc.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Swin Transformer CU-partition classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config file; command-line flags override it.")
    parser.add_argument("--train-path", type=str, default=None,
                        help="Balanced training set (ImageFolder, 0/ .. 5/).")
    parser.add_argument("--val-path", type=str, default=None,
                        help="Validation set (ImageFolder, 0/ .. 5/).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for checkpoints, config and history.")
    parser.add_argument("--epochs", type=int, default=None, dest="num_epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="EMA smoothing coefficient beta.")
    parser.add_argument("--mixup-alpha", type=float, default=None)
    parser.add_argument("--no-mixup", action="store_true",
                        help="Disable MixUp (not the reported configuration).")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Force a device, e.g. cuda:0 or cpu.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic cuDNN kernels (slower).")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Load the base config, then apply every explicitly given flag."""
    config = Config.from_yaml(args.config) if args.config else Config()

    overridable = (
        "train_path", "val_path", "output_dir", "num_epochs", "batch_size",
        "lr", "weight_decay", "ema_decay", "mixup_alpha", "num_workers", "seed",
    )
    for name in overridable:
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)

    if args.no_mixup:
        config.use_mixup = False

    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if not config.train_path or not config.val_path:
        raise SystemExit(
            "--train-path and --val-path are required (or set them in --config)."
        )

    set_seed(config.seed, deterministic=args.deterministic)
    device = get_device(args.device)
    print(f"Device: {device}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(output_dir / "config.yaml")

    print("Building data loaders ...")
    train_loader, val_loader = build_dataloaders(config)
    print(f"  train: {len(train_loader.dataset):,} samples")
    print(f"  val  : {len(val_loader.dataset):,} samples")

    print(f"Building model {config.model_name} (pretrained={config.pretrained}) ...")
    model = build_model_from_config(config).to(device)

    print("Starting training ...\n")
    history, best_val_acc = train(model, train_loader, val_loader, config, device)

    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump([dataclasses.asdict(entry) for entry in history], handle, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint : {config.checkpoint_path}")
    print(f"History    : {history_path}")


if __name__ == "__main__":
    main()
