#!/usr/bin/env python3
"""
Stage 4b - Evaluate a trained checkpoint on a labelled split.

Reports the confusion matrix, per-class accuracy / precision / recall /
specificity / F1, the weighted averages, and one-vs-rest AUC, and optionally
saves the corresponding figures.

Example:
    python scripts/evaluate.py \
        --checkpoint runs/swin_base/best_model.pth \
        --val-path datasets/validation \
        --plot-dir runs/swin_base/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from swinvvc.config import Config
from swinvvc.constants import CLASS_NAMES
from swinvvc.data.datasets import build_transform
from swinvvc.models.swin import build_model_from_config, get_device, load_checkpoint
from swinvvc.training.evaluate import collect_predictions, compute_metrics, format_report
from swinvvc.utils.plots import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_roc_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CU-partition classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Trained weights (.pth).")
    parser.add_argument("--val-path", required=True, type=str,
                        help="Labelled split to evaluate (ImageFolder, 0/ .. 5/).")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config describing the architecture. Defaults "
                             "to the reported architecture.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None,
                        help="If set, save confusion matrix, per-class accuracy "
                             "and ROC figures here.")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="If set, write the metrics to this JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    config.val_path = args.val_path
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    # Architecture only; the checkpoint supplies the weights.
    config.pretrained = False

    device = get_device(args.device)
    print(f"Device: {device}")

    from torch.utils.data import DataLoader
    from torchvision import datasets

    dataset = datasets.ImageFolder(root=config.val_path,
                                   transform=build_transform(config.input_size))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers, pin_memory=True)
    print(f"Evaluating on {len(dataset):,} samples from {config.val_path}")

    model = build_model_from_config(config)
    load_checkpoint(model, args.checkpoint, device=device)
    model.to(device)

    labels, predictions, probabilities = collect_predictions(model, loader, device)
    metrics = compute_metrics(labels, predictions, probabilities, CLASS_NAMES)

    print()
    print(format_report(metrics, CLASS_NAMES))

    if args.json_out:
        serialisable = {
            "accuracy": metrics["accuracy"],
            "per_class": metrics["per_class"],
            "weighted": metrics["weighted"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "auc": {CLASS_NAMES[k]: v["auc"] for k, v in metrics["roc"].items()},
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)
        print(f"\nWrote metrics to {args.json_out}")

    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES,
                              out_path=args.plot_dir / "confusion_matrix.png")
        plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, normalize=True,
                              out_path=args.plot_dir / "confusion_matrix_normalised.png")
        plot_per_class_accuracy(metrics["confusion_matrix"], CLASS_NAMES,
                                out_path=args.plot_dir / "per_class_accuracy.png")
        plot_roc_curves(metrics["roc"], CLASS_NAMES,
                        out_path=args.plot_dir / "roc_curves.png")
        print(f"Saved figures to {args.plot_dir}")


if __name__ == "__main__":
    main()
