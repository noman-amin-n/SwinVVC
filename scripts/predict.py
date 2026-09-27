#!/usr/bin/env python3
"""
Stage 6 - Predict a split mode for every 32x32 block.

Runs one forward pass per block - replacing the encoder's exhaustive
rate-distortion search over the six candidates - and writes a
``filename,predicted_class`` CSV that ``csv_to_txt.py`` turns into the
per-frame HCPM files VTM reads.

Example:
    python scripts/predict.py \
        --checkpoint runs/swin_base/best_model.pth \
        --blocks blocks/ParkRunning3 \
        --output predictions/ParkRunning3.csv
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from swinvvc.config import Config
from swinvvc.constants import CLASS_NAMES
from swinvvc.data.datasets import build_inference_loader
from swinvvc.inference.predict import predict_blocks, write_predictions_csv
from swinvvc.models.swin import build_model_from_config, get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict CU split modes for a directory of 32x32 blocks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Trained weights (.pth).")
    parser.add_argument("--blocks", required=True, type=Path,
                        help="Directory of block PNGs from generate_blocks.py.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Destination prediction CSV.")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config describing the architecture. Defaults "
                             "to the reported architecture.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size; larger is faster than the "
                             "training batch size and does not affect results.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-probabilities", action="store_true",
                        help="Append one softmax column per class to the CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    # Architecture only; the checkpoint supplies the weights.
    config.pretrained = False

    device = get_device(args.device)
    print(f"Device: {device}")

    loader = build_inference_loader(
        args.blocks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=config.input_size,
    )
    print(f"Blocks: {len(loader.dataset):,} from {args.blocks}")

    model = build_model_from_config(config)
    load_checkpoint(model, args.checkpoint, device=device)
    model.to(device)

    started = time.time()
    results = predict_blocks(model, loader, device,
                             return_probabilities=args.save_probabilities)
    elapsed = time.time() - started

    written = write_predictions_csv(
        results, args.output,
        include_probabilities=args.save_probabilities,
        num_classes=config.num_classes,
    )

    print(f"\nPredicted {written:,} block(s) in {elapsed:.1f}s "
          f"({written / max(elapsed, 1e-9):.0f} blocks/s)")
    print(f"Wrote {args.output}")

    distribution = Counter(prediction for _, prediction, _ in results)
    print("\nPredicted class distribution:")
    for class_id in range(len(CLASS_NAMES)):
        count = distribution.get(class_id, 0)
        share = 100.0 * count / written if written else 0.0
        print(f"  {class_id} ({CLASS_NAMES[class_id]:>3}): {count:>9,}  ({share:5.2f}%)")


if __name__ == "__main__":
    main()
