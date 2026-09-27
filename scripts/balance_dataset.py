#!/usr/bin/env python3
"""
Stage 3 - Undersample the dataset to a uniform class distribution, and split off
a validation set.

The natural 32x32 split-mode distribution is dominated by no-split and
quad-tree, so training is performed on a randomly undersampled, uniform subset
(see ``swinvvc.data.balance`` for why undersampling rather than re-weighting).

Example:
    # Balance to the size of the rarest class, then hold out 1500 per class.
    python scripts/balance_dataset.py \
        --source datasets/raw \
        --output datasets/balanced \
        --val-output datasets/validation \
        --val-per-class 1500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from swinvvc.constants import CLASS_NAMES
from swinvvc.data.balance import balance_dataset, count_per_class, split_validation
from swinvvc.utils.plots import plot_class_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class-balance the CU dataset by random undersampling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, type=Path,
                        help="Unbalanced ImageFolder dataset (0/ .. 5/).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Destination for the balanced training set.")
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="Samples to keep per class. Default: the size of "
                             "the smallest class.")
    parser.add_argument("--val-output", type=Path, default=None,
                        help="If set, move a validation split out of --output "
                             "into this directory.")
    parser.add_argument("--val-per-class", type=int, default=1500,
                        help="Validation samples per class (with --val-output).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the sampling RNG.")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying them. Copying "
                             "leaves the unbalanced dataset intact.")
    parser.add_argument("--plot-dir", type=Path, default=None,
                        help="If set, save before/after distribution plots here.")
    return parser.parse_args()


def _report(title: str, counts: dict) -> None:
    print(f"\n{title}")
    for class_id in sorted(counts):
        print(f"  {class_id} ({CLASS_NAMES[class_id]:>3}): {counts[class_id]:>9,}")
    print(f"  {'total':>9}: {sum(counts.values()):>9,}")


def main() -> None:
    args = parse_args()

    original = count_per_class(args.source)
    _report(f"Original distribution ({args.source}):", original)

    balanced = balance_dataset(
        source_root=args.source,
        target_root=args.output,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        move=args.move,
    )
    _report(f"Balanced distribution ({args.output}):", balanced)

    if args.val_output:
        held_out = split_validation(
            source_root=args.output,
            val_root=args.val_output,
            samples_per_class=args.val_per_class,
            seed=args.seed,
        )
        _report(f"Validation split ({args.val_output}):", held_out)
        _report(f"Remaining training set ({args.output}):", count_per_class(args.output))

    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        plot_class_distribution(
            original, "Original Dataset Class Distribution",
            args.plot_dir / "distribution_original.png", color="skyblue",
        )
        plot_class_distribution(
            count_per_class(args.output), "Balanced Dataset Class Distribution",
            args.plot_dir / "distribution_balanced.png", color="lightgreen",
        )
        print(f"\nSaved distribution plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
