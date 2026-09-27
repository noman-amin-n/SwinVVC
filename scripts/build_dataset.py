#!/usr/bin/env python3
"""
Stage 2 - Cut labelled 32x32 blocks out of the source YUV into an ImageFolder tree.

Consumes the label CSV from ``extract_labels.py`` and writes one PNG per
labelled block into ``<output>/<class>/``, which is the layout
``torchvision.datasets.ImageFolder`` expects.

Repeated invocations with different sequences and the same ``--output``
accumulate into one dataset; the ``--sequence-name`` prefix keeps filenames
unique across sequences.

Example:
    python scripts/build_dataset.py \
        --yuv sequences/BlowingBubbles_416x240_50.yuv \
        --labels labels/BlowingBubbles_416x240_50_qp22.csv \
        --width 416 --height 240 \
        --sequence-name BlowingBubbles_416x240_50 \
        --output datasets/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from swinvvc.constants import BLOCK_SIZE, CLASS_NAMES
from swinvvc.data.blocks import extract_labelled_blocks
from swinvvc.data.yuv import count_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract labelled 32x32 luma blocks into an ImageFolder tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yuv", required=True, type=Path,
                        help="Raw 8-bit 4:2:0 YUV source file.")
    parser.add_argument("--labels", required=True, type=Path,
                        help="Label CSV from extract_labels.py.")
    parser.add_argument("--width", required=True, type=int,
                        help="Source frame width in luma samples.")
    parser.add_argument("--height", required=True, type=int,
                        help="Source frame height in luma samples.")
    parser.add_argument("--sequence-name", required=True,
                        help="Filename prefix; keeps blocks from different "
                             "sequences distinguishable in a shared tree.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Root of the ImageFolder tree to populate.")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help="Block side length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    available = count_frames(args.yuv, args.width, args.height)
    print(f"{args.yuv.name}: {available} frame(s) at {args.width}x{args.height}")

    print(f"Extracting blocks into {args.output} ...")
    counts = extract_labelled_blocks(
        yuv_path=args.yuv,
        label_csv=args.labels,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        sequence_name=args.sequence_name,
        block_size=args.block_size,
    )

    total = sum(counts.values())
    print(f"\nWrote {total:,} block(s):")
    for class_id in sorted(counts):
        print(f"  {class_id} ({CLASS_NAMES[class_id]:>3}): {counts[class_id]:>9,}")


if __name__ == "__main__":
    main()
