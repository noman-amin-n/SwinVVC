#!/usr/bin/env python3
"""
Stage 5 - Cut a sequence into 32x32 luma blocks for inference.

Zero-pads each frame to a multiple of the 128x128 CTU size, then tiles it
exhaustively into 32x32 blocks named ``<sequence>_poc<POC>_x<X>_y<Y>.png``.
The coordinates in those names are how ``csv_to_txt.py`` later places each
prediction into its CTU's HCPM record, so the naming must not be changed on
its own.

CTU-aligned (not block-aligned) padding is required here: the encoder walks a
whole number of CTUs, so the block grid must tile complete CTUs.

Example:
    python scripts/generate_blocks.py \
        --yuv sequences/ParkRunning3_3840x2160_50.yuv \
        --width 3840 --height 2160 \
        --start-poc 0 --end-poc 0 \
        --output blocks/ParkRunning3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from swinvvc.constants import BLOCK_SIZE, CTU_SIZE
from swinvvc.data.blocks import extract_inference_blocks
from swinvvc.data.yuv import count_frames, get_padded_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut a YUV sequence into 32x32 luma blocks for inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yuv", required=True, type=Path,
                        help="Raw 8-bit 4:2:0 YUV source file.")
    parser.add_argument("--width", required=True, type=int,
                        help="Source frame width in luma samples.")
    parser.add_argument("--height", required=True, type=int,
                        help="Source frame height in luma samples.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory to write the block PNGs into.")
    parser.add_argument("--sequence-name", default=None,
                        help="Filename prefix. Defaults to the YUV file stem.")
    parser.add_argument("--start-poc", type=int, default=0,
                        help="First frame to process (inclusive).")
    parser.add_argument("--end-poc", type=int, default=0,
                        help="Last frame to process (inclusive).")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help="Block side length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_name = args.sequence_name or args.yuv.stem

    available = count_frames(args.yuv, args.width, args.height)
    if args.end_poc >= available:
        raise SystemExit(
            f"--end-poc {args.end_poc} is out of range: {args.yuv.name} holds "
            f"{available} frame(s) at {args.width}x{args.height} (POC 0..{available - 1})."
        )

    padded_w, padded_h = get_padded_size(args.width, args.height, CTU_SIZE, CTU_SIZE)
    blocks_per_frame = (padded_w // args.block_size) * (padded_h // args.block_size)
    frames = args.end_poc - args.start_poc + 1

    print(f"Sequence      : {sequence_name}")
    print(f"Source size   : {args.width}x{args.height}")
    print(f"CTU-padded    : {padded_w}x{padded_h}")
    print(f"Frames        : POC {args.start_poc}..{args.end_poc} ({frames})")
    print(f"Blocks/frame  : {blocks_per_frame:,}")
    print(f"Blocks total  : {blocks_per_frame * frames:,}")
    print(f"Output        : {args.output}\n")

    written = extract_inference_blocks(
        yuv_path=args.yuv,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        sequence_name=sequence_name,
        start_poc=args.start_poc,
        end_poc=args.end_poc,
        block_size=args.block_size,
    )
    print(f"Wrote {written:,} block image(s) to {args.output}")


if __name__ == "__main__":
    main()
