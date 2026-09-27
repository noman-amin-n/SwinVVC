#!/usr/bin/env python3
"""
Stage 1 - Extract ground-truth 32x32 partition labels from a VTM DTrace file.

Reads the block-statistics trace of a *full-RDO* VTM encode, rebuilds the CU
boundary map of each frame, classifies every aligned 32x32 region into one of
the six split modes, and writes the result as a headerless
``poc,x,y,mode`` CSV.

The trace must come from a VTM build with ``ENABLE_TRACING=1`` and
``BLOCK_STATS_AS_CSV=1`` in ``source/Lib/CommonLib/TypeDef.h``.  See
``docs/dataset_generation.md``.

Example:
    python scripts/extract_labels.py \
        --statistics traces/BlowingBubbles_416x240_50_qp22.csv \
        --width 416 --height 240 \
        --output labels/BlowingBubbles_416x240_50_qp22.csv \
        --render-dir figures/BlowingBubbles
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401  (side effect: sys.path)

from swinvvc.constants import BLOCK_SIZE, CLASS_NAMES
from swinvvc.data.dtrace import build_boundary_maps
from swinvvc.data.labels import label_sequence, write_label_csv
from swinvvc.data.visualize import render_label_map, render_wireframe
from swinvvc.data.yuv import get_padded_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 32x32 CU partition labels from a VTM DTrace file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--statistics", required=True, type=Path,
                        help="VTM DTrace block-statistics file (CSV format).")
    parser.add_argument("--width", required=True, type=int,
                        help="Source frame width in luma samples.")
    parser.add_argument("--height", required=True, type=int,
                        help="Source frame height in luma samples.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Destination label CSV (poc,x,y,mode).")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help="CU side length to label.")
    parser.add_argument("--render-dir", type=Path, default=None,
                        help="If set, save partition wireframes and label maps here "
                             "for visual verification.")
    parser.add_argument("--render-limit", type=int, default=1,
                        help="Number of frames to render when --render-dir is given.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Labels live on the block grid, so pad to a multiple of the block size.
    padded_w, padded_h = get_padded_size(args.width, args.height,
                                         args.block_size, args.block_size)
    print(f"Source {args.width}x{args.height} -> padded {padded_w}x{padded_h}")

    print(f"Parsing {args.statistics} ...")
    boundary_maps = build_boundary_maps(args.statistics, padded_w, padded_h)
    print(f"Recovered boundary maps for {len(boundary_maps)} frame(s).")

    rows = label_sequence(boundary_maps, padded_w, padded_h, args.block_size)
    write_label_csv(rows, args.output)
    print(f"Wrote {len(rows)} labels to {args.output}")

    distribution = Counter(mode for _, _, _, mode in rows)
    print("\nClass distribution:")
    for class_id in range(len(CLASS_NAMES)):
        count = distribution.get(class_id, 0)
        share = 100.0 * count / len(rows) if rows else 0.0
        print(f"  {class_id} ({CLASS_NAMES[class_id]:>3}): {count:>9,}  ({share:5.2f}%)")

    if args.render_dir:
        args.render_dir.mkdir(parents=True, exist_ok=True)
        for poc, boundary in enumerate(boundary_maps[: args.render_limit]):
            render_wireframe(boundary, padded_w, padded_h,
                             args.render_dir / f"wireframe_poc{poc}.png")
            render_label_map(boundary, padded_w, padded_h,
                             args.render_dir / f"labels_poc{poc}.png",
                             block_size=args.block_size)
        print(f"\nRendered {min(args.render_limit, len(boundary_maps))} frame(s) "
              f"to {args.render_dir}")


if __name__ == "__main__":
    main()
