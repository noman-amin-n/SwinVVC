#!/usr/bin/env python3
"""
Stage 7 - Convert block predictions into the per-frame HCPM files VTM reads.

Writes one ``POC_<n>.txt`` per frame, each holding 341 integers per CTU in
raster order.  The sixteen 32x32 entries (lines 6-21 of every CTU record) carry
the predicted class ids; every other entry is ``-2``, which the patched encoder
reads as "no override, keep the encoder's own decision".

The output directory is what the patched ``EncCu.cpp`` points at - see
``docs/vtm_integration.md`` for where to set that path.

Example:
    python scripts/csv_to_txt.py \
        --predictions predictions/ParkRunning3.csv \
        --output vtm_input/ParkRunning3 \
        --width 3840 --height 2160
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from swinvvc.inference.csv_to_txt import convert, validate_frame_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a prediction CSV into per-frame HCPM text files for VTM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Prediction CSV from predict.py.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory to write POC_<n>.txt files into.")
    parser.add_argument("--width", type=int, default=None,
                        help="Source frame width. Given together with --height, "
                             "each output file is validated for completeness.")
    parser.add_argument("--height", type=int, default=None,
                        help="Source frame height (see --width).")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on a block filename that cannot be parsed, "
                             "instead of warning and skipping it.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-frame progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    written = convert(
        csv_path=args.predictions,
        output_dir=args.output,
        strict=args.strict,
        verbose=not args.quiet,
    )
    print(f"\nWrote {len(written)} frame file(s) to {args.output}")

    if args.width and args.height:
        print("\nValidating output against the CTU grid:")
        failures = 0
        for poc in sorted(written):
            ok, message = validate_frame_file(written[poc], args.width, args.height)
            print(f"  {message}")
            failures += 0 if ok else 1

        if failures:
            raise SystemExit(
                f"\n{failures} file(s) have an unexpected line count. The encoder "
                f"would read misaligned partition decisions; check that the block "
                f"directory covered every CTU of every frame."
            )
        print("\nAll frame files have the expected line count.")
    else:
        print("\nPass --width/--height to validate the files against the CTU grid.")


if __name__ == "__main__":
    main()
