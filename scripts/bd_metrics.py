#!/usr/bin/env python3
"""
Stage 8 - Compute BD-BR, BD-PSNR, encoding-time savings and RD plots.

Takes the anchor (unmodified VTM) and proposed-method measurements for one
sequence across a set of QPs, and reports the Bjontegaard deltas plus the
average time saving.  Input can be given on the command line or in a JSON file.

JSON format:
    {
      "sequence": "BasketballPass_416x240_50",
      "qps":    [22, 27, 32, 37],
      "anchor":   {"rate": [...], "psnr": [...], "time": [...],
                   "ssim": [...], "vmaf": [...]},
      "proposed": {"rate": [...], "psnr": [...], "time": [...],
                   "ssim": [...], "vmaf": [...]}
    }
`rate` is in kbps, `psnr` in dB, `time` in seconds.  `ssim` and `vmaf` are
optional; when present, their mean relative differences are reported too.

Example:
    python scripts/bd_metrics.py --json results/BasketballPass.json \
        --plot-dir results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import _bootstrap  # noqa: F401

from swinvvc.utils.bjontegaard import bd_psnr, bd_rate, relative_difference, time_saving
from swinvvc.utils.plots import plot_encoding_time, plot_rd_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Bjontegaard metrics and encoding-time savings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json", type=Path, default=None,
                        help="JSON file holding the measurements (see module docstring).")
    parser.add_argument("--sequence", default="", help="Sequence name, for plot titles.")
    parser.add_argument("--qps", type=int, nargs="+", default=None)
    parser.add_argument("--anchor-rate", type=float, nargs="+", default=None,
                        help="Anchor bitrates in kbps.")
    parser.add_argument("--anchor-psnr", type=float, nargs="+", default=None,
                        help="Anchor Y-PSNR in dB.")
    parser.add_argument("--anchor-time", type=float, nargs="+", default=None,
                        help="Anchor encoding times in seconds.")
    parser.add_argument("--proposed-rate", type=float, nargs="+", default=None)
    parser.add_argument("--proposed-psnr", type=float, nargs="+", default=None)
    parser.add_argument("--proposed-time", type=float, nargs="+", default=None)
    parser.add_argument("--plot-dir", type=Path, default=None,
                        help="If set, save RD and encoding-time plots here.")
    return parser.parse_args()


def load_measurements(args: argparse.Namespace) -> dict:
    """Assemble the measurement dict from a JSON file or the command line."""
    if args.json:
        with open(args.json, "r", encoding="utf-8") as handle:
            return json.load(handle)

    required = (
        args.anchor_rate, args.anchor_psnr, args.anchor_time,
        args.proposed_rate, args.proposed_psnr, args.proposed_time,
    )
    if any(value is None for value in required):
        raise SystemExit(
            "Provide --json, or all of --anchor-rate/--anchor-psnr/--anchor-time "
            "and --proposed-rate/--proposed-psnr/--proposed-time."
        )

    return {
        "sequence": args.sequence,
        "qps": args.qps or list(range(len(args.anchor_rate))),
        "anchor": {"rate": args.anchor_rate, "psnr": args.anchor_psnr,
                   "time": args.anchor_time},
        "proposed": {"rate": args.proposed_rate, "psnr": args.proposed_psnr,
                     "time": args.proposed_time},
    }


def main() -> None:
    args = parse_args()
    data = load_measurements(args)

    sequence = data.get("sequence", args.sequence)
    qps: List[int] = data.get("qps", [])
    anchor, proposed = data["anchor"], data["proposed"]

    print(f"Sequence: {sequence or '(unnamed)'}")
    print(f"QPs     : {qps}\n")

    bdbr = bd_rate(anchor["rate"], anchor["psnr"], proposed["rate"], proposed["psnr"])
    bdpsnr = bd_psnr(anchor["rate"], anchor["psnr"], proposed["rate"], proposed["psnr"])
    times = time_saving(anchor["time"], proposed["time"])

    print(f"BD-BR            : {bdbr:+.4f} %   (positive = more bits at equal quality)")
    print(f"BD-PSNR          : {bdpsnr:+.4f} dB (negative = quality loss at equal rate)")
    print()
    for qp, saving in zip(qps, times["per_qp"]):
        print(f"  QP {qp:>2}: time saving {saving:6.2f} %")
    print(f"\nAverage dT       : {times['average']:.2f} %")

    if bdbr != 0:
        # Time saved per 1% of bitrate overhead; the trade-off figure reported
        # in the paper's comparison table.
        print(f"dT / BD-BR       : {times['average'] / bdbr:.2f}")

    for name in ("ssim", "vmaf"):
        if name in anchor and name in proposed:
            delta = relative_difference(anchor[name], proposed[name])
            print(f"Average d{name.upper():<4}    : {delta['average']:.4f} %")

    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        stem = sequence or "sequence"

        plot_rd_curve(anchor["rate"], anchor["psnr"],
                      proposed["rate"], proposed["psnr"],
                      metric_name="Y-PSNR (dB)", title=sequence,
                      out_path=args.plot_dir / f"rd_psnr_{stem}.pdf")

        for name, label in (("ssim", "SSIM"), ("vmaf", "VMAF")):
            if name in anchor and name in proposed:
                plot_rd_curve(anchor["rate"], anchor[name],
                              proposed["rate"], proposed[name],
                              metric_name=label, title=sequence,
                              out_path=args.plot_dir / f"rd_{name}_{stem}.pdf")

        plot_encoding_time(qps, anchor["time"], proposed["time"], title=sequence,
                           out_path=args.plot_dir / f"time_{stem}.pdf")
        print(f"\nSaved plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
