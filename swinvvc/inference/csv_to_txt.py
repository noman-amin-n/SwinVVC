"""
Conversion of block predictions into the per-frame HCPM files VTM reads.

The patched encoder does not read the prediction CSV directly.  For each frame
it expects a plain text file of one integer per line, holding a fixed-size
Hierarchical CU Partition Map (HCPM) record for every CTU in raster order:

    CTU 0 : 341 lines
    CTU 1 : 341 lines
    ...

Within one CTU record the 341 entries are ordered by CU size, each level in
raster order within the CTU::

    line   1        (index   0)  : the 128x128 CU
    lines  2 -   5  (index   1-  4): the four  64x64  CUs
    lines  6 -  21  (index   5- 20): the sixteen 32x32 CUs   <-- predicted here
    lines 22 -  85  (index  21- 84): the 64  16x16 CUs
    lines 86 - 341  (index  85-340): the 256   8x8  CUs

This model predicts the 32x32 level only, so entries 5..20 carry class ids
0..5 and every other entry is filled with ``-2``.  The encoder maps ``-2`` to
``ETM_INTRA``, which means "no override, keep the encoder's own decision", so
the untouched levels continue to use full rate-distortion optimisation.

The mapping from a block's pixel coordinates to its HCPM slot is::

    ctu_x, ctu_y           = x // 128, y // 128          # which CTU
    subblock_col, row      = (x % 128) // 32, (y % 128) // 32   # slot in CTU
    index                  = 5 + subblock_row * 4 + subblock_col

which mirrors ``indexIn21Data = 5 + uhLeftInCTU / 32 + uhTopInCTU / 32 * 4``
in the patched ``EncCu.cpp``.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..constants import (
    BLOCK_SIZE,
    BLOCKS_PER_CTU,
    CTU_SIZE,
    HCPM_ENTRIES_PER_CTU,
    HCPM_OFFSET_32,
    HCPM_UNUSED,
)

#: Matches the block filenames produced by
#: `swinvvc.data.blocks.extract_inference_blocks`:
#: ``<sequence>_poc<POC>_x<X>_y<Y>.png``
FILENAME_PATTERN = re.compile(r"poc(\d+).*_x(\d+)_y(\d+)\.png", re.IGNORECASE)

#: ``data[poc][(ctu_y, ctu_x)] = [(subblock_row, subblock_col, class_id), ...]``
PredictionIndex = Dict[int, Dict[Tuple[int, int], List[Tuple[int, int, int]]]]


def parse_block_filename(filename: str) -> Optional[Tuple[int, int, int]]:
    """
    Extract ``(poc, x, y)`` from a block filename.

    Returns:
        The triple, or ``None`` when the name does not match the expected
        pattern (the caller decides whether to skip or fail).
    """
    match = FILENAME_PATTERN.search(filename)
    if match is None:
        return None
    poc, x, y = match.groups()
    return int(poc), int(x), int(y)


def index_predictions(csv_path: str | Path, strict: bool = False) -> PredictionIndex:
    """
    Group predictions by frame and by the CTU each block belongs to.

    Args:
        csv_path: prediction CSV with ``filename`` and ``predicted_class``
                  columns, as written by `swinvvc.inference.predict`.
        strict:   raise on a filename that does not match the expected
                  pattern, instead of warning and skipping it.

    Returns:
        A `PredictionIndex`.

    Raises:
        ValueError: if `strict` and a filename cannot be parsed, or if the CSV
                    lacks the required columns.
    """
    data: PredictionIndex = {}

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"filename", "predicted_class"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing column(s): {sorted(missing)}")

        for row in reader:
            filename = row["filename"]
            parsed = parse_block_filename(filename)
            if parsed is None:
                message = f"Filename {filename!r} does not match the expected pattern"
                if strict:
                    raise ValueError(message)
                print(f"Warning: {message}; skipping.")
                continue

            poc, x, y = parsed
            class_id = int(row["predicted_class"])

            # Which CTU this 32x32 block lives in, and where inside it.
            ctu_key = (y // CTU_SIZE, x // CTU_SIZE)
            subblock_col = (x % CTU_SIZE) // BLOCK_SIZE
            subblock_row = (y % CTU_SIZE) // BLOCK_SIZE

            data.setdefault(poc, {}).setdefault(ctu_key, []).append(
                (subblock_row, subblock_col, class_id)
            )

    return data


def build_frame_lines(
    ctu_predictions: Dict[Tuple[int, int], List[Tuple[int, int, int]]],
    poc: int,
    verbose: bool = True,
) -> List[str]:
    """
    Assemble the HCPM lines for one frame.

    CTUs are emitted in raster order (by ``ctu_y`` then ``ctu_x``), which is
    the order of ``ctuRsAddr`` in the encoder, and within each CTU the sixteen
    32x32 predictions are written in raster order.

    Args:
        ctu_predictions: this frame's slice of a `PredictionIndex`.
        poc:             frame index, used only in warning messages.
        verbose:         warn about CTUs that do not carry all 16 predictions.

    Returns:
        One string per output line.
    """
    lines: List[str] = []

    for ctu_key in sorted(ctu_predictions, key=lambda key: (key[0], key[1])):
        subblocks = sorted(ctu_predictions[ctu_key], key=lambda item: (item[0], item[1]))

        if verbose and len(subblocks) != BLOCKS_PER_CTU:
            print(
                f"Warning: frame poc{poc}, CTU {ctu_key} has {len(subblocks)} "
                f"sub-blocks, expected {BLOCKS_PER_CTU}"
            )

        # Start from an all-"no override" record, then fill the 32x32 level.
        record = [str(HCPM_UNUSED)] * HCPM_ENTRIES_PER_CTU
        for offset, (_, _, class_id) in enumerate(subblocks[:BLOCKS_PER_CTU]):
            record[HCPM_OFFSET_32 + offset] = str(class_id)

        lines.extend(record)

    return lines


def convert(
    csv_path: str | Path,
    output_dir: str | Path,
    strict: bool = False,
    verbose: bool = True,
) -> Dict[int, Path]:
    """
    Convert a prediction CSV into one HCPM text file per frame.

    Output files are named ``POC_<n>.txt``, which is the name the patched
    encoder builds with ``sprintf(phFileName, "<dir>/POC_%d.txt", poc)``.

    Args:
        csv_path:   prediction CSV from `swinvvc.inference.predict`.
        output_dir: directory to write the per-frame files into.
        strict:     fail on unparseable filenames instead of skipping them.
        verbose:    print per-frame progress and completeness warnings.

    Returns:
        Mapping from POC to the path written.
    """
    data = index_predictions(csv_path, strict=strict)
    if not data:
        raise ValueError(f"No usable predictions found in {csv_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[int, Path] = {}
    for poc in sorted(data):
        lines = build_frame_lines(data[poc], poc, verbose=verbose)
        out_path = output_dir / f"POC_{poc}.txt"
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        written[poc] = out_path
        if verbose:
            ctu_count = len(data[poc])
            print(f"Wrote POC {poc}: {ctu_count} CTUs, {len(lines)} lines -> {out_path}")

    return written


def validate_frame_file(
    txt_path: str | Path,
    width: int,
    height: int,
) -> Tuple[bool, str]:
    """
    Check that an HCPM file has exactly one complete record per CTU.

    A short or long file is the most common cause of the encoder reading
    garbage partition decisions, so this is worth running once per sequence
    before launching a long encode.

    Args:
        txt_path: HCPM file to check.
        width:    source frame width  in luma samples.
        height:   source frame height in luma samples.

    Returns:
        ``(ok, message)``.
    """
    ctus_wide = (width + CTU_SIZE - 1) // CTU_SIZE
    ctus_high = (height + CTU_SIZE - 1) // CTU_SIZE
    expected = ctus_wide * ctus_high * HCPM_ENTRIES_PER_CTU

    with open(txt_path, "r", encoding="utf-8") as handle:
        actual = sum(1 for line in handle if line.strip())

    if actual != expected:
        return False, (
            f"{txt_path}: expected {expected} lines "
            f"({ctus_wide}x{ctus_high} CTUs x {HCPM_ENTRIES_PER_CTU}), found {actual}"
        )
    return True, f"{txt_path}: OK ({actual} lines, {ctus_wide}x{ctus_high} CTUs)"
