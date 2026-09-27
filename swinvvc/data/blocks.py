"""
Extraction of 32x32 luma blocks from raw YUV sequences.

Two distinct extractions are needed by this project, and they pad differently
on purpose:

* **Training** (`extract_labelled_blocks`) pads to a multiple of the *block*
  size and emits only the blocks named in a label CSV, filed into per-class
  directories.  The result is an ``ImageFolder`` tree.
* **Inference** (`extract_inference_blocks`) pads to a multiple of the *CTU*
  size (128) and emits *every* block of the frame, named with its coordinates.
  CTU-aligned padding is required because the predictions are later folded
  into per-CTU HCPM records, so the block grid must tile whole CTUs.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from ..constants import BLOCK_SIZE, CTU_SIZE, NUM_CLASSES
from .labels import read_label_csv
from .yuv import get_padded_size, pad_frame, read_y_frame


def _make_class_dirs(root: str | Path, num_classes: int = NUM_CLASSES) -> None:
    """Create the ``0/`` .. ``<num_classes-1>/`` ImageFolder subdirectories."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for class_id in range(num_classes):
        (root / str(class_id)).mkdir(exist_ok=True)


def extract_labelled_blocks(
    yuv_path: str | Path,
    label_csv: str | Path,
    output_dir: str | Path,
    width: int,
    height: int,
    sequence_name: str,
    block_size: int = BLOCK_SIZE,
) -> Dict[int, int]:
    """
    Cut the 32x32 blocks listed in a label CSV and file them by class.

    Every row of `label_csv` names one block by ``(poc, x, y)`` and gives its
    ground-truth class.  The corresponding luma samples are written as an
    8-bit PNG to ``<output_dir>/<class>/<sequence_name>_frame_<poc>_x<x>_y<y>.png``.

    Rows are grouped by POC first so that each frame is read from disk exactly
    once, which matters for 4K sequences.

    Args:
        yuv_path:      raw 8-bit 4:2:0 YUV file.
        label_csv:     headerless ``poc,x,y,mode`` CSV from `swinvvc.data.labels`.
        output_dir:    root of the ImageFolder tree to populate.
        width:         source frame width.
        height:        source frame height.
        sequence_name: prefix for output filenames; keeps blocks from
                       different sequences distinguishable in one tree.
        block_size:    block side length.

    Returns:
        Number of blocks written per class id.
    """
    rows_by_poc: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for poc, x, y, mode in read_label_csv(label_csv):
        rows_by_poc[poc].append((x, y, mode))

    _make_class_dirs(output_dir)
    padded_w, padded_h = get_padded_size(width, height, block_size, block_size)

    counts: Dict[int, int] = {class_id: 0 for class_id in range(NUM_CLASSES)}

    for poc in sorted(rows_by_poc):
        frame = pad_frame(read_y_frame(yuv_path, width, height, poc), padded_w, padded_h)

        for x, y, mode in rows_by_poc[poc]:
            block = frame[y : y + block_size, x : x + block_size]
            out_path = Path(output_dir) / str(mode) / (
                f"{sequence_name}_frame_{poc}_x{x}_y{y}.png"
            )
            cv2.imwrite(str(out_path), block)
            counts[mode] = counts.get(mode, 0) + 1

    return counts


def extract_inference_blocks(
    yuv_path: str | Path,
    output_dir: str | Path,
    width: int,
    height: int,
    sequence_name: str,
    start_poc: int = 0,
    end_poc: int = 0,
    block_size: int = BLOCK_SIZE,
) -> int:
    """
    Cut *every* 32x32 luma block of a POC range, for inference.

    The frame is first zero-padded to a multiple of the CTU size (128), then
    tiled exhaustively into `block_size` blocks in raster order.  Each block is
    written as ``<sequence_name>_poc<poc>_x<x>_y<y>.png``; the coordinates in
    the filename are what `swinvvc.inference.csv_to_txt` later parses to place
    each prediction in its CTU's HCPM record, so the naming is part of the
    pipeline contract and must not be changed in isolation.

    Args:
        yuv_path:      raw 8-bit 4:2:0 YUV file.
        output_dir:    directory to write block PNGs into (created if absent).
        width:         source frame width.
        height:        source frame height.
        sequence_name: filename prefix.
        start_poc:     first frame to process (inclusive).
        end_poc:       last  frame to process (inclusive).
        block_size:    block side length.

    Returns:
        Total number of blocks written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CTU-aligned padding: the block grid must tile whole 128x128 CTUs.
    padded_w, padded_h = get_padded_size(width, height, CTU_SIZE, CTU_SIZE)

    written = 0
    for poc in range(start_poc, end_poc + 1):
        frame = pad_frame(read_y_frame(yuv_path, width, height, poc), padded_w, padded_h)

        for y in range(0, padded_h, block_size):
            for x in range(0, padded_w, block_size):
                block = frame[y : y + block_size, x : x + block_size]
                out_path = output_dir / f"{sequence_name}_poc{poc}_x{x}_y{y}.png"
                cv2.imwrite(str(out_path), block)
                written += 1

    return written
