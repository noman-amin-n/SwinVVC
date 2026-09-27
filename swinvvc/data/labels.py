"""
Recovery of 32x32 partition labels from CU boundary maps.

Given the boundary ("wireframe") map of a frame produced by
`swinvvc.data.dtrace.build_boundary_maps`, the split mode of a 32x32 region is
determined by testing which internal division lines are fully drawn:

* a complete horizontal line at mid-height **and** a complete vertical line at
  mid-width  -> quad-tree split (QT);
* only the horizontal mid-line                -> binary tree horizontal (BTH);
* only the vertical   mid-line                -> binary tree vertical   (BTV);
* complete horizontal lines at 1/4 and 3/4 height -> ternary tree horizontal (TTH);
* complete vertical   lines at 1/4 and 3/4 width  -> ternary tree vertical   (TTV);
* none of the above                            -> no split (NS).

QT is tested first because its signature (both mid-lines present) subsumes the
binary cases, and the binary cases are tested before the ternary ones because
a mid-line is the cheaper discriminator.  This ordering is what makes the six
classes mutually exclusive.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence, Tuple

from ..constants import BLOCK_SIZE, BT_H, BT_V, NO_SPLIT, QT, TT_H, TT_V

BoundaryMap = Sequence[Sequence[int]]


def get_split_mode(boundary: BoundaryMap, x: int, y: int, size_x: int, size_y: int) -> int:
    """
    Classify the partition mode of one region of a boundary map.

    Args:
        boundary: binary boundary map, addressed as ``boundary[y][x]``.
        x:        left edge of the region, in luma samples.
        y:        top edge of the region, in luma samples.
        size_x:   region width  (32 for this work).
        size_y:   region height (32 for this work).

    Returns:
        One of the class ids in `swinvvc.constants` (``NO_SPLIT`` .. ``TT_V``).
    """
    # --- binary / quad signatures: mid-height and mid-width lines ---------- #
    horizontal_mid = True
    for i in range(size_x):
        if boundary[y + size_y // 2][x + i] == 0:
            horizontal_mid = False
            break

    vertical_mid = True
    for i in range(size_y):
        if boundary[y + i][x + size_x // 2] == 0:
            vertical_mid = False
            break

    if horizontal_mid and vertical_mid:
        return QT
    if horizontal_mid:
        return BT_H
    if vertical_mid:
        return BT_V

    # --- ternary signatures: quarter and three-quarter lines --------------- #
    horizontal_thirds = True
    for i in range(size_x):
        if boundary[y + size_y // 4][x + i] == 0:
            horizontal_thirds = False
            break
        if boundary[y + 3 * (size_y // 4)][x + i] == 0:
            horizontal_thirds = False
            break

    vertical_thirds = True
    for i in range(size_y):
        if boundary[y + i][x + size_x // 4] == 0:
            vertical_thirds = False
            break
        if boundary[y + i][x + 3 * (size_x // 4)] == 0:
            vertical_thirds = False
            break

    if horizontal_thirds:
        return TT_H
    if vertical_thirds:
        return TT_V

    return NO_SPLIT


def label_frame(
    boundary: BoundaryMap,
    width: int,
    height: int,
    poc: int,
    block_size: int = BLOCK_SIZE,
) -> List[Tuple[int, int, int, int]]:
    """
    Label every aligned `block_size` x `block_size` region of one frame.

    The scan order is column-major (x outer, y inner), matching the original
    label-generation code so that previously produced CSV files remain
    byte-comparable.  Order does not affect training or inference, since every
    row carries its own coordinates.

    Args:
        boundary:   binary boundary map for this frame.
        width:      padded frame width.
        height:     padded frame height.
        poc:        picture order count recorded in each row.
        block_size: region side length.

    Returns:
        Rows of ``(poc, x, y, mode)``.
    """
    rows: List[Tuple[int, int, int, int]] = []
    for x in range(0, width + 1 - block_size, block_size):
        for y in range(0, height + 1 - block_size, block_size):
            mode = get_split_mode(boundary, x, y, block_size, block_size)
            rows.append((poc, x, y, mode))
    return rows


def label_sequence(
    boundary_maps: Sequence[BoundaryMap],
    width: int,
    height: int,
    block_size: int = BLOCK_SIZE,
) -> List[Tuple[int, int, int, int]]:
    """
    Label every frame of a sequence.  See `label_frame`.

    Returns:
        Rows of ``(poc, x, y, mode)`` for all frames, concatenated.
    """
    rows: List[Tuple[int, int, int, int]] = []
    for poc, boundary in enumerate(boundary_maps):
        rows.extend(label_frame(boundary, width, height, poc, block_size))
    return rows


def write_label_csv(rows: Sequence[Tuple[int, int, int, int]], path: str | Path) -> None:
    """
    Write label rows to a headerless CSV: ``poc,x,y,mode`` per line.

    The file is headerless because it is consumed by `swinvvc.data.blocks`,
    which reads it positionally.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def read_label_csv(path: str | Path) -> List[Tuple[int, int, int, int]]:
    """Read a headerless ``poc,x,y,mode`` label CSV back into tuples."""
    rows: List[Tuple[int, int, int, int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            poc, x, y, mode = (int(v) for v in line.split(","))
            rows.append((poc, x, y, mode))
    return rows
