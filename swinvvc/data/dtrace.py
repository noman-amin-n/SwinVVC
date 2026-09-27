"""
Parsing of VTM DTrace block-statistics files into CU boundary maps.

Ground-truth partition labels are not written directly by the encoder.  They
are recovered from the *geometry* of the CUs that the encoder's exhaustive
rate-distortion optimisation (RDO) settled on: a fully RDO-encoded sequence is
traced, every coded CU's position and size is read back, and the CU outlines
are rasterised into a binary boundary map per frame.  The split mode of any
32x32 region can then be read off that map (see `swinvvc.data.labels`).

Expected input
--------------
A VTM trace produced with ``ENABLE_TRACING=1`` and ``BLOCK_STATS_AS_CSV=1``,
i.e. semicolon-separated lines of the form::

    BlockStat;<poc>;<x>;<y>;<width>;<height>;<statistic>;<value>

Only lines carrying the ``Luma_IntraMode`` statistic are used, because those
are emitted once per coded luma CU and therefore enumerate exactly the final
partitioning.  Comment lines (containing ``#``) are skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

#: DTrace statistic that is emitted exactly once per coded luma CU.
CU_STATISTIC = "Luma_IntraMode"

#: Field indices within a `BLOCK_STATS_AS_CSV` line.
_F_POC, _F_X, _F_Y, _F_W, _F_H = 1, 2, 3, 4, 5


def iter_coded_cus(statistics_path: str | Path) -> Iterator[Tuple[int, int, int, int, int]]:
    """
    Yield every coded luma CU in a DTrace statistics file.

    Args:
        statistics_path: path to the DTrace output of a full-RDO encode.

    Yields:
        ``(poc, x, y, width, height)`` per coded luma CU, in trace order.
    """
    with open(statistics_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if CU_STATISTIC not in line or "#" in line:
                continue
            fields = line.split(";")
            yield (
                int(fields[_F_POC]),
                int(fields[_F_X]),
                int(fields[_F_Y]),
                int(fields[_F_W]),
                int(fields[_F_H]),
            )


def build_boundary_maps(
    statistics_path: str | Path,
    width: int,
    height: int,
) -> List[List[List[int]]]:
    """
    Rasterise the CU outlines of every frame into binary boundary maps.

    For each coded CU the four edges of its rectangle are set to 1 in the map
    of the frame it belongs to; everything else stays 0.  The result is the
    partition "wireframe" of the frame.

    The maps are ``(height + 1) x (width + 1)`` so that a CU touching the
    right or bottom edge of the padded frame can still have its far edge
    drawn at ``x + w`` / ``y + h`` without going out of bounds.

    Frames are collected in the order their POCs first appear in the trace;
    a new map is started whenever the POC field changes.

    Args:
        statistics_path: path to the DTrace output of a full-RDO encode.
        width:           padded frame width  (multiple of the block size).
        height:          padded frame height (multiple of the block size).

    Returns:
        A list of boundary maps, indexed by position in the trace.  Each map
        is a list of rows, addressed as ``maps[frame][y][x]``.
    """

    def _blank() -> List[List[int]]:
        return [[0] * (width + 1) for _ in range(height + 1)]

    maps: List[List[List[int]]] = []
    boundary = _blank()
    current_poc = 0

    for poc, x, y, size_x, size_y in iter_coded_cus(statistics_path):
        if poc != current_poc:
            maps.append(boundary)
            boundary = _blank()
            current_poc = poc

        # Top and bottom edges of this CU.
        for i in range(x, x + size_x + 1):
            boundary[y][i] = 1
            boundary[y + size_y][i] = 1
        # Left and right edges of this CU.
        for i in range(y, y + size_y + 1):
            boundary[i][x] = 1
            boundary[i][x + size_x] = 1

    maps.append(boundary)
    return maps
