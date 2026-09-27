"""Shared pytest fixtures and path setup."""

import sys
from pathlib import Path

import pytest

# Make the package importable from a source checkout without installing it.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def synthetic_trace(tmp_path):
    """
    Build a DTrace statistics file with known 32x32 partitions.

    For every 32x32 region a split mode is chosen deterministically, and the
    child CUs that the mode implies are emitted as trace records - exactly what
    a real full-RDO encode would produce. The ground truth is returned so that
    label recovery can be checked against it.

    Returns:
        ``(path, width, height, {(poc, x, y): mode})``
    """
    import random

    width, height = 256, 256  # 2x2 CTUs
    rng = random.Random(7)
    truth = {}
    path = tmp_path / "stats.csv"

    def emit(handle, poc, x, y, w, h):
        handle.write(f"BlockStat;{poc};{x:4d};{y:4d};{w:2d};{h:2d};Luma_IntraMode;7\n")

    with open(path, "w", encoding="utf-8") as handle:
        # A comment line that also contains the statistic name, to confirm that
        # comment filtering takes precedence over the name match.
        handle.write("# header mentioning Luma_IntraMode must be skipped\n")
        for poc in (0, 1):
            for by in range(0, height, 32):
                for bx in range(0, width, 32):
                    mode = rng.choice([0, 1, 2, 3, 4, 5])
                    truth[(poc, bx, by)] = mode

                    if mode == 0:  # NS
                        emit(handle, poc, bx, by, 32, 32)
                    elif mode == 1:  # QT
                        for dy in (0, 16):
                            for dx in (0, 16):
                                emit(handle, poc, bx + dx, by + dy, 16, 16)
                    elif mode == 2:  # BTH
                        for dy in (0, 16):
                            emit(handle, poc, bx, by + dy, 32, 16)
                    elif mode == 3:  # BTV
                        for dx in (0, 16):
                            emit(handle, poc, bx + dx, by, 16, 32)
                    elif mode == 4:  # TTH
                        emit(handle, poc, bx, by, 32, 8)
                        emit(handle, poc, bx, by + 8, 32, 16)
                        emit(handle, poc, bx, by + 24, 32, 8)
                    elif mode == 5:  # TTV
                        emit(handle, poc, bx, by, 8, 32)
                        emit(handle, poc, bx + 8, by, 16, 32)
                        emit(handle, poc, bx + 24, by, 8, 32)

    return path, width, height, truth


@pytest.fixture
def synthetic_predictions(tmp_path):
    """
    Build a prediction CSV covering every block of a 256x256, 2-frame sequence.

    Returns:
        ``(path, width, height)``
    """
    import csv

    width, height = 256, 256
    path = tmp_path / "preds.csv"

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "predicted_class"])
        for poc in (0, 1):
            for y in range(0, height, 32):
                for x in range(0, width, 32):
                    writer.writerow([f"Seq_poc{poc}_x{x}_y{y}.png", (x // 32 + y // 32 + poc) % 6])

    return path, width, height
