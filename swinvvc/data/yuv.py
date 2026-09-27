"""
Raw YUV I/O helpers.

All sequences used in this work are 8-bit 4:2:0 planar YUV.  Only the luma
(Y) plane is ever needed: the CU-partition decision the model predicts is
made on luma in VVC, and the encoder patch overrides luma partitioning only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def get_padded_size(width: int, height: int, block_w: int, block_h: int) -> Tuple[int, int]:
    """
    Round `width` / `height` up to the next multiple of the block dimensions.

    Args:
        width:   source frame width in luma samples.
        height:  source frame height in luma samples.
        block_w: alignment in the horizontal direction.
        block_h: alignment in the vertical direction.

    Returns:
        ``(padded_width, padded_height)``.

    Example:
        >>> get_padded_size(416, 240, 128, 128)
        (512, 256)
    """
    padded_w = (width + block_w - 1) // block_w * block_w
    padded_h = (height + block_h - 1) // block_h * block_h
    return padded_w, padded_h


def read_y_frame(
    yuv_path: str | Path,
    width: int,
    height: int,
    frame_number: int = 0,
) -> np.ndarray:
    """
    Read the luma plane of a single frame from an 8-bit 4:2:0 YUV file.

    A 4:2:0 frame occupies ``1.5 * width * height`` bytes (the Y plane
    followed by half-resolution Cb and Cr planes), so frame `n` starts at
    ``n * 1.5 * width * height``.  Only the leading ``width * height`` bytes
    of that frame are read.

    If the file ends early the remainder is zero-filled rather than raising,
    which keeps block extraction robust at the tail of a truncated sequence.

    Args:
        yuv_path:     path to the raw YUV file.
        width:        frame width in luma samples.
        height:       frame height in luma samples.
        frame_number: zero-based picture order count (POC) to read.

    Returns:
        ``uint8`` array of shape ``(height, width)``.
    """
    frame_size = width * height
    offset = int(frame_number * frame_size * 1.5)

    with open(yuv_path, "rb") as handle:
        handle.seek(offset)
        y_data = handle.read(frame_size)

    if len(y_data) < frame_size:
        y_data = y_data + b"\x00" * (frame_size - len(y_data))

    # `frombuffer` returns a read-only view; copy so callers may pad in place.
    return np.frombuffer(y_data, dtype=np.uint8).reshape((height, width)).copy()


def pad_frame(frame: np.ndarray, padded_w: int, padded_h: int) -> np.ndarray:
    """
    Zero-pad a luma frame to ``(padded_h, padded_w)``, anchored top-left.

    This mirrors what VVC does at the picture boundary: the CTU grid always
    covers a whole number of CTUs, and samples beyond the picture are not
    coded.  Padding with zeros keeps the block grid aligned with the encoder's
    CTU raster so that HCPM indices computed here match the encoder's.

    Args:
        frame:    ``(height, width)`` luma array.
        padded_w: target width  (must be >= frame width).
        padded_h: target height (must be >= frame height).

    Returns:
        The frame itself when no padding is needed, otherwise a new array.
    """
    height, width = frame.shape
    if padded_w == width and padded_h == height:
        return frame

    if padded_w < width or padded_h < height:
        raise ValueError(
            f"padded size ({padded_w}x{padded_h}) is smaller than the frame "
            f"({width}x{height})"
        )

    padded = np.zeros((padded_h, padded_w), dtype=frame.dtype)
    padded[:height, :width] = frame
    return padded


def count_frames(yuv_path: str | Path, width: int, height: int) -> int:
    """
    Return the number of whole 4:2:0 frames in a raw YUV file.

    Useful for validating a requested POC range before a long extraction run.
    """
    frame_bytes = int(width * height * 1.5)
    return Path(yuv_path).stat().st_size // frame_bytes
