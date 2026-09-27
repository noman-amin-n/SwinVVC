"""
Rendering of CU partition wireframes and label maps.

These figures are diagnostic, not part of the pipeline: they are how a label
extraction run is sanity-checked before a full dataset is built, and they
produce the partition-comparison figures used in the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, ImageDraw

from ..constants import BLOCK_SIZE, CLASS_COLORS
from .labels import get_split_mode

BoundaryMap = Sequence[Sequence[int]]


def render_wireframe(
    boundary: BoundaryMap,
    width: int,
    height: int,
    out_path: Optional[str | Path] = None,
) -> Image.Image:
    """
    Draw the CU boundary map as a black-on-white partition wireframe.

    Args:
        boundary: binary boundary map, addressed as ``boundary[y][x]``.
        width:    padded frame width.
        height:   padded frame height.
        out_path: when given, the image is also saved here.

    Returns:
        The rendered PIL image.
    """
    image = Image.new("RGB", (width + 1, height + 1), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for x in range(width + 1):
        for y in range(height + 1):
            if boundary[y][x] == 1:
                draw.point((x, y), (0, 0, 0))

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
    return image


def render_label_map(
    boundary: BoundaryMap,
    width: int,
    height: int,
    out_path: Optional[str | Path] = None,
    block_size: int = BLOCK_SIZE,
) -> Image.Image:
    """
    Draw one colour-coded, numbered tile per 32x32 block, by predicted class.

    Each block is filled with its class colour (see
    `swinvvc.constants.CLASS_COLORS`) and annotated with the class id, which
    makes a mislabelled region immediately visible.

    Args:
        boundary:   binary boundary map for the frame.
        width:      padded frame width.
        height:     padded frame height.
        out_path:   when given, the image is also saved here.
        block_size: block side length.

    Returns:
        The rendered PIL image.
    """
    image = Image.new("RGB", (width + 1, height + 1), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for x in range(0, width + 1 - block_size, block_size):
        for y in range(0, height + 1 - block_size, block_size):
            mode = get_split_mode(boundary, x, y, block_size, block_size)
            draw.rectangle(
                (x, y, x + block_size, y + block_size),
                fill=CLASS_COLORS[mode],
                outline=(0, 0, 0),
            )
            draw.text((x + block_size // 2, y + block_size // 2), str(mode), fill=(0, 0, 0))

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
    return image
