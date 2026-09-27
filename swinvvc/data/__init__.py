"""
Dataset construction: YUV I/O, DTrace parsing, labelling, block extraction.

Names are resolved lazily so that the stages which need no deep-learning stack
- label extraction, block extraction, HCPM conversion - can run in an
environment where only NumPy, OpenCV and Pillow are installed.  Importing
`build_dataloaders` or `BlockDataset` pulls in torch/torchvision at that point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .balance import balance_dataset, count_per_class, split_validation
from .blocks import extract_inference_blocks, extract_labelled_blocks
from .dtrace import build_boundary_maps, iter_coded_cus
from .labels import (
    get_split_mode,
    label_frame,
    label_sequence,
    read_label_csv,
    write_label_csv,
)
from .yuv import count_frames, get_padded_size, pad_frame, read_y_frame

#: Attributes served on demand from `swinvvc.data.datasets`, which imports torch.
_LAZY = {
    "BlockDataset": "datasets",
    "build_dataloaders": "datasets",
    "build_inference_loader": "datasets",
    "build_transform": "datasets",
}

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from .datasets import (
        BlockDataset,
        build_dataloaders,
        build_inference_loader,
        build_transform,
    )


def __getattr__(name: str) -> Any:
    """Import torch-dependent submodules only when they are actually used."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    return getattr(module, name)


__all__ = [
    "balance_dataset",
    "count_per_class",
    "split_validation",
    "extract_inference_blocks",
    "extract_labelled_blocks",
    "BlockDataset",
    "build_dataloaders",
    "build_inference_loader",
    "build_transform",
    "build_boundary_maps",
    "iter_coded_cus",
    "get_split_mode",
    "label_frame",
    "label_sequence",
    "read_label_csv",
    "write_label_csv",
    "count_frames",
    "get_padded_size",
    "pad_frame",
    "read_y_frame",
]
