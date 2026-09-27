"""
Inference: block-level prediction and HCPM file generation for VTM.

`csv_to_txt` needs nothing but the standard library, so it is imported eagerly;
`predict` pulls in torch and is resolved lazily, which lets the conversion
stage run on a machine without a deep-learning stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .csv_to_txt import (
    build_frame_lines,
    convert,
    index_predictions,
    parse_block_filename,
    validate_frame_file,
)

#: Attributes served on demand from `swinvvc.inference.predict`.
_LAZY = {"predict_blocks": "predict", "write_predictions_csv": "predict"}

if TYPE_CHECKING:  # pragma: no cover
    from .predict import predict_blocks, write_predictions_csv


def __getattr__(name: str) -> Any:
    """Import torch-dependent submodules only when they are actually used."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)


__all__ = [
    "build_frame_lines",
    "convert",
    "index_predictions",
    "parse_block_filename",
    "validate_frame_file",
    "predict_blocks",
    "write_predictions_csv",
]
