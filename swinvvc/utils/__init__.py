"""
Shared utilities: BD metrics, plotting, seeding.

`bjontegaard` needs only NumPy and is imported eagerly; `set_seed` touches
torch and is resolved lazily, so that result analysis works in an environment
without a deep-learning stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .bjontegaard import (
    bd_psnr,
    bd_rate,
    bjontegaard,
    relative_difference,
    time_saving,
)

#: Attributes served on demand from `swinvvc.utils.seed`.
_LAZY = {"set_seed": "seed"}

if TYPE_CHECKING:  # pragma: no cover
    from .seed import set_seed


def __getattr__(name: str) -> Any:
    """Import torch-dependent submodules only when they are actually used."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)


__all__ = [
    "bd_psnr",
    "bd_rate",
    "bjontegaard",
    "relative_difference",
    "time_saving",
    "set_seed",
]
