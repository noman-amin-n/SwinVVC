"""
SwinVVC - Vision Transformer-driven CU partitioning for VVC intra coding.

Reference implementation for:

    Md. Zahirul Islam, Noman Amin, Manoranjan Paul, Zhenni Pan, Quazi Mamun,
    "Intra-Mode Encoding Complexity Reduction: Vision Transformer-Driven CU
    Partitioning for VVC", IEEE ICDM Workshops (ICDMW), 2025.

The package predicts the partition mode of each 32x32 luma coding unit with a
Swin Transformer, letting the VVC encoder skip the exhaustive rate-distortion
search at that level.

Pipeline
--------
1. `swinvvc.data`      - build the labelled dataset from VTM DTrace output.
2. `swinvvc.models`    - the Swin Transformer classifier and its EMA.
3. `swinvvc.training`  - training loop and evaluation metrics.
4. `swinvvc.inference` - block prediction and HCPM file generation for VTM.
5. `swinvvc.utils`     - BD-BR / BD-PSNR metrics and plotting.
"""

__version__ = "1.0.0"

from .config import Config
from .constants import (
    BLOCK_SIZE,
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    CTU_SIZE,
    HCPM_ENTRIES_PER_CTU,
    NUM_CLASSES,
)

__all__ = [
    "__version__",
    "Config",
    "BLOCK_SIZE",
    "CLASS_DESCRIPTIONS",
    "CLASS_NAMES",
    "CTU_SIZE",
    "HCPM_ENTRIES_PER_CTU",
    "NUM_CLASSES",
]
