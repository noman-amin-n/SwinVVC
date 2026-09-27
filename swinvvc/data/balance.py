"""
Class balancing for the CU-partition dataset.

The natural distribution of 32x32 split modes is heavily skewed: no-split and
quad-tree together account for roughly three quarters of all CUs, while the
ternary modes are rare.  Training directly on that distribution collapses the
ternary classes, so the dataset is **randomly undersampled** to a uniform
distribution before training, as described in the paper.

Undersampling (rather than oversampling or loss re-weighting) is a deliberate
methodological choice of the original work and is preserved here.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from ..constants import NUM_CLASSES


def count_per_class(root: str | Path, num_classes: int = NUM_CLASSES) -> Dict[int, int]:
    """
    Count the image files in each class directory of an ImageFolder tree.

    Args:
        root:        directory containing ``0/`` .. ``<num_classes-1>/``.
        num_classes: number of class directories to inspect.

    Returns:
        Mapping from class id to file count (0 for a missing directory).
    """
    root = Path(root)
    counts: Dict[int, int] = {}
    for class_id in range(num_classes):
        class_dir = root / str(class_id)
        counts[class_id] = (
            sum(1 for p in class_dir.iterdir() if p.is_file()) if class_dir.is_dir() else 0
        )
    return counts


def balance_dataset(
    source_root: str | Path,
    target_root: str | Path,
    samples_per_class: Optional[int] = None,
    num_classes: int = NUM_CLASSES,
    seed: int = 42,
    move: bool = False,
) -> Dict[int, int]:
    """
    Build a class-balanced copy of an ImageFolder dataset by undersampling.

    Args:
        source_root:       dataset to sample from (``0/`` .. ``5/``).
        target_root:       destination for the balanced dataset.
        samples_per_class: how many files to keep per class.  When ``None``,
                           the size of the smallest class is used, which keeps
                           every sample of the rarest class.
        num_classes:       number of classes.
        seed:              seed for the sampling RNG, for reproducibility.
        move:              move files instead of copying them.  Copying (the
                           default) leaves the unbalanced dataset intact so
                           that a different balance can be drawn later.

    Returns:
        Number of files placed in each class directory.
    """
    source_root = Path(source_root)
    target_root = Path(target_root)
    rng = random.Random(seed)

    counts = count_per_class(source_root, num_classes)
    empty = [class_id for class_id, n in counts.items() if n == 0]
    if empty:
        raise ValueError(f"No samples found for class(es) {empty} under {source_root}")

    if samples_per_class is None:
        samples_per_class = min(counts.values())

    target_root.mkdir(parents=True, exist_ok=True)
    written: Dict[int, int] = {}

    for class_id in range(num_classes):
        class_dir = source_root / str(class_id)
        files: List[Path] = sorted(p for p in class_dir.iterdir() if p.is_file())

        # `min` guards the case where a class is smaller than the requested
        # count; the class is then kept whole rather than raising.
        take = min(samples_per_class, len(files))
        selected = rng.sample(files, take)

        out_dir = target_root / str(class_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in selected:
            dst = out_dir / src.name
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))

        written[class_id] = take

    return written


def split_validation(
    source_root: str | Path,
    val_root: str | Path,
    samples_per_class: int,
    num_classes: int = NUM_CLASSES,
    seed: int = 42,
) -> Dict[int, int]:
    """
    Move a held-out validation set out of a balanced training tree.

    Files are **moved**, not copied, so that the training and validation sets
    stay disjoint.

    Args:
        source_root:       balanced dataset to draw the validation set from.
        val_root:          destination for the validation set.
        samples_per_class: validation samples to hold out per class.
        num_classes:       number of classes.
        seed:              seed for the sampling RNG.

    Returns:
        Number of files held out per class.
    """
    return balance_dataset(
        source_root=source_root,
        target_root=val_root,
        samples_per_class=samples_per_class,
        num_classes=num_classes,
        seed=seed,
        move=True,
    )
