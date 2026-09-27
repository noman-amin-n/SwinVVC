"""
Torch datasets and input transforms.

Pre-processing follows the paper exactly:

1. the 32x32 luma block is bilinearly resized to 224x224 (the resolution the
   ImageNet-pretrained Swin backbone expects);
2. the single luma channel is replicated three times, so that pretrained
   three-channel weights can be used unchanged;
3. samples are normalised to ``[-1, 1]``.

No geometric or photometric augmentation is applied.  Flips and rotations are
deliberately avoided: a horizontal flip maps a BTV partition onto another BTV
partition but relocates the content, and more importantly the label of a block
is a *directional* property, so appearance-altering augmentation risks teaching
the model label-inconsistent invariances.  Regularisation is provided by MixUp
instead (`swinvvc.training.mixup`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ..config import Config
from ..constants import INPUT_SIZE, NORM_MEAN, NORM_STD

#: Image extensions accepted by `BlockDataset`.
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def build_transform(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """
    Build the pre-processing pipeline used for training, validation and test.

    A single transform is used for all three splits because the reported model
    trains without image-space augmentation; MixUp is applied later, on the
    batch tensor, inside the training loop.

    Args:
        input_size: side length the block is resized to.

    Returns:
        The composed transform.
    """
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(NORM_MEAN), std=list(NORM_STD)),
        ]
    )


class BlockDataset(Dataset):
    """
    Flat directory of unlabelled 32x32 block images, for inference.

    Unlike ``torchvision.datasets.ImageFolder`` this expects no class
    subdirectories and returns the filename alongside the tensor, because the
    coordinates encoded in the filename are what map a prediction back to its
    position in the frame.
    """

    def __init__(self, image_dir: str | Path, transform: Optional[Callable] = None):
        """
        Args:
            image_dir: directory of block PNGs produced by
                       `swinvvc.data.blocks.extract_inference_blocks`.
            transform: transform applied to each loaded image.
        """
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"{self.image_dir} is not a directory")

        # Sorted for determinism: the prediction CSV should not depend on the
        # order the filesystem happens to enumerate entries in.
        self.image_paths: List[Path] = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No block images found in {self.image_dir}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, path.name


def build_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Build the training and validation loaders from an ImageFolder layout.

    Both roots must contain one subdirectory per class, named ``0`` .. ``5``;
    ``ImageFolder`` sorts those names, so directory ``k`` maps to class index
    ``k`` and the label space matches `swinvvc.constants`.

    Args:
        config: run configuration; `train_path` and `val_path` must be set.

    Returns:
        ``(train_loader, val_loader)``.
    """
    if not config.train_path or not config.val_path:
        raise ValueError("config.train_path and config.val_path must both be set")

    transform = build_transform(config.input_size)
    train_dataset = datasets.ImageFolder(root=config.train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=config.val_path, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_inference_loader(
    image_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    input_size: int = INPUT_SIZE,
) -> DataLoader:
    """
    Build a loader over a flat directory of block images.

    `shuffle` is off so that predictions come out in a stable order.
    """
    dataset = BlockDataset(image_dir, transform=build_transform(input_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
