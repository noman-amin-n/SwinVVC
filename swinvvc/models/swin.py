"""
Swin Transformer backbone for 32x32 CU split-mode classification.

The architecture is `swin_base_patch4_window7_224` from ``timm``, pretrained on
ImageNet, with the classifier head replaced by a 6-way head:

* patch embedding: 4x4 non-overlapping patches of the 224x224x3 input,
  i.e. 4*4*3 = 48 values per patch, linearly projected to 128 channels;
* four stages with depths (2, 2, 18, 2) and feature maps
  56x56x128 -> 28x28x256 -> 14x14x512 -> 7x7x1024, each transition performed
  by a patch-merging layer that aggregates 2x2 neighbouring patches;
* window-based multi-head self-attention (W-MSA) alternating with its shifted
  variant (SW-MSA), so that receptive fields grow across windows;
* final LayerNorm, global average pooling, dropout, and a linear 6-way head.

Shifted-window attention is what makes this a stronger fit than a CNN for the
task: the split decision for a 32x32 CU depends on structure across the whole
block and its surroundings, not on a local neighbourhood.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..config import Config
from ..constants import NUM_CLASSES


def build_model(
    model_name: str = "swin_base_patch4_window7_224",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    attn_drop_rate: float = 0.1,
) -> nn.Module:
    """
    Instantiate the Swin Transformer classifier.

    Args:
        model_name:     ``timm`` model identifier.
        num_classes:    size of the output layer (6 partition modes).
        pretrained:     load ImageNet weights.  Required to reproduce the
                        reported results; training from scratch on this dataset
                        does not converge to comparable accuracy.
        drop_rate:      dropout applied before the classifier head.
        attn_drop_rate: dropout applied to attention weights.

    Returns:
        The model, on CPU.  Move it to the target device yourself.
    """
    import timm

    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )


def build_model_from_config(config: Config) -> nn.Module:
    """Instantiate the model described by `config`."""
    return build_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        drop_rate=config.drop_rate,
        attn_drop_rate=config.attn_drop_rate,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load a state dict saved by the training script into `model`.

    The checkpoints produced here are plain ``state_dict`` files holding the
    **EMA** weights (see `swinvvc.models.ema`), which is what the reported
    results were obtained with.

    Args:
        model:           model to load into, already built with the same
                         architecture and class count.
        checkpoint_path: path to the ``.pth`` file.
        device:          device to map tensors to while loading.
        strict:          require an exact key match.

    Returns:
        `model`, mutated in place and returned for convenience.
    """
    map_location = device if device is not None else "cpu"
    state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

    # Tolerate checkpoints wrapped in a dict by other training harnesses.
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=strict)
    return model


def get_device(prefer: Optional[str] = None) -> torch.device:
    """
    Pick a compute device: the requested one, else CUDA, else Apple MPS, else CPU.

    Args:
        prefer: explicit device string (e.g. ``"cuda:0"``, ``"cpu"``).
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
