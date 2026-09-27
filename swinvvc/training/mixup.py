"""
MixUp and CutMix batch augmentation.

MixUp draws a mixing weight ``lam ~ Beta(alpha, alpha)`` and forms a vicinal
sample by convex interpolation of two training points:

    x = lam * x_i + (1 - lam) * x_j
    y = lam * y_i + (1 - lam) * y_j

Following the Vicinal Risk Minimisation principle this encourages linear
behaviour between samples and smoother decision boundaries, which is valuable
here because neighbouring partition classes are genuinely close in appearance
(a block that is nearly BTH is nearly TTH).  The mixed target is realised not
by mixing one-hot vectors but by the equivalent weighted sum of two
cross-entropy terms (see `mixup_criterion`).

CutMix is provided for completeness; the reported model is trained with MixUp
only (``Config.use_cutmix = False``).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Interpolate a batch with a random permutation of itself.

    Args:
        x:     input batch, shape ``(N, C, H, W)``.
        y:     integer targets, shape ``(N,)``.
        alpha: Beta distribution parameter.  ``alpha <= 0`` disables mixing
               (``lam = 1``), leaving the batch untouched.

    Returns:
        ``(mixed_x, y_a, y_b, lam)`` where the effective target is
        ``lam * y_a + (1 - lam) * y_b``.
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0

    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    """
    Sample a random box whose area is ``1 - lam`` of the image area.

    Args:
        size: shape of the batch tensor, ``(N, C, H, W)``.
        lam:  mixing weight; the box covers ``1 - lam`` of the area.

    Returns:
        ``(x1, y1, x2, y2)`` clipped to the image bounds.
    """
    width, height = size[2], size[3]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)

    center_x, center_y = np.random.randint(width), np.random.randint(height)

    x1 = int(np.clip(center_x - cut_w // 2, 0, width))
    y1 = int(np.clip(center_y - cut_h // 2, 0, height))
    x2 = int(np.clip(center_x + cut_w // 2, 0, width))
    y2 = int(np.clip(center_y + cut_h // 2, 0, height))
    return x1, y1, x2, y2


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Paste a random patch from a permuted batch into `x`.

    `lam` is recomputed from the realised box area, because clipping at the
    image border can make the actual patch smaller than the sampled one.

    Args:
        x:     input batch, shape ``(N, C, H, W)``.  Modified in place.
        y:     integer targets, shape ``(N,)``.
        alpha: Beta distribution parameter.

    Returns:
        ``(mixed_x, y_a, y_b, lam)``.
    """
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)

    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    return x, y, y[index], lam


def mixup_criterion(
    criterion: nn.Module,
    outputs: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Loss for a mixed batch: ``lam * L(out, y_a) + (1 - lam) * L(out, y_b)``.

    For cross-entropy this is exactly the loss against the interpolated
    one-hot target, without materialising a soft label tensor.
    """
    return lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
