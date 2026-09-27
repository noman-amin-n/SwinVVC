"""
Exponential Moving Average (EMA) of model parameters.

Averaging the weight trajectory suppresses the variance that stochastic
gradient updates introduce and yields a more stable predictor, which matters
here because a single misclassified CU changes the encoder's search path.  The
EMA weights - not the raw ones - are what gets checkpointed, validated, and
shipped for inference.

The update rule is

    theta_ema <- beta * theta_ema + (1 - beta) * theta
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    """Maintains a shadow copy of a model holding its EMA weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: model to track.  A deep copy is taken immediately, so the
                   EMA starts from the initial (pretrained) weights.
            decay: smoothing coefficient ``beta``.  Higher values average over
                   a longer history.
        """
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.ema = deepcopy(model).eval()
        self.decay = decay

        # The shadow model is never optimised directly.
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Fold the current parameters of `model` into the EMA.

        Only parameters are averaged; buffers (e.g. running statistics) are
        left as copied at construction time, matching the original
        implementation.
        """
        for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
            ema_param.mul_(self.decay).add_(model_param, alpha=1.0 - self.decay)

    def state_dict(self):
        """State dict of the averaged model, for checkpointing."""
        return self.ema.state_dict()

    def to(self, device: torch.device) -> "ModelEMA":
        """Move the shadow model to `device`."""
        self.ema.to(device)
        return self
