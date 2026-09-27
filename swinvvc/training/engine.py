"""
Training and validation loops.

The loop reproduces the reported training procedure:

* AdamW (lr 3e-5, weight decay 0.05) with cosine annealing to 1e-6;
* MixUp applied to a batch with probability `Config.aug_prob`;
* gradient-norm clipping at `Config.grad_clip`;
* an EMA of the weights updated after every optimiser step;
* validation performed with the **EMA** weights, and the EMA weights saved
  whenever validation accuracy improves;
* early stopping after `Config.patience` epochs without improvement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..config import Config
from ..models.ema import ModelEMA
from .mixup import cutmix_data, mixup_criterion, mixup_data


@dataclass
class EpochResult:
    """Metrics for a single epoch."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    seconds: float

    def format(self, total_epochs: int) -> str:
        """One-line human-readable summary."""
        return (
            f"Epoch {self.epoch}/{total_epochs} | "
            f"Train Loss: {self.train_loss:.4f} | Train Acc: {self.train_acc:.2f}% | "
            f"Val Loss: {self.val_loss:.4f} | Val Acc: {self.val_acc:.2f}% | "
            f"LR: {self.lr:.2e} | Time: {self.seconds:.2f}s"
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: Config,
    ema: Optional[ModelEMA] = None,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Accuracy under MixUp is measured against the interpolated target, i.e. a
    prediction is credited ``lam`` for matching `y_a` and ``1 - lam`` for
    matching `y_b`.  It is therefore a soft, optimistic figure and is reported
    only as a training diagnostic; the validation accuracy is the metric to
    judge the model by.

    Args:
        model:     model in training mode.
        loader:    training data loader.
        criterion: per-sample loss (cross-entropy).
        optimizer: optimiser.
        device:    compute device.
        config:    run configuration.
        ema:       EMA tracker updated after each step, if any.

    Returns:
        ``(mean_loss, accuracy_percent)``.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # --- batch augmentation ------------------------------------------- #
        if config.use_mixup and np.random.rand() < config.aug_prob:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, config.mixup_alpha)
        elif config.use_cutmix and np.random.rand() < config.aug_prob:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, config.cutmix_alpha)
        else:
            targets_a, targets_b, lam = targets, targets, 1.0

        # --- forward / backward ------------------------------------------- #
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        # --- metrics ------------------------------------------------------- #
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs.detach(), 1)
        running_correct += (
            lam * predicted.eq(targets_a).sum().item()
            + (1.0 - lam) * predicted.eq(targets_b).sum().item()
        )
        total += batch_size

    return running_loss / total, 100.0 * running_correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate `model` on `loader`.

    Args:
        model:     model to evaluate (set to eval mode here).
        loader:    validation data loader.
        criterion: per-sample loss.
        device:    compute device.

    Returns:
        ``(mean_loss, accuracy_percent)``.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, 100.0 * correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
) -> Tuple[List[EpochResult], float]:
    """
    Train the model, checkpointing the best EMA weights.

    Args:
        model:        model to train, already on `device`.
        train_loader: training loader.
        val_loader:   validation loader.
        config:       run configuration.
        device:       compute device.

    Returns:
        ``(history, best_val_acc)``.  The best EMA weights are written to
        `config.checkpoint_path`.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.min_lr
    )
    ema = ModelEMA(model, config.ema_decay).to(device)

    checkpoint_path = config.checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: List[EpochResult] = []
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        started = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config, ema
        )
        # Validate the averaged weights: those are the ones that get deployed.
        val_loss, val_acc = validate(ema.ema, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=current_lr,
            seconds=time.time() - started,
        )
        history.append(result)
        print(result.format(config.num_epochs), flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(ema.state_dict(), checkpoint_path)
            print(f"  -> saved new best EMA weights to {checkpoint_path}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered.", flush=True)
                break

    return history, best_val_acc
