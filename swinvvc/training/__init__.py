"""Training loop, batch augmentation, and evaluation metrics."""

from .engine import EpochResult, train, train_one_epoch, validate
from .evaluate import (
    collect_predictions,
    compute_metrics,
    compute_roc,
    format_report,
    per_class_specificity,
)
from .mixup import cutmix_data, mixup_criterion, mixup_data, rand_bbox

__all__ = [
    "EpochResult",
    "train",
    "train_one_epoch",
    "validate",
    "collect_predictions",
    "compute_metrics",
    "compute_roc",
    "format_report",
    "per_class_specificity",
    "cutmix_data",
    "mixup_criterion",
    "mixup_data",
    "rand_bbox",
]
