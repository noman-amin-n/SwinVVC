"""
Classification metrics for the CU-partition model.

Reported quantities are the confusion matrix, per-class and overall accuracy,
precision, recall (sensitivity), specificity, F1, and one-vs-rest ROC/AUC.
Specificity is computed explicitly from the confusion matrix because
scikit-learn's classification report does not include it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..constants import CLASS_NAMES, NUM_CLASSES


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run `model` over `loader` and gather labels, predictions and probabilities.

    Args:
        model:  model to evaluate (set to eval mode here).
        loader: loader yielding ``(inputs, targets)``.
        device: compute device.

    Returns:
        ``(labels, predictions, probabilities)`` with shapes ``(N,)``, ``(N,)``
        and ``(N, num_classes)``.
    """
    model.eval()
    labels: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    probs: List[np.ndarray] = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        batch_probs = torch.softmax(outputs, dim=1)

        labels.append(targets.numpy())
        preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
        probs.append(batch_probs.cpu().numpy())

    return (
        np.concatenate(labels),
        np.concatenate(preds),
        np.concatenate(probs),
    )


def per_class_specificity(confusion: np.ndarray) -> np.ndarray:
    """
    Specificity (true-negative rate) per class, from a confusion matrix.

    For class ``i``, treating it as the positive class:

        TN = total - row_i - column_i + diagonal_i
        FP = column_i - diagonal_i
        specificity = TN / (TN + FP)

    Args:
        confusion: ``(C, C)`` matrix with true labels on rows.

    Returns:
        ``(C,)`` array of specificities; 0 where the denominator vanishes.
    """
    total = confusion.sum()
    specificities = np.zeros(confusion.shape[0], dtype=float)

    for i in range(confusion.shape[0]):
        true_negative = total - confusion[i, :].sum() - confusion[:, i].sum() + confusion[i, i]
        false_positive = confusion[:, i].sum() - confusion[i, i]
        denominator = true_negative + false_positive
        specificities[i] = true_negative / denominator if denominator else 0.0

    return specificities


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, object]:
    """
    Compute the full metric suite reported for this model.

    Args:
        labels:        ground-truth class ids, shape ``(N,)``.
        predictions:   predicted class ids, shape ``(N,)``.
        probabilities: softmax outputs, shape ``(N, C)``.  Required for
                       ROC/AUC; omit to skip those entries.
        class_names:   display names, indexed by class id.

    Returns:
        Dict with keys ``confusion_matrix``, ``accuracy``, ``per_class``,
        ``weighted``, and (when probabilities are given) ``roc``.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    confusion = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    # Row sums can be zero if a class is absent from the evaluation split.
    row_sums = confusion.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        class_accuracy = np.where(row_sums > 0, confusion.diagonal() / row_sums, 0.0)

    specificity = per_class_specificity(confusion)

    per_class = {}
    for index, name in enumerate(class_names):
        per_class[name] = {
            "accuracy": float(class_accuracy[index]),
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "specificity": float(specificity[index]),
            "support": int(report[name]["support"]),
        }

    metrics: Dict[str, object] = {
        "confusion_matrix": confusion,
        "accuracy": float(confusion.diagonal().sum() / confusion.sum()),
        "per_class": per_class,
        "weighted": {
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1": float(report["weighted avg"]["f1-score"]),
            # Weighted by the number of true instances of each class, matching
            # scikit-learn's "weighted avg" convention.
            "specificity": float(np.average(specificity, weights=row_sums))
            if row_sums.sum()
            else 0.0,
        },
    }

    if probabilities is not None:
        metrics["roc"] = compute_roc(labels, probabilities, len(class_names))

    return metrics


def compute_roc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[int, Dict[str, object]]:
    """
    One-vs-rest ROC curve and AUC per class.

    Args:
        labels:        ground-truth class ids, shape ``(N,)``.
        probabilities: softmax outputs, shape ``(N, C)``.
        num_classes:   number of classes.

    Returns:
        ``{class_id: {"fpr": ..., "tpr": ..., "auc": ...}}``.
    """
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    binarized = label_binarize(labels, classes=list(range(num_classes)))

    curves: Dict[int, Dict[str, object]] = {}
    for class_id in range(num_classes):
        fpr, tpr, _ = roc_curve(binarized[:, class_id], probabilities[:, class_id])
        curves[class_id] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
    return curves


def format_report(metrics: Dict[str, object], class_names: Sequence[str] = CLASS_NAMES) -> str:
    """Render `compute_metrics` output as a fixed-width text table."""
    lines: List[str] = []
    lines.append("Confusion matrix (rows = true, columns = predicted):")
    confusion = metrics["confusion_matrix"]
    header = "        " + "".join(f"{name:>8}" for name in class_names)
    lines.append(header)
    for index, name in enumerate(class_names):
        lines.append(f"{name:>8}" + "".join(f"{value:>8d}" for value in confusion[index]))

    lines.append("")
    lines.append(
        f"{'Class':>8}{'Acc':>10}{'Prec':>10}{'Recall':>10}"
        f"{'F1':>10}{'Spec':>10}{'N':>8}"
    )
    for name in class_names:
        stats = metrics["per_class"][name]
        lines.append(
            f"{name:>8}"
            f"{stats['accuracy'] * 100:>9.2f}%"
            f"{stats['precision'] * 100:>9.2f}%"
            f"{stats['recall'] * 100:>9.2f}%"
            f"{stats['f1'] * 100:>9.2f}%"
            f"{stats['specificity'] * 100:>9.2f}%"
            f"{stats['support']:>8d}"
        )

    weighted = metrics["weighted"]
    lines.append("")
    lines.append(f"Overall accuracy            : {metrics['accuracy'] * 100:.2f}%")
    lines.append(f"Weighted precision          : {weighted['precision'] * 100:.2f}%")
    lines.append(f"Weighted recall (sensitivity): {weighted['recall'] * 100:.2f}%")
    lines.append(f"Weighted F1-score           : {weighted['f1'] * 100:.2f}%")
    lines.append(f"Weighted specificity        : {weighted['specificity'] * 100:.2f}%")

    if "roc" in metrics:
        lines.append("")
        lines.append("One-vs-rest AUC:")
        for class_id, curve in metrics["roc"].items():
            lines.append(f"  {class_names[class_id]:>5}: {curve['auc']:.4f}")

    return "\n".join(lines)
