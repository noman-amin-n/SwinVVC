"""
Plotting helpers for the figures reported in the paper.

Every function takes an optional `out_path` and returns the Matplotlib figure,
so plots can be saved from a script or further customised in a notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from ..constants import CLASS_DESCRIPTIONS, CLASS_NAMES


def _save(figure, out_path: Optional[str | Path], dpi: int = 300):
    """Save `figure` to `out_path` if given, creating parent directories."""
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return figure


def plot_class_distribution(
    counts: Dict[int, int],
    title: str = "Class Distribution",
    out_path: Optional[str | Path] = None,
    color: str = "skyblue",
):
    """
    Bar chart of samples per partition class, annotated with exact counts.

    Args:
        counts:   mapping from class id to sample count.
        title:    figure title.
        out_path: when given, the figure is saved here.
        color:    bar colour.
    """
    import matplotlib.pyplot as plt

    class_ids = sorted(counts)
    values = [counts[class_id] for class_id in class_ids]
    labels = [f"{class_id}\n{CLASS_NAMES[class_id]}" for class_id in class_ids]

    figure, axes = plt.subplots(figsize=(9, 5))
    bars = axes.bar(labels, values, color=color)

    for bar in bars:
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{int(bar.get_height()):,}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    axes.set_xlabel("Partition class", fontsize=12, fontweight="bold")
    axes.set_ylabel("Number of samples", fontsize=12, fontweight="bold")
    axes.set_title(title, fontsize=13, fontweight="bold")
    axes.grid(axis="y", linestyle="--", alpha=0.7)
    figure.tight_layout()

    return _save(figure, out_path)


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    normalize: bool = False,
    out_path: Optional[str | Path] = None,
):
    """
    Heat map of a confusion matrix.

    Args:
        confusion:   ``(C, C)`` matrix, true labels on rows.
        class_names: tick labels.
        normalize:   show row-normalised fractions instead of counts.
        out_path:    when given, the figure is saved here.
    """
    import matplotlib.pyplot as plt

    matrix = confusion.astype(float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    figure, axes = plt.subplots(figsize=(6.5, 5.5))
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes)

    axes.set_xticks(range(len(class_names)), class_names)
    axes.set_yticks(range(len(class_names)), class_names)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.set_title("Confusion Matrix" + (" (normalised)" if normalize else ""))

    # Annotate each cell, switching text colour for legibility on dark cells.
    threshold = matrix.max() / 2.0 if matrix.max() else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = f"{matrix[i, j]:.2f}" if normalize else f"{int(confusion[i, j])}"
            axes.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=9,
            )

    figure.tight_layout()
    return _save(figure, out_path)


def plot_per_class_accuracy(
    confusion: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    out_path: Optional[str | Path] = None,
):
    """Bar chart of per-class accuracy derived from a confusion matrix."""
    import matplotlib.pyplot as plt

    row_sums = confusion.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = np.where(row_sums > 0, 100.0 * confusion.diagonal() / row_sums, 0.0)

    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.bar(list(class_names), accuracy, color="steelblue")
    axes.set_xlabel("Partition class")
    axes.set_ylabel("Accuracy (%)")
    axes.set_title("Per-Class Accuracy")
    axes.set_ylim(0, 100)
    axes.grid(axis="y", linestyle="--", alpha=0.7)
    figure.tight_layout()

    return _save(figure, out_path)


def plot_roc_curves(
    roc: Dict[int, Dict[str, object]],
    class_names: Sequence[str] = CLASS_NAMES,
    out_path: Optional[str | Path] = None,
):
    """
    One-vs-rest ROC curves, one per class, with AUC in the legend.

    Args:
        roc:         output of `swinvvc.training.evaluate.compute_roc`.
        class_names: legend labels.
        out_path:    when given, the figure is saved here.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7.5, 6))
    for class_id, curve in sorted(roc.items()):
        axes.plot(
            curve["fpr"],
            curve["tpr"],
            lw=1.5,
            label=f"{class_names[class_id]} (AUC = {curve['auc']:.3f})",
        )

    axes.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("True Positive Rate")
    axes.set_title("One-vs-Rest ROC Curves")
    axes.legend(loc="lower right", fontsize=9)
    axes.grid(True, alpha=0.3)
    figure.tight_layout()

    return _save(figure, out_path)


def plot_rd_curve(
    rate_anchor: Sequence[float],
    metric_anchor: Sequence[float],
    rate_test: Sequence[float],
    metric_test: Sequence[float],
    metric_name: str = "Y-PSNR (dB)",
    title: str = "",
    rate_scale: float = 1024.0,
    rate_label: str = "Bitrate (Mbps)",
    out_path: Optional[str | Path] = None,
):
    """
    Rate-distortion comparison between the anchor and the proposed method.

    Args:
        rate_anchor:   anchor bitrates (kbps by default; see `rate_scale`).
        metric_anchor: anchor quality values (PSNR, SSIM or VMAF).
        rate_test:     proposed-method bitrates.
        metric_test:   proposed-method quality values.
        metric_name:   y-axis label.
        title:         figure title, e.g. the sequence name.
        rate_scale:    divisor applied to the rates (1024 converts kbps to Mbps).
        rate_label:    x-axis label.
        out_path:      when given, the figure is saved here.
    """
    import matplotlib.pyplot as plt

    rate_anchor = np.asarray(rate_anchor, dtype=float) / rate_scale
    rate_test = np.asarray(rate_test, dtype=float) / rate_scale

    figure, axes = plt.subplots(figsize=(5, 4))
    axes.plot(
        rate_anchor, metric_anchor,
        color="black", linestyle="-", linewidth=0.8,
        marker="^", markersize=5, label="VVC (VTM anchor)",
    )
    axes.plot(
        rate_test, metric_test,
        color="red", linestyle="-", linewidth=0.8,
        marker="*", markersize=7, label="Proposed Method",
    )

    axes.set_xlabel(rate_label, fontsize=11)
    axes.set_ylabel(metric_name, fontsize=11)
    if title:
        axes.set_title(title, fontsize=11)

    legend = axes.legend(loc="lower right", fontsize=9, framealpha=0.9)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("black")
    figure.tight_layout()

    return _save(figure, out_path)


def plot_encoding_time(
    qps: Sequence[int],
    time_anchor: Sequence[float],
    time_test: Sequence[float],
    title: str = "",
    out_path: Optional[str | Path] = None,
):
    """Encoding time of the anchor and the proposed method, per QP."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(5, 4))
    axes.plot(qps, time_anchor, color="black", marker="^", markersize=5,
              linewidth=0.8, label="VVC (VTM anchor)")
    axes.plot(qps, time_test, color="red", marker="*", markersize=7,
              linewidth=0.8, label="Proposed Method")

    axes.set_xlabel("QP", fontsize=11)
    axes.set_ylabel("Encoding time (s)", fontsize=11)
    axes.set_xticks(list(qps))
    if title:
        axes.set_title(title, fontsize=11)
    axes.legend(fontsize=9)
    axes.grid(alpha=0.3)
    figure.tight_layout()

    return _save(figure, out_path)
