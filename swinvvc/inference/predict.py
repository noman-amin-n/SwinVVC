"""
Block-level inference: predict a split mode for every 32x32 block.

One forward pass per block replaces the encoder's exhaustive rate-distortion
search over the six partition candidates.  Predictions are written to a CSV
keyed by block filename; the coordinates embedded in those filenames are what
`swinvvc.inference.csv_to_txt` uses to assemble the per-frame HCPM files the
patched encoder reads.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#: Header of the prediction CSV.  `csv_to_txt` reads these column names.
CSV_FIELDNAMES = ("filename", "predicted_class")


@torch.no_grad()
def predict_blocks(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_probabilities: bool = False,
) -> List[Tuple[str, int, Optional[List[float]]]]:
    """
    Predict a split mode for every block served by `loader`.

    Args:
        model:                trained model (set to eval mode here).
        loader:               loader over a `swinvvc.data.BlockDataset`,
                              yielding ``(images, filenames)``.
        device:               compute device.
        return_probabilities: also return the softmax vector per block, for
                              confidence analysis.

    Returns:
        One ``(filename, predicted_class, probabilities_or_None)`` per block.
    """
    model.eval()
    results: List[Tuple[str, int, Optional[List[float]]]] = []

    for images, filenames in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        if return_probabilities:
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            for filename, prediction, probability in zip(filenames, predictions, probabilities):
                results.append((filename, int(prediction), probability.tolist()))
        else:
            for filename, prediction in zip(filenames, predictions):
                results.append((filename, int(prediction), None))

    return results


def write_predictions_csv(
    results: Iterable[Tuple[str, int, Optional[List[float]]]],
    path: str | Path,
    include_probabilities: bool = False,
    num_classes: int = 6,
) -> int:
    """
    Write predictions to CSV.

    The file always begins with the ``filename,predicted_class`` columns so
    that it stays readable by `csv_to_txt` even when probabilities are
    appended.

    Args:
        results:               output of `predict_blocks`.
        path:                  destination CSV path.
        include_probabilities: append one ``p<k>`` column per class.
        num_classes:           number of probability columns to write.

    Returns:
        Number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(CSV_FIELDNAMES)
    if include_probabilities:
        fieldnames += [f"p{k}" for k in range(num_classes)]

    written = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for filename, prediction, probabilities in results:
            row: List[object] = [filename, prediction]
            if include_probabilities:
                row += [f"{value:.6f}" for value in (probabilities or [])]
            writer.writerow(row)
            written += 1

    return written
