"""
Training / inference configuration.

The defaults reproduce the experiment reported in the paper.  They can be
overridden from a YAML file (see `configs/default.yaml`) or from the command
line by the scripts in `scripts/`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .constants import INPUT_SIZE, NUM_CLASSES


@dataclass
class Config:
    """Hyper-parameters for the Swin Transformer CU-partition classifier."""

    # -- reproducibility ---------------------------------------------------- #
    seed: int = 42

    # -- data --------------------------------------------------------------- #
    #: Root of the class-balanced training set (an ImageFolder tree `0/` .. `5/`).
    train_path: Optional[str] = None
    #: Root of the validation set (same layout as `train_path`).
    val_path: Optional[str] = None
    num_workers: int = 4
    input_size: int = INPUT_SIZE

    # -- model -------------------------------------------------------------- #
    model_name: str = "swin_base_patch4_window7_224"
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    drop_rate: float = 0.2
    attn_drop_rate: float = 0.1

    # -- optimisation ------------------------------------------------------- #
    batch_size: int = 16
    num_epochs: int = 12
    lr: float = 3e-5
    weight_decay: float = 0.05
    min_lr: float = 1e-6          # cosine-annealing floor (eta_min)
    grad_clip: float = 1.0

    # -- regularisation ----------------------------------------------------- #
    use_mixup: bool = True
    mixup_alpha: float = 0.8
    #: CutMix is implemented but disabled; the reported model uses MixUp only.
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    #: Probability of applying MixUp (or CutMix) to a given batch.
    aug_prob: float = 0.5
    #: Retained for reference; the reported run uses plain cross-entropy.
    label_smoothing: float = 0.1

    # -- exponential moving average ----------------------------------------- #
    ema_decay: float = 0.999

    # -- early stopping ----------------------------------------------------- #
    patience: int = 3

    # -- output ------------------------------------------------------------- #
    output_dir: str = "runs/swin_base"
    checkpoint_name: str = "best_model.pth"

    @property
    def checkpoint_path(self) -> Path:
        """Full path the best (EMA) weights are written to."""
        return Path(self.output_dir) / self.checkpoint_name

    # ---------------------------------------------------------------------- #
    # (de)serialisation
    # ---------------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build a Config from a dict, ignoring unknown keys with a warning."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a Config from a YAML file."""
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        """Write this Config to a YAML file (for run provenance)."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
