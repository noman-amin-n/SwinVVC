# Training

```bash
python scripts/train.py \
    --train-path datasets/balanced \
    --val-path datasets/validation \
    --output-dir runs/swin_base
```

Or from a config file:

```bash
python scripts/train.py --config configs/default.yaml
```

Command-line flags override the YAML. The resolved configuration is written to
`<output-dir>/config.yaml` and the per-epoch history to
`<output-dir>/history.json`.

## Architecture

ImageNet-pretrained `swin_base_patch4_window7_224` from timm, with the classifier
head replaced by a 6-way head. About 86.7M parameters.

| Stage | Blocks | Feature map |
|---|---|---|
| patch embedding | | 56x56x128 |
| 1 | 2 | 56x56x128 |
| 2 | 2 | 28x28x256 |
| 3 | 18 | 14x14x512 |
| 4 | 2 | 7x7x1024 |
| head | | LayerNorm, global average pool, dropout 0.2, Linear(6) |

The patch embedding splits the 224x224x3 input into 4x4 patches, giving 48 values
per patch, projected to 128 channels. Each block alternates window-based
multi-head self-attention (W-MSA) with its shifted variant (SW-MSA), with residual
connections and LayerNorm throughout.

Shifted-window attention is the reason a transformer fits this task. The split
decision for a 32x32 CU depends on structure across the whole block and its
surroundings, which a local convolutional receptive field models poorly.

## Pre-processing

1. Resize the 32x32 luma block to 224x224 with bilinear interpolation.
2. Replicate the single channel three times so ImageNet weights apply unchanged.
3. Normalise to [-1, 1] with mean 0.5 and std 0.5 per channel.

No geometric or photometric augmentation is used. The partition label is a
directional property of the block, so a flip or rotation would change the correct
answer while leaving the label unchanged. Regularisation comes from MixUp instead.

## Hyper-parameters

| Parameter | Value |
|---|---|
| optimiser | AdamW |
| learning rate | 3e-5 |
| weight decay | 0.05 |
| schedule | cosine annealing, eta_min 1e-6, T_max = epochs |
| batch size | 16 |
| stopping | early stopping on validation accuracy |
| gradient clipping | 1.0 max norm |
| MixUp alpha | 0.8, applied to 50% of batches |
| CutMix | implemented, disabled |
| loss | cross-entropy |
| EMA decay | 0.999 |
| seed | 42 |

### MixUp

A mixing weight `lam ~ Beta(0.8, 0.8)` interpolates two samples and their labels:

```
x = lam * x_i + (1 - lam) * x_j
y = lam * y_i + (1 - lam) * y_j
```

This is implemented as a weighted sum of two cross-entropy terms, which is
equivalent to the interpolated one-hot target without building a soft label
tensor. Following Vicinal Risk Minimisation it smooths decision boundaries, which
helps here because adjacent partition classes are close in appearance.

### EMA

An exponential moving average of the weights is updated after every optimiser step:

```
theta_ema = beta * theta_ema + (1 - beta) * theta
```

The EMA weights are what gets validated, checkpointed and used for inference, not
the raw weights. Averaging suppresses SGD variance and gives a more stable
predictor, which is useful here since a single misclassified CU changes the
encoder's search path.

The decay is set by `--ema-decay`.

## Accuracy reporting

The training accuracy printed per epoch is measured against the interpolated
MixUp target, crediting `lam` for matching `y_a` and `1 - lam` for `y_b`. It is a
soft diagnostic. Judge the model by validation accuracy, which is computed on
unmixed data with the EMA weights.

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint runs/swin_base/best_model.pth \
    --val-path datasets/validation \
    --plot-dir runs/swin_base/figures \
    --json-out runs/swin_base/metrics.json
```

This reports the confusion matrix, per-class accuracy, precision, recall,
specificity and F1, the weighted averages, and one-vs-rest AUC. Figures written
are the confusion matrix (raw and row-normalised), per-class accuracy and ROC
curves.

Specificity is computed from the confusion matrix, since scikit-learn's
classification report does not include it:

```
TN_i = total - row_i - col_i + diag_i
FP_i = col_i - diag_i
specificity_i = TN_i / (TN_i + FP_i)
```

## Reducing memory use

Try `--batch-size 8` first, then `--num-workers 2` on a machine with limited
RAM. A smaller backbone such as
`swin_small_patch4_window7_224` departs from the published architecture and should
be reported as such.

## Reproducibility

`--seed` seeds Python, NumPy and PyTorch. `--deterministic` also forces
deterministic cuDNN kernels at a cost in throughput. Exact bitwise reproduction is
not expected.
