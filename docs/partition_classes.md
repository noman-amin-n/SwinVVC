# Partition Classes

## The six modes

VVC partitions a CTU with a quad-tree followed by a multi-type tree (QTMT). At a
32x32 luma CU, rate-distortion optimisation picks one of six options:

| Class | Name | Meaning | Result |
|---|---|---|---|
| 0 | NS | no split | 1 x 32x32 |
| 1 | QT | quad tree | 4 x 16x16 |
| 2 | BTH | binary tree, horizontal | 2 x 32x16 |
| 3 | BTV | binary tree, vertical | 2 x 16x32 |
| 4 | TTH | ternary tree, horizontal | 32x8, 32x16, 32x8 |
| 5 | TTV | ternary tree, vertical | 8x32, 16x32, 8x32 |

These integers are used by the label CSV, the ImageFolder directory names, the
network output layer, the prediction CSV, and the text files VTM reads. They are
defined in [swinvvc/constants.py](../swinvvc/constants.py) and must match
`ChangeTestMode` in the VTM patch.

## How labels are recovered

The encoder does not report its partition choice, so it is reconstructed from the
geometry of the CUs a full-RDO encode produced:

1. Encode the sequence with unmodified VTM, tracing every coded CU
   ([vtm_integration.md](vtm_integration.md) section 2).
2. For each frame, draw the outline of every coded CU into a binary map.
3. For each aligned 32x32 region, check which internal division lines are
   complete.

The rule, in [swinvvc/data/labels.py](../swinvvc/data/labels.py):

```
complete line at mid-height and at mid-width     -> QT
complete line at mid-height only                 -> BTH
complete line at mid-width only                  -> BTV
complete lines at 1/4 and 3/4 height             -> TTH
complete lines at 1/4 and 3/4 width              -> TTV
none of the above                                -> NS
```

Test order is significant. QT's signature (both mid-lines) is a superset of both
binary signatures, so QT is tested first. The binary mid-line test is cheaper and
more specific than the ternary quarter-line test, so it comes next. This ordering
makes the six classes mutually exclusive.

Deeper splits do not confuse the rule. If a 32x32 CU is BTH-split and one of its
32x16 children is split again, the mid-height line is still complete and the extra
lines fall inside a half, which the rule does not test. The label describes the
first split decision at this CU, which is what the encoder needs predicted.

To check a label run visually, pass `--render-dir` to
`scripts/extract_labels.py`. It writes the boundary map and a colour-coded,
numbered label map per frame.

## Class distribution

The natural distribution is skewed. NS and QT dominate and the ternary modes are
rare. From the dataset used in the paper:

| Class | Name | Original | Balanced |
|---|---|---|---|
| 0 | NS | 174,073 | 12,478 |
| 1 | QT | 172,012 | 12,478 |
| 2 | BTH | 76,666 | 12,478 |
| 3 | BTV | 51,934 | 12,478 |
| 4 | TTH | 18,758 | 12,478 |
| 5 | TTV | 12,477 | 12,477 |
| | Total | 505,920 | 74,867 |

Training on the raw distribution collapses the ternary classes, so the dataset is
randomly undersampled to uniform before training
([swinvvc/data/balance.py](../swinvvc/data/balance.py)).

The trade-off is that the model is trained on a prior that does not match
deployment, where NS and QT really are much more common. It buys usable accuracy
on the ternary modes, which prior CNN-based methods did not predict at all.

## The -2 filler

The text files handed to VTM use one extra value, `-2`, for every CU level the
model does not predict, meaning all sizes other than 32x32. The patched encoder
maps it to `ETM_INTRA`, which in `ChangeTestMode` means no override, so those
levels keep their full RDO search. `-2` is not a seventh class, it is the absence
of a prediction. See [vtm_integration.md](vtm_integration.md) section 5.5.
