# Dataset Generation

Three stages turn raw video into a class-balanced ImageFolder dataset.

```
YUV -> VTM encode with full RDO -> DTrace statistics (.csv)
    -> extract_labels.py  -> labels (poc,x,y,mode)
    -> build_dataset.py   -> raw/0..5/*.png
    -> balance_dataset.py -> balanced/0..5/*.png and validation/0..5/*.png
```

## Requirements

A VTM-20.2 build with tracing enabled (build A in
[vtm_integration.md](vtm_integration.md) section 2), raw 8-bit 4:2:0 YUV
sequences, and the Python environment from the [README](../README.md). Stages 1
to 3 need only NumPy, OpenCV, Pillow and Matplotlib.

## Source data

The model in the paper was trained on Inter4K, 1000 five-second 4K clips released
for the ICIP 2024 Grand Challenge on Video Complexity:
<https://alexandrosstergiou.github.io/datasets/Inter4K/index.html>

Sequences were encoded at QP 22 in All-Intra configuration to get labels.
Evaluation uses the JVET and JCT-VC common test sequences, which are not part of
the training corpus.

## Stage 1: extract labels

Encode a sequence with unmodified VTM and capture the block statistics trace, then:

```bash
python scripts/extract_labels.py \
    --statistics traces/BlowingBubbles_416x240_50_qp22.csv \
    --width 416 --height 240 \
    --output labels/BlowingBubbles_416x240_50_qp22.csv \
    --render-dir figures/BlowingBubbles
```

Output is a headerless CSV with one row per 32x32 block:

```
0,0,0,1
0,0,32,3
0,0,64,0
```

The columns are `poc,x,y,mode`. The script prints the class distribution, and
`--render-dir` writes a boundary map and a colour-coded label map for the first
frame. Check these once per new sequence. A wrong width or height produces a
visibly skewed boundary map and is otherwise easy to miss.

Frames are padded to a multiple of 32 at this stage, not 128, because labels live
on the block grid. Inference pads to 128 instead, see [inference.md](inference.md).

## Stage 2: extract labelled blocks

```bash
python scripts/build_dataset.py \
    --yuv sequences/BlowingBubbles_416x240_50.yuv \
    --labels labels/BlowingBubbles_416x240_50_qp22.csv \
    --width 416 --height 240 \
    --sequence-name BlowingBubbles_416x240_50 \
    --output datasets/raw
```

This writes one 32x32 grayscale PNG per labelled block to
`datasets/raw/<class>/<sequence>_frame_<poc>_x<x>_y<y>.png`.

Run it once per sequence with the same `--output` to build up a single dataset.
`--sequence-name` keeps filenames from colliding.

For many sequences:

```bash
while IFS=, read -r name width height; do
    python scripts/extract_labels.py \
        --statistics "traces/${name}_qp22.csv" \
        --width "$width" --height "$height" \
        --output "labels/${name}.csv"
    python scripts/build_dataset.py \
        --yuv "sequences/${name}.yuv" \
        --labels "labels/${name}.csv" \
        --width "$width" --height "$height" \
        --sequence-name "$name" \
        --output datasets/raw
done < configs/sequences.csv
```

## Stage 3: balance and split

```bash
python scripts/balance_dataset.py \
    --source datasets/raw \
    --output datasets/balanced \
    --val-output datasets/validation \
    --val-per-class 1500 \
    --plot-dir figures/distribution
```

By default every class is undersampled to the size of the smallest one. Pass
`--samples-per-class N` to set it explicitly. `--val-output` moves a held-out
split out of the balanced training set, so the two stay disjoint.

Files are copied rather than moved from `--source`, so the unbalanced dataset
survives and a different balance can be drawn later with a different `--seed`.

Distribution figures and the reasoning behind undersampling are in
[partition_classes.md](partition_classes.md).

## Before training

Check that `extract_labels.py` reports a plausible distribution, with NS and QT
dominant and TTV rarest, and that the rendered boundary map matches the frame's
structure. `--width` and `--height` are the source dimensions, not padded ones.
Block counts per class from `build_dataset.py` should match the label
distribution from stage 1.
