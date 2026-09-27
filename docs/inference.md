# Inference

Turning a sequence into the partition decision files the patched VTM reads.

```
YUV -> generate_blocks.py -> blocks/*.png
    -> predict.py         -> predictions.csv
    -> csv_to_txt.py      -> POC_0.txt, POC_1.txt, ...
    -> patched VTM        -> bitstream
```

## Stage 1: generate blocks

```bash
python scripts/generate_blocks.py \
    --yuv sequences/ParkRunning3_3840x2160_50.yuv \
    --width 3840 --height 2160 \
    --start-poc 0 --end-poc 0 \
    --output blocks/ParkRunning3
```

Each frame is zero-padded to a multiple of 128, the CTU size, then cut into 32x32
blocks named:

```
<sequence>_poc<POC>_x<X>_y<Y>.png
```

Two things to keep in mind. Padding here is CTU-aligned, while dataset generation
pads to 32. The encoder walks whole CTUs, so the block grid must cover complete
CTUs or the HCPM indices will not line up. And the filename carries the
coordinates that `csv_to_txt.py` parses to place each prediction in the right
slot, so renaming blocks breaks the pipeline without an error.

Block counts grow quickly. 3840x2160 pads to 3840x2176, giving 8160 blocks per
frame.

## Stage 2: predict

```bash
python scripts/predict.py \
    --checkpoint runs/swin_base/best_model.pth \
    --blocks blocks/ParkRunning3 \
    --output predictions/ParkRunning3.csv \
    --batch-size 64
```

One forward pass per block replaces the encoder's RDO search over the six
candidates. Output:

```csv
filename,predicted_class
ParkRunning3_poc0_x0_y0.png,1
ParkRunning3_poc0_x32_y0.png,3
```

`--batch-size` can be larger than the training batch size. It affects throughput
only. `--save-probabilities` appends one softmax column per class.

The printed class distribution is a useful check. If it collapses onto a single
class, the blocks or the checkpoint are wrong.

## Stage 3: convert to HCPM text files

```bash
python scripts/csv_to_txt.py \
    --predictions predictions/ParkRunning3.csv \
    --output vtm_input/ParkRunning3 \
    --width 3840 --height 2160
```

This writes one `POC_<n>.txt` per frame, 341 integers per CTU in raster order.
Entries 5 to 20 of each record hold the sixteen 32x32 predictions and everything
else is `-2`, meaning no override. Layout details are in
[vtm_integration.md](vtm_integration.md) section 5.5.

Always pass `--width` and `--height`. They enable a check that each file holds
exactly `ceil(W/128) * ceil(H/128) * 341` lines. A short file makes every CTU
after the gap read another CTU's decisions, and the encoder will not complain.

This stage needs no PyTorch, which helps when the prediction and encoding steps
run on different machines.

## Stage 4: encode

Point the patched encoder at the output directory and run it. The HCPM path is
hard-coded in `EncCu.cpp` and must be edited per sequence and QP, see
[vtm_integration.md](vtm_integration.md) sections 3.1 and 4. There is no error
handling on that file open, so a wrong path fails silently.

## Stage 5: measure

```bash
python scripts/bd_metrics.py --json results/ParkRunning3.json --plot-dir results/figures
```

This reports BD-BR, BD-PSNR, per-QP and average encoding time saving, the
time-to-bitrate trade-off ratio, and optionally dSSIM and dVMAF. The JSON input
format is in the script's docstring. Encode with both the patched and an unpatched
build at QPs {22, 27, 32, 37} to fill it in.

## Full example

```bash
SEQ=BlowingBubbles_416x240_50
W=416; H=240

python scripts/generate_blocks.py \
    --yuv "sequences/${SEQ}.yuv" --width $W --height $H \
    --start-poc 0 --end-poc 4 --output "blocks/${SEQ}"

python scripts/predict.py \
    --checkpoint runs/swin_base/best_model.pth \
    --blocks "blocks/${SEQ}" --output "predictions/${SEQ}.csv"

python scripts/csv_to_txt.py \
    --predictions "predictions/${SEQ}.csv" \
    --output "vtm_input/${SEQ}" --width $W --height $H
```

Then edit the path in `EncCu.cpp` to `vtm_input/${SEQ}`, rebuild, and encode.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `No block images found` | wrong `--blocks` directory, or stage 1 not run |
| `Filename ... does not match the expected pattern` | blocks renamed, or not produced by `generate_blocks.py` |
| `expected N lines, found M` | block directory did not cover every CTU of every frame |
| `CTU ... has k sub-blocks, expected 16` | blocks missing, partial prediction CSV |
| all predictions one class | wrong checkpoint, or blank blocks |
| encoder ignores predictions | HCPM path in `EncCu.cpp` not updated |
