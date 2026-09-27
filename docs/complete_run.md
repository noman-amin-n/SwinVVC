# Complete Run

A full walkthrough from an empty machine to an encoded bitstream, using the
released checkpoint. The example sequence is `BlowingBubbles_416x240_50`, which is
small enough to finish quickly.

Class D (416x240) is the right resolution to start with, since inference cost
scales with pixel count.

To train a model instead of using the released checkpoint, see
[dataset_generation.md](dataset_generation.md) and [training.md](training.md),
then resume here at step 4.

## Requirements

| Item | Notes |
|---|---|
| Python 3.9 or newer | 3.11 is what `environment.yml` pins |
| `best_model.pth` | about 347 MB, from the authors or a release asset |
| A raw YUV sequence | 8-bit 4:2:0 planar |
| A C++ toolchain and CMake | to build VTM |

Most of the elapsed time for a short run is spent compiling VTM.

## 1. Python environment

```bash
cd /path/to/SwinVVC
conda env create -f environment.yml
conda activate swinvvc
```

Or with venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

Activation is per terminal session. Confirm the install:

```bash
pytest -q
```

Expect `53 passed`. If this fails, fix the environment before continuing.

## 2. Build the patched encoder

```bash
git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
cd VVCSoftware_VTM
git checkout VTM-20.2
git apply --whitespace=nowarn /path/to/SwinVVC/vtm_patch/01-partition-override.patch
```

Do not apply `02-block-statistics-tracing.patch` to this build. That patch is only
for generating training labels, and the tracing it enables costs runtime that
distorts timing measurements.

Leave the hard-coded path alone for now. It is set in step 7, once the location of
the prediction files is known.

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

This takes 10 to 15 minutes. The binary lands in `bin/`, named
`EncoderAppStatic` or similar depending on platform.

An unpatched build is also needed, to serve as the anchor for comparison. Clone
VTM-20.2 a second time into a different directory, apply nothing, and build it the
same way. The two are referred to below as `VTM-patched` and `VTM-anchor`.

## 3. Add the sequence

```bash
cd /path/to/SwinVVC
mkdir -p sequences
cp /path/to/BlowingBubbles_416x240_50.yuv sequences/
```

`sequences/` is git-ignored. Raw CTC clips come from the JVET distribution, and
the Xiph derf collection carries many common test sequences.

To exercise the pipeline before a real sequence is available, generate a synthetic
one:

```bash
python -c "
import numpy as np
w,h,n = 416,240,5
rng = np.random.default_rng(0)
with open('sequences/synthetic_416x240.yuv','wb') as f:
    for _ in range(n):
        y = np.zeros((h,w), np.uint8)
        for j in range(0,w,16): y[:, j:j+8] = 200
        f.write((y + rng.integers(0,40,(h,w))).clip(0,255).astype(np.uint8).tobytes())
        f.write(np.full(h//2*w//2*2, 128, np.uint8).tobytes())
"
```

It encodes correctly and exercises every stage, but the partition predictions are
not meaningful on noise.

## 4. Generate blocks

```bash
python scripts/generate_blocks.py \
    --yuv sequences/BlowingBubbles_416x240_50.yuv \
    --width 416 --height 240 \
    --start-poc 0 --end-poc 4 \
    --output blocks/BlowingBubbles
```

Expected output:

```
CTU-padded    : 512x256
Frames        : POC 0..4 (5)
Blocks/frame  : 128
Blocks total  : 640
```

The frame is padded from 416x240 to 512x256 so the block grid covers whole
128x128 CTUs. Check that the padded size is a multiple of 128 in both dimensions.
A wrong `--width` or `--height` shows up as a wrong block count.

Do not rename the block files. Their names carry the coordinates that step 6
parses.

## 5. Predict

```bash
python scripts/predict.py \
    --checkpoint best_model.pth \
    --blocks blocks/BlowingBubbles \
    --output predictions/BlowingBubbles.csv
```

Use `--device` to select the compute device.

The script prints the predicted class distribution. A plausible result has NS and
QT well represented and the ternary modes less common. Everything landing in one
class indicates a wrong checkpoint or unreadable blocks.

Predictions depend only on the source luma, not on QP, so one prediction run
serves every QP encoded later.

## 6. Build the HCPM files

```bash
python scripts/csv_to_txt.py \
    --predictions predictions/BlowingBubbles.csv \
    --output vtm_input/BlowingBubbles \
    --width 416 --height 240
```

Expected output:

```
Wrote POC 0: 8 CTUs, 2728 lines -> vtm_input/BlowingBubbles/POC_0.txt
...
All frame files have the expected line count.
```

8 CTUs because 512x256 is 4 by 2 CTUs, and 2728 because 8 times 341. Always pass
`--width` and `--height` so this check runs. A short file makes the encoder read
another CTU's decisions for every CTU after the gap, without any error.

Record the absolute path of the output directory for the next step:

```bash
cd /path/to/SwinVVC && pwd
# together with vtm_input/BlowingBubbles this gives the full path
```

## 7. Point the encoder at those files

This is the one manual edit in the process. In the patched VTM checkout, open
`source/Lib/EncoderLib/EncCu.cpp` and find line 294:

```c
sprintf(phFileName, "qp_32/POC/Traffic_2560x1600_30/POC_%d.txt", (int) cs.slice->getPOC());
```

Replace the literal with the directory from step 6, keeping `POC_%d.txt` intact:

```c
sprintf(phFileName, "/abs/path/to/SwinVVC/vtm_input/BlowingBubbles/POC_%d.txt", (int) cs.slice->getPOC());
```

An absolute path is safer. A relative one resolves against the encoder's working
directory, which is easy to get wrong.

Rebuild. Only one file changed, so this is quick:

```bash
cd build && make -j8
```

Because predictions are QP-independent, this edit is per sequence, not per QP. All
four QPs can be encoded after a single edit. It has to be repeated for the next
sequence.

The `fopen` on the following line is unchecked, so a wrong path produces no
complaint. The encoder reads an uninitialised partition array and either crashes
or emits wrong output. Verify the path before trusting a result.

## 8. Encode

```bash
cd /path/to/VTM-patched/build
mkdir -p out

./bin/EncoderAppStatic \
  -c ../cfg/encoder_intra_vtm.cfg \
  -c ../cfg/per-sequence/BlowingBubbles.cfg \
  -i /abs/path/to/SwinVVC/sequences/BlowingBubbles_416x240_50.yuv \
  -f 5 \
  -q 22 \
  -b out/bb_q22_fast.bin \
  -o out/bb_q22_fast.yuv \
  > out/bb_q22_fast.log 2>&1
```

`-f 5` is required. The per-sequence cfg sets `FramesToBeEncoded: 500`, and the
encoder would then look for `POC_5.txt` onward, which does not exist. Keep `-f` at
or below the number of POC files.

Sanity checks on the log:

- it reaches "Total Time" without aborting
- the reported resolution is 416x240
- Y-PSNR is in a sensible range, roughly 36 to 38 dB at QP 22 for this clip

Then the anchor, same arguments, unpatched binary:

```bash
cd /path/to/VTM-anchor/build
mkdir -p out
./bin/EncoderAppStatic \
  -c ../cfg/encoder_intra_vtm.cfg \
  -c ../cfg/per-sequence/BlowingBubbles.cfg \
  -i /abs/path/to/SwinVVC/sequences/BlowingBubbles_416x240_50.yuv \
  -f 5 -q 22 \
  -b out/bb_q22_anchor.bin -o out/bb_q22_anchor.yuv \
  > out/bb_q22_anchor.log 2>&1
```

The patched run should finish faster. On five frames the difference is small and
noisy, so a single QP on a short clip proves little.

## 9. Measure

Repeat step 8 at QPs 22, 27, 32 and 37 for both builds. Four points per curve is
what the Bjontegaard cubic fit requires. Read the bitrate, Y-PSNR and total time
from each log, then:

```bash
mkdir -p results
cat > results/BlowingBubbles.json <<'JSON'
{
  "sequence": "BlowingBubbles_416x240_50",
  "qps": [22, 27, 32, 37],
  "anchor":   {"rate": [...], "psnr": [...], "time": [...]},
  "proposed": {"rate": [...], "psnr": [...], "time": [...]}
}
JSON

python scripts/bd_metrics.py \
    --json results/BlowingBubbles.json \
    --plot-dir results/figures
```

This reports BD-BR, BD-PSNR, per-QP and average time saving, and the time to
bitrate ratio, and writes RD and timing plots.

Encode more than five frames before comparing against the paper. The published
figures come from full CTC sequence lengths, where startup effects do not dominate.

## Summary

```bash
# once
conda activate swinvvc
pytest -q

# per sequence
python scripts/generate_blocks.py --yuv <yuv> --width W --height H \
       --start-poc 0 --end-poc N --output blocks/<seq>
python scripts/predict.py --checkpoint best_model.pth \
       --blocks blocks/<seq> --output predictions/<seq>.csv
python scripts/csv_to_txt.py --predictions predictions/<seq>.csv \
       --output vtm_input/<seq> --width W --height H
# edit EncCu.cpp line 294 to vtm_input/<seq>, rebuild

# per QP
./bin/EncoderAppStatic -c ...cfg -i <yuv> -f <frames> -q <qp> -b out.bin -o rec.yuv
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `pytest: command not found` | pytest not installed, run `pip install pytest` |
| `No block images found` | wrong `--blocks` path, or step 4 not run |
| `Filename ... does not match the expected pattern` | block files renamed |
| `expected N lines, found M` | block directory missed CTUs, re-run step 4 for the full POC range |
| all predictions one class | wrong checkpoint, or blank blocks |
| encoder crashes immediately | wrong path in step 7, or it was not rebuilt |
| partitions appear untouched | `EncCu.cpp` edited but `make` not re-run |
| encoder fails partway through | `-f` larger than the number of POC files |
| no time saving | tracing patch applied to the encoding build |
| `git apply` fails | not a clean VTM-20.2 checkout, or already applied |
