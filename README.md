# SwinVVC: Vision Transformer CU Partitioning for VVC

> **Intra-Mode Encoding Complexity Reduction: Vision Transformer-Driven CU Partitioning for VVC**

VVC reduces bitrate by about 40% compared to HEVC, but encoding takes 10 to 16
times longer. Much of that cost is the rate-distortion optimisation (RDO) search
over the quad-tree plus multi-type tree (QTMT) partition candidates.

This project replaces that search at the 32x32 luma CU level. A Swin Transformer
predicts one of six partition modes per CU, and a patched VTM encoder tests only
that candidate instead of all six.

Results on VTM-20.2, All-Intra, QPs {22, 27, 32, 37}, 21 sequences:

| Metric | Value |
|---|---|
| Encoding time reduction | 49.01% average, 70.55% best case |
| BD-BR | +1.85% |
| BD-PSNR | -0.07 dB |
| dSSIM | 0.31% |
| dVMAF | 0.13% |

Only 32x32 CUs are targeted. 128x128 CTUs are always quad-tree split in intra
coding, and 64x64 CUs allow only no-split or QT. 32x32 is the only size that
allows all six modes, so it accounts for most of the RDO cost.

## Partition classes

| Class | Name | Result |
|---|---|---|
| 0 | NS | no split |
| 1 | QT | 4 x 16x16 |
| 2 | BTH | 2 x 32x16 |
| 3 | BTV | 2 x 16x32 |
| 4 | TTH | 32x8, 32x16, 32x8 |
| 5 | TTV | 8x32, 16x32, 8x32 |

See [docs/partition_classes.md](docs/partition_classes.md).

## Pipeline

Training:

```
YUV -> VTM encode with tracing -> DTrace stats -> labels -> 32x32 blocks
    -> balance classes -> train -> best_model.pth
```

Inference:

```
YUV -> 32x32 blocks -> predictions.csv -> POC_<n>.txt -> patched VTM -> bitstream
```

Two VTM builds are needed: one with tracing enabled for label extraction, one
with the partition patch for fast encoding. See
[docs/vtm_integration.md](docs/vtm_integration.md).

## Installation

```bash
git clone https://github.com/noman-amin-n/SwinVVC.git
cd SwinVVC
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate swinvvc
```

Python 3.9 or newer.

`pip install -e .` is optional. The scripts add the repository root to
`sys.path` themselves.

Label extraction, block extraction, CSV to TXT conversion and BD metrics import
torch lazily, so they work with only NumPy, OpenCV, Pillow and Matplotlib
installed.

### VTM

VTM is not bundled. Clone it and apply the patch for the required build:

```bash
git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
cd VVCSoftware_VTM && git checkout VTM-20.2

# fast encoder
git apply --whitespace=nowarn /path/to/SwinVVC/vtm_patch/01-partition-override.patch

# label extraction, in a separate checkout
git apply /path/to/SwinVVC/vtm_patch/02-block-statistics-tracing.patch
```

`01-partition-override.patch` reads prediction files from a path hard-coded in
`EncCu.cpp` that must be edited for each sequence and QP. The file open is
unchecked, so a wrong path fails silently. See
[docs/vtm_integration.md](docs/vtm_integration.md) section 3.1.

## Inference with a trained checkpoint

```bash
SEQ=BlowingBubbles_416x240_50
W=416; H=240

python scripts/generate_blocks.py \
    --yuv "sequences/${SEQ}.yuv" --width $W --height $H \
    --start-poc 0 --end-poc 4 --output "blocks/${SEQ}"

python scripts/predict.py \
    --checkpoint best_model.pth \
    --blocks "blocks/${SEQ}" --output "predictions/${SEQ}.csv"

python scripts/csv_to_txt.py \
    --predictions "predictions/${SEQ}.csv" \
    --output "vtm_input/${SEQ}" --width $W --height $H
```

Then point `EncCu.cpp` at `vtm_input/${SEQ}`, rebuild, and encode. Details in
[docs/inference.md](docs/inference.md).

## Dataset generation

Encode a sequence with unmodified VTM and capture the block statistics trace
([docs/vtm_integration.md](docs/vtm_integration.md) section 2), then:

```bash
python scripts/extract_labels.py \
    --statistics traces/${SEQ}_qp22.csv --width $W --height $H \
    --output labels/${SEQ}.csv --render-dir figures/${SEQ}

python scripts/build_dataset.py \
    --yuv sequences/${SEQ}.yuv --labels labels/${SEQ}.csv \
    --width $W --height $H --sequence-name ${SEQ} --output datasets/raw

python scripts/balance_dataset.py \
    --source datasets/raw --output datasets/balanced \
    --val-output datasets/validation --val-per-class 1500
```

See [docs/dataset_generation.md](docs/dataset_generation.md).

## Training

```bash
python scripts/train.py \
    --train-path datasets/balanced --val-path datasets/validation \
    --output-dir runs/swin_base
```

See [docs/training.md](docs/training.md).

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint runs/swin_base/best_model.pth --val-path datasets/validation \
    --plot-dir runs/swin_base/figures --json-out runs/swin_base/metrics.json

python scripts/bd_metrics.py --json results/${SEQ}.json --plot-dir results/figures
```

## Layout

```
swinvvc/
  config.py              Config dataclass, loadable from YAML
  constants.py           class ids, block and CTU geometry, HCPM layout
  data/
    yuv.py               raw YUV reading and padding
    dtrace.py            VTM block statistics parsing
    labels.py            split mode recovery from CU boundary maps
    blocks.py            32x32 block extraction
    balance.py           class balancing by undersampling
    datasets.py          transforms and data loaders
    visualize.py         wireframe and label map rendering
  models/
    swin.py              Swin Transformer backbone, checkpoint loading
    ema.py               exponential moving average of weights
  training/
    mixup.py             MixUp and CutMix
    engine.py            train and validate loops
    evaluate.py          confusion matrix, per-class metrics, ROC
  inference/
    predict.py           block level prediction
    csv_to_txt.py        predictions to HCPM files
  utils/
    bjontegaard.py       BD-BR, BD-PSNR, time saving
    plots.py             figures
    seed.py              seeding

scripts/
  extract_labels.py      DTrace to labels
  build_dataset.py       labels and YUV to ImageFolder
  balance_dataset.py     undersample and split validation
  train.py               training
  evaluate.py            classification metrics
  generate_blocks.py     YUV to inference blocks
  predict.py             blocks to predictions CSV
  csv_to_txt.py          CSV to POC_<n>.txt
  bd_metrics.py          BD-BR, BD-PSNR, time saving

vtm_patch/               VTM-20.2 patches
docs/                    setup and usage documentation
configs/                 default hyper-parameters, test sequence list
tests/                   test suite
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Built on

The [VVC reference software](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM)
and [Swin Transformer](https://github.com/microsoft/Swin-Transformer) through
[timm](https://github.com/huggingface/pytorch-image-models). Training data comes
from [Inter4K](https://alexandrosstergiou.github.io/datasets/Inter4K/index.html).
