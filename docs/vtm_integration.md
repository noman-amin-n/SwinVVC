# VTM Integration

## Two builds

| Build | Purpose | Partition patch | Tracing |
|---|---|---|---|
| A | label extraction for the dataset | not applied | on |
| B | fast encoding with predictions | applied | off |

Build A must run the unmodified RDO search, otherwise the labels would describe
the model's own decisions instead of the optimum. Build B must have tracing off,
since `ENABLE_TRACING` costs a noticeable fraction of runtime and would affect
the encoding time measurements.

## 1. Get VTM-20.2

```bash
git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
cd VVCSoftware_VTM
git checkout VTM-20.2
```

## 2. Build A: label extraction

VTM does not write partition decisions directly, so labels are recovered from the
geometry of the CUs the RDO search selects. That geometry comes from the block
statistics trace.

Apply the tracing patch:

```bash
git apply /path/to/SwinVVC/vtm_patch/02-block-statistics-tracing.patch
```

It sets `ENABLE_TRACING` and `BLOCK_STATS_AS_CSV` to 1 in
`source/Lib/CommonLib/TypeDef.h`, and comments out an `assert(0)` in
`dtrace_blockstatistics.cpp` that otherwise aborts the encoder on the chroma
intra mode tracing path.

`BLOCK_STATS_AS_CSV` makes each record a
`BlockStat;<poc>;<x>;<y>;<w>;<h>;<statistic>;<value>` line, which is the format
`swinvvc.data.dtrace` parses. Without it the trace uses a human readable format
the parser does not accept.

Do not apply the partition override patch to this build. Label extraction must
run the encoder's own unmodified RDO search.

Build:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

Encode with full RDO and capture the trace:

```bash
./bin/EncoderAppStatic \
  -c ../cfg/encoder_intra_vtm.cfg \
  -c ../cfg/per-sequence/BlowingBubbles.cfg \
  -i BlowingBubbles_416x240_50.yuv \
  -q 22 -f 5 \
  -b out.bin -o rec.yuv \
  --TraceFile=BlowingBubbles_416x240_50_qp22.csv \
  --TraceRule="D_BLOCK_STATISTICS_CODED:poc>=0"
```

The resulting CSV is the input to `scripts/extract_labels.py`. Traces are large,
tens to hundreds of MB per sequence, since every coded CU emits several
statistics. Only `Luma_IntraMode` records are read.

## 3. Build B: fast encoder

Apply the patch to a clean VTM-20.2 checkout:

```bash
cd VVCSoftware_VTM
git apply --check                    /path/to/SwinVVC/vtm_patch/01-partition-override.patch
git apply --whitespace=nowarn        /path/to/SwinVVC/vtm_patch/01-partition-override.patch
```

`--whitespace=nowarn` is needed because a few added lines carry trailing
whitespace from the original source.

Leave `ENABLE_TRACING` at its default of 0 for this build, meaning do not apply
the tracing patch here.

### 3.1. Set the HCPM input path

The patch reads prediction files from a path hard-coded in
`source/Lib/EncoderLib/EncCu.cpp`, inside `EncCu::compressCtu`:

```c
sprintf(phFileName, "qp_32/POC/Traffic_2560x1600_30/POC_%d.txt", (int) cs.slice->getPOC());
```

Edit this for every sequence encoded. It is relative to the encoder's
working directory and must point at the directory `scripts/csv_to_txt.py` wrote,
the one holding `POC_0.txt`, `POC_1.txt` and so on.

There is no error handling on this `fopen`. If the path is wrong, `fpCUDepth` is
`NULL`, the following `fscanf` reads nothing, and the encoder runs with an
uninitialised partition array. That either crashes or produces wrong output
without any message. Check the path before every run.

### 3.2. Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

## 4. Running the fast encoder

```bash
./bin/EncoderAppStatic \
  -c ../cfg/encoder_intra_vtm.cfg \
  -c ../cfg/per-sequence/BlowingBubbles.cfg \
  -i BlowingBubbles_416x240_50.yuv \
  -q 22 -f 5 \
  -b out_fast.bin -o rec_fast.yuv
```

The working directory must be the one the hard-coded path is relative to.

Use the All-Intra configuration at QPs {22, 27, 32, 37}, per the JVET common test
conditions. Compare against the same encode from an unpatched build to get BD-BR,
BD-PSNR and the encoding time saving with `scripts/bd_metrics.py`.

## 5. What the patch changes

The override lives in seven regions across four files, each delimited in the
source by `// 310723-tanvir added` and `// 310723-tanvir ended`. The tracing
patch changes two further files and is described in section 5.6.

### 5.1. `source/Lib/CommonLib/Picture.h`

Stores the per-frame HCPM array on the `Picture` object so the CU level code can
reach it.

```c
void setCUDepth(int* pCUDepth) { m_pCUDepth = pCUDepth; }
int* getCUDepth() { return m_pCUDepth; }
...
int*  m_pCUDepth;   // predicted CU partition map (HCPM format) for one frame
```

### 5.2. `source/Lib/EncoderLib/EncModeCtrl.h`

Declares the methods needed to replace the encoder's candidate list:

```c
static_vector<ComprCUCtx, (MAX_CU_DEPTH << 2)>& getVecCompr();
void ChangeTestMode(int mode);
void Modify(CodingStructure* cs, Partitioner& partitioner);
```

### 5.3. `source/Lib/EncoderLib/EncModeCtrl.cpp`

`ChangeTestMode(int mode)` clears the list of partition modes to be tested and
pushes back only the predicted one:

| `mode` | `EncTestModeType` |
|---|---|
| 0 | `ETM_INTRA` |
| 1 | `ETM_SPLIT_QT` |
| 2 | `ETM_SPLIT_BT_H` |
| 3 | `ETM_SPLIT_BT_V` |
| 4 | `ETM_SPLIT_TT_H` |
| 5 | `ETM_SPLIT_TT_V` |

Any other value falls through to `ETM_INTRA`, which is how the `-2` filler
behaves. The candidate list ends up with one entry instead of six, so the RDO
loop evaluates one partitioning instead of all of them. That is where the time is
saved.

`Modify()` restores the full six-mode list if the chosen mode turns out not to be
allowed at this position, for example a split that would cross the picture
boundary.

### 5.4. `source/Lib/EncoderLib/EncCu.cpp`

Three regions.

A file-scope `int current_ctu_address;` records which CTU is being compressed so
the CU level code can index the HCPM array.

In `EncCu::compressCtu`, on the first CTU of a frame (`ctuRsAddr == 0`), the whole
`POC_<n>.txt` file is read into a
`validWidthInCTU * validHeightInCTU * 341` integer array and attached to the
`Picture` with `setCUDepth`.

In `EncCu::xCompressCU`, the override is guarded by:

```c
if ((partitioner.chType == ChannelType::LUMA)
    && (currCsArea.lumaSize().height == 32)
    && (currCsArea.lumaSize().width == 32))
```

so only 32x32 luma CUs are affected and every other size keeps the full RDO
search. Inside the guard the HCPM index is computed:

```c
indexIn21Data = 5 + uhLeftInCTU / 32 + uhTopInCTU / 32 * 4;
res = plfProbArray[341 * iCTUAddr + indexIn21Data];
```

The mode is checked with `tryModeMaster`, and on success `ChangeTestMode(res)`
collapses the candidate list. CUs at multi-type depth 3, 4x4 CUs, and CUs
extending past the picture boundary are skipped.

### 5.5. HCPM record layout

One record of 341 integers per CTU, CTUs in raster order (`ctuRsAddr`). Within a
record the entries are ordered by CU size, each level in raster order:

| Index | Count | CU size | Written by this project |
|---|---|---|---|
| 0 | 1 | 128x128 | `-2` |
| 1 to 4 | 4 | 64x64 | `-2` |
| 5 to 20 | 16 | 32x32 | predicted class 0 to 5 |
| 21 to 84 | 64 | 16x16 | `-2` |
| 85 to 340 | 256 | 8x8 | `-2` |

1 + 4 + 16 + 64 + 256 = 341. The `-2` filler maps to `ETM_INTRA` in
`ChangeTestMode`, which leaves those levels to the encoder's own decision.

`scripts/csv_to_txt.py --width W --height H` checks that each output file holds
exactly `ceil(W/128) * ceil(H/128) * 341` lines. A wrong line count means every
CTU after the discrepancy reads another CTU's decisions.

### 5.6. The tracing patch

`02-block-statistics-tracing.patch` covers two files:

| File | Change |
|---|---|
| `source/Lib/CommonLib/TypeDef.h` | `ENABLE_TRACING` 0 to 1, `BLOCK_STATS_AS_CSV` 0 to 1 |
| `source/Lib/CommonLib/dtrace_blockstatistics.cpp` | comments out `assert(0)` on the chroma intra mode path |

The assert fires when block statistics are traced for a dual-tree chroma intra
mode and aborts the encoder, so it has to go for a trace run to complete.

## 6. Verifying the patches

Both patches were generated by diffing the official VTM-20.2 release against the
working tree used for the paper. Applying both to a clean checkout reproduces
that tree byte for byte, and those six files are the only differences in the
whole repository.

```bash
git apply --check vtm_patch/01-partition-override.patch
git apply --check vtm_patch/02-block-statistics-tracing.patch
```

## 7. Troubleshooting

| Symptom | Cause |
|---|---|
| Crash early, or garbage output | wrong HCPM path, `fopen` returned `NULL` (3.1) |
| Time saving well below expectations | `ENABLE_TRACING` left at 1 in build B |
| Partitions ignored for some CTUs | HCPM file short, check the line count (5.5) |
| Wrong CTU count | blocks padded to 32 instead of 128, use `scripts/generate_blocks.py` |
| `git apply` fails | not a clean VTM-20.2 checkout, or already applied |
