"""
Project-wide constants for the SwinVVC CU-partition predictor.

These values define the contract between the four stages of the pipeline
(label extraction, training, inference, VTM integration).  Changing any of
them requires a matching change in the VTM patch.
"""

# --------------------------------------------------------------------------- #
# Partition classes
# --------------------------------------------------------------------------- #
# The six mutually exclusive split modes a 32x32 luma CU may take in VVC
# intra coding.  The integer values are the class indices used by the network,
# by the label CSV files, by the ImageFolder directory names, and by the
# `.txt` files consumed by the patched encoder.
NO_SPLIT = 0   # NS  - CU is coded as-is (no further partitioning)
QT       = 1   # QT  - quad-tree split into four 16x16 CUs
BT_H     = 2   # BTH - binary tree, horizontal (two 32x16 CUs)
BT_V     = 3   # BTV - binary tree, vertical   (two 16x32 CUs)
TT_H     = 4   # TTH - ternary tree, horizontal (32x8, 32x16, 32x8)
TT_V     = 5   # TTV - ternary tree, vertical   (8x32, 16x32, 8x32)

NUM_CLASSES = 6

#: Short names, indexed by class id.  Used in plots and reports.
CLASS_NAMES = ["NS", "QT", "BTH", "BTV", "TTH", "TTV"]

#: Human-readable names, indexed by class id.
CLASS_DESCRIPTIONS = [
    "No Split",
    "Quad Tree",
    "Binary Tree Horizontal",
    "Binary Tree Vertical",
    "Ternary Tree Horizontal",
    "Ternary Tree Vertical",
]

#: RGB overlay colours used when rendering label maps (see data.visualize).
CLASS_COLORS = {
    NO_SPLIT: (240, 240, 240),  # light gray
    QT:       (255,  99,  71),  # tomato
    BT_H:     ( 50, 205,  50),  # lime green
    BT_V:     ( 30, 144, 255),  # dodger blue
    TT_H:     (255, 215,   0),  # gold
    TT_V:     (147, 112, 219),  # medium purple
}

# --------------------------------------------------------------------------- #
# Block / CTU geometry
# --------------------------------------------------------------------------- #
#: Side length of the CUs this work predicts for.  32x32 is the only CU size
#: that admits all six partition modes, which is why it is the target: a
#: 128x128 CTU is always QT-split, and a 64x64 CU only admits NS or QT.
BLOCK_SIZE = 32

#: VVC CTU side length.  Frames are zero-padded to a multiple of this value
#: before being cut into blocks, so that the block grid aligns with the CTU
#: grid the encoder walks.
CTU_SIZE = 128

#: Number of 32x32 sub-blocks inside one 128x128 CTU (4 x 4).
BLOCKS_PER_CTU = (CTU_SIZE // BLOCK_SIZE) ** 2  # 16

# --------------------------------------------------------------------------- #
# HCPM (Hierarchical CU Partition Map) layout
# --------------------------------------------------------------------------- #
# The patched encoder reads one flat integer array per frame, laid out as
# `HCPM_ENTRIES_PER_CTU` consecutive entries per CTU in raster order.  Within
# one CTU the entries are ordered by CU size, each level in raster order:
#
#   index   0       : 1   x 128x128
#   index   1 -  4  : 4   x  64x64
#   index   5 - 20  : 16  x  32x32   <-- the only level this model predicts
#   index  21 - 84  : 64  x  16x16
#   index  85 - 340 : 256 x   8x8
#
# Entries the model does not predict are filled with `HCPM_UNUSED`.  The
# offsets below mirror `indexIn21Data` in the patched EncCu.cpp.
HCPM_ENTRIES_PER_CTU = 341

HCPM_OFFSET_128 = 0
HCPM_OFFSET_64  = 1
HCPM_OFFSET_32  = 5    # first 32x32 entry
HCPM_OFFSET_16  = 21
HCPM_OFFSET_8   = 85

#: Filler written for every HCPM entry the model does not predict.  The
#: patched encoder maps this value to ETM_INTRA, i.e. "fall back to the
#: encoder's own unmodified decision".
HCPM_UNUSED = -2

# --------------------------------------------------------------------------- #
# Model input
# --------------------------------------------------------------------------- #
#: Side length the 32x32 luma block is bilinearly upsampled to, so that
#: ImageNet-pretrained Swin weights can be used unchanged.
INPUT_SIZE = 224

#: Normalisation statistics.  Blocks are single-channel luma replicated to
#: three channels, so a symmetric [-1, 1] mapping is used rather than the
#: ImageNet per-channel statistics.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
