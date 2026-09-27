"""
Verbatim copies of the original research code, kept only as a test oracle.

These functions are transcribed from the project's original notebooks
(`final_dataset_creation.ipynb`, `32x32_block_image_generator.ipynb`) with their
logic unchanged. The equivalence tests compare the refactored package against
them, which is what pins the refactor to "structure only, no methodology
change".

Do not import these from `swinvvc`, and do not tidy them - their value is that
they are unmodified.
"""

from __future__ import annotations

import csv
import os
import re

import numpy as np


# --------------------------------------------------------------------------- #
# From final_dataset_creation.ipynb
# --------------------------------------------------------------------------- #
def original_build_boundary_maps(statistics_file, width, height):
    """Original DTrace parsing + boundary rasterisation."""
    images = []
    with open(statistics_file) as f:
        image = [[0] * (width + 1) for _ in range(height + 1)]
        current_poc = 0
        for line in f:
            if line.find("Luma_IntraMode") != -1 and line.find("#") == -1:
                line_list = line.split(";")
                poc = int(line_list[1])
                if poc != current_poc:
                    images.append(image)
                    image = [[0] * (width + 1) for _ in range(height + 1)]
                    current_poc = poc
                x = int(line_list[2])
                y = int(line_list[3])
                sx = int(line_list[4])
                sy = int(line_list[5])
                for i in range(x, x + sx + 1):
                    image[y][i] = 1
                    image[y + sy][i] = 1
                for i in range(y, y + sy + 1):
                    image[i][x] = 1
                    image[i][x + sx] = 1
        images.append(image)
    return images


def original_get_split_mode(image, x, y, sx, sy):
    """Original split-mode classification."""
    hbt = True
    for i in range(sx):
        if image[y + sy // 2][x + i] == 0:
            hbt = False
            break
    vbt = True
    for i in range(sy):
        if image[y + i][x + sx // 2] == 0:
            vbt = False
            break
    if hbt and vbt:
        return 1
    elif hbt:
        return 2
    elif vbt:
        return 3
    hbt = True
    for i in range(sx):
        if image[y + sy // 4][x + i] == 0:
            hbt = False
            break
        if image[y + 3 * (sy // 4)][x + i] == 0:
            hbt = False
            break
    vbt = True
    for i in range(sy):
        if image[y + i][x + sx // 4] == 0:
            vbt = False
            break
        if image[y + i][x + 3 * (sx // 4)] == 0:
            vbt = False
            break
    if hbt:
        return 4
    elif vbt:
        return 5
    return 0


def original_label_sequence(images, width, height, block_size=32):
    """Original dataset-row generation (column-major scan)."""
    dataset = []
    for image, poc in zip(images, range(len(images))):
        for x in range(0, width + 1 - block_size, block_size):
            for y in range(0, height + 1 - block_size, block_size):
                mode = original_get_split_mode(image, x, y, block_size, block_size)
                dataset.append([poc, x, y, mode])
    return dataset


def original_get_padded_size(w, h, dw, dh):
    """Original padding helper."""
    return (w + dw - 1) // dw * dw, (h + dh - 1) // dh * dh


def original_read_y_frame(yuv_filename, width, height, frame_number=0):
    """Original luma-plane reader."""
    frame_size = width * height
    offset = int(frame_number * frame_size * 1.5)
    with open(yuv_filename, "rb") as f:
        f.seek(offset)
        y_data = f.read(frame_size)
        if len(y_data) < frame_size:
            y_data = y_data + b"\x00" * (frame_size - len(y_data))
    return np.frombuffer(y_data, dtype=np.uint8).reshape((height, width))


# --------------------------------------------------------------------------- #
# From 32x32_block_image_generator.ipynb
# --------------------------------------------------------------------------- #
def original_csv_to_txt(csv_file, out_dir):
    """
    Original prediction-CSV to HCPM-text conversion.

    Note: the original wrote ``poc<n>.txt``. The refactored converter writes
    ``POC_<n>.txt`` to match the filename the patched encoder actually builds
    (``sprintf(..., "POD_%d.txt")`` -> ``POC_%d.txt``); only the name differs,
    never the contents. Tests compare contents.
    """
    data = {}
    pattern = re.compile(r"poc(\d+).*_x(\d+)_y(\d+).png")

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            pred_val = row["predicted_class"]
            try:
                pred_val = int(pred_val)
            except Exception:
                pred_val = float(pred_val)

            m = pattern.search(filename)
            if m is None:
                continue

            poc_str, x_str, y_str = m.groups()
            poc = int(poc_str)
            x = int(x_str)
            y = int(y_str)

            block_x = x // 128
            block_y = y // 128
            subblock_col = (x % 128) // 32
            subblock_row = (y % 128) // 32

            if poc not in data:
                data[poc] = {}
            block_key = (block_y, block_x)
            if block_key not in data[poc]:
                data[poc][block_key] = []
            data[poc][block_key].append((subblock_row, subblock_col, pred_val))

    os.makedirs(out_dir, exist_ok=True)
    written = {}

    for poc, blocks in data.items():
        sorted_block_keys = sorted(blocks.keys(), key=lambda x: (x[0], x[1]))
        output_lines = []
        num_lines_per_block = 341
        prediction_line_start = 5

        for block_key in sorted_block_keys:
            subblock_list = blocks[block_key]
            subblock_list_sorted = sorted(subblock_list, key=lambda t: (t[0], t[1]))
            block_lines = ["-2"] * num_lines_per_block
            for i in range(min(16, len(subblock_list_sorted))):
                block_lines[prediction_line_start + i] = str(subblock_list_sorted[i][2])
            output_lines.extend(block_lines)

        out_filename = os.path.join(out_dir, f"poc{poc}.txt")
        with open(out_filename, "w") as out_f:
            for line in output_lines:
                out_f.write(line + "\n")
        written[poc] = out_filename

    return written


def original_bjontegaard(R1, PSNR1, R2, PSNR2, mode):
    """Original Bjontegaard metric implementation."""
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    if mode == "dsnr":
        p1 = np.polyfit(lR1, PSNR1, 3)
        p2 = np.polyfit(lR2, PSNR2, 3)
        min_int = max(np.min(lR1), np.min(lR2))
        max_int = min(np.max(lR1), np.max(lR2))
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
        return (int2 - int1) / (max_int - min_int)

    elif mode == "rate":
        p1 = np.polyfit(PSNR1, lR1, 3)
        p2 = np.polyfit(PSNR2, lR2, 3)
        min_int = max(np.min(PSNR1), np.min(PSNR2))
        max_int = min(np.max(PSNR1), np.max(PSNR2))
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
        avg_exp_diff = (int2 - int1) / (max_int - min_int)
        return (np.exp(avg_exp_diff) - 1) * 100

    else:
        raise ValueError("Invalid mode. Use 'dsnr' or 'rate'.")
