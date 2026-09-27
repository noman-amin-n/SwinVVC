"""
Unit tests for the pipeline contract: HCPM layout, block geometry, config.

These cover the invariants that must hold for the encoder to read the right
partition decisions, independently of the original implementation.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from swinvvc.config import Config
from swinvvc.constants import (
    BLOCK_SIZE,
    BLOCKS_PER_CTU,
    CLASS_COLORS,
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    CTU_SIZE,
    HCPM_ENTRIES_PER_CTU,
    HCPM_OFFSET_8,
    HCPM_OFFSET_16,
    HCPM_OFFSET_32,
    HCPM_OFFSET_64,
    HCPM_OFFSET_128,
    HCPM_UNUSED,
    NUM_CLASSES,
)
from swinvvc.inference.csv_to_txt import (
    build_frame_lines,
    convert,
    index_predictions,
    parse_block_filename,
    validate_frame_file,
)


class TestConstants:
    def test_hcpm_levels_sum_to_record_size(self):
        """1 + 4 + 16 + 64 + 256 == 341, the size EncCu.cpp allocates per CTU."""
        assert 1 + 4 + 16 + 64 + 256 == HCPM_ENTRIES_PER_CTU

    def test_hcpm_offsets_are_consecutive_levels(self):
        """Each level must start where the previous one ends."""
        assert HCPM_OFFSET_128 == 0
        assert HCPM_OFFSET_64 == HCPM_OFFSET_128 + 1
        assert HCPM_OFFSET_32 == HCPM_OFFSET_64 + 4
        assert HCPM_OFFSET_16 == HCPM_OFFSET_32 + 16
        assert HCPM_OFFSET_8 == HCPM_OFFSET_16 + 64
        assert HCPM_OFFSET_8 + 256 == HCPM_ENTRIES_PER_CTU

    def test_blocks_per_ctu(self):
        assert BLOCKS_PER_CTU == (CTU_SIZE // BLOCK_SIZE) ** 2 == 16

    def test_class_tables_are_complete(self):
        assert len(CLASS_NAMES) == NUM_CLASSES
        assert len(CLASS_DESCRIPTIONS) == NUM_CLASSES
        assert set(CLASS_COLORS) == set(range(NUM_CLASSES))

    def test_unused_filler_is_not_a_class(self):
        """`-2` must not collide with a real class id."""
        assert HCPM_UNUSED not in range(NUM_CLASSES)


class TestFilenameParsing:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("Seq_poc0_x0_y0.png", (0, 0, 0)),
            ("Seq_poc3_x128_y256.png", (3, 128, 256)),
            ("A_B_C_1920x1080_poc12_x3808_y2144.png", (12, 3808, 2144)),
            ("ParkRunning3_3840x2160_50_poc0_x32_y0.PNG", (0, 32, 0)),
        ],
    )
    def test_parses_valid_names(self, filename, expected):
        assert parse_block_filename(filename) == expected

    @pytest.mark.parametrize(
        "filename",
        ["no_coordinates.png", "Seq_x0_y0.png", "Seq_poc0_x0.png", "Seq_poc0_x0_y0.jpg"],
    )
    def test_rejects_invalid_names(self, filename):
        assert parse_block_filename(filename) is None


class TestHcpmAssembly:
    def test_record_structure(self):
        """
        A single CTU must yield exactly 341 entries with predictions at 5..20.

        This is the invariant the encoder depends on:
        `indexIn21Data = 5 + uhLeftInCTU/32 + uhTopInCTU/32 * 4`.
        """
        # Sixteen sub-blocks with distinguishable class ids (cycling 0..5).
        subblocks = [
            (row, col, (row * 4 + col) % NUM_CLASSES)
            for row in range(4)
            for col in range(4)
        ]
        lines = build_frame_lines({(0, 0): subblocks}, poc=0, verbose=False)

        assert len(lines) == HCPM_ENTRIES_PER_CTU

        # Levels the model does not predict stay at the filler value.
        for index in range(0, HCPM_OFFSET_32):
            assert lines[index] == str(HCPM_UNUSED)
        for index in range(HCPM_OFFSET_32 + BLOCKS_PER_CTU, HCPM_ENTRIES_PER_CTU):
            assert lines[index] == str(HCPM_UNUSED)

        # The 32x32 level carries the predictions in raster order.
        for offset, (row, col, class_id) in enumerate(subblocks):
            assert lines[HCPM_OFFSET_32 + offset] == str(class_id)

    def test_ctus_emitted_in_raster_order(self, tmp_path):
        """
        CTU records must follow ctuRsAddr order (row-major), not insertion order.

        The CSV below lists CTUs deliberately out of order.
        """
        csv_path = tmp_path / "preds.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "predicted_class"])
            # Marker per CTU, keyed by (ctu_x, ctu_y) and numbered by the raster
            # position the converter must place it at. Listed out of order here.
            order = [((128, 128), 4), ((128, 0), 2), ((0, 128), 3), ((0, 0), 1)]
            for (ctu_x, ctu_y), marker in order:
                for dy in range(0, CTU_SIZE, BLOCK_SIZE):
                    for dx in range(0, CTU_SIZE, BLOCK_SIZE):
                        writer.writerow(
                            [f"S_poc0_x{ctu_x + dx}_y{ctu_y + dy}.png", marker]
                        )

        written = convert(csv_path, tmp_path / "out", verbose=False)
        values = open(written[0], encoding="utf-8").read().split()

        assert len(values) == 4 * HCPM_ENTRIES_PER_CTU
        # Raster order is (y, x): (0,0)=1, (0,128)=2, (128,0)=3, (128,128)=4.
        for position, marker in enumerate([1, 2, 3, 4]):
            base = position * HCPM_ENTRIES_PER_CTU + HCPM_OFFSET_32
            assert values[base : base + BLOCKS_PER_CTU] == [str(marker)] * BLOCKS_PER_CTU

    def test_subblock_order_within_ctu(self, tmp_path):
        """
        Within a CTU, slot 5+k must correspond to row-major sub-block k.

        Verified by giving each sub-block a class id equal to `k % 6`.
        """
        csv_path = tmp_path / "preds.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "predicted_class"])
            for row in range(4):
                for col in range(4):
                    k = row * 4 + col
                    writer.writerow(
                        [f"S_poc0_x{col * BLOCK_SIZE}_y{row * BLOCK_SIZE}.png", k % NUM_CLASSES]
                    )

        written = convert(csv_path, tmp_path / "out", verbose=False)
        values = open(written[0], encoding="utf-8").read().split()

        for k in range(BLOCKS_PER_CTU):
            assert values[HCPM_OFFSET_32 + k] == str(k % NUM_CLASSES)


class TestValidation:
    def test_accepts_complete_file(self, synthetic_predictions, tmp_path):
        csv_path, width, height = synthetic_predictions
        written = convert(csv_path, tmp_path / "out", verbose=False)

        for path in written.values():
            ok, message = validate_frame_file(path, width, height)
            assert ok, message

    def test_rejects_truncated_file(self, synthetic_predictions, tmp_path):
        """A short file would make the encoder read another CTU's decisions."""
        csv_path, width, height = synthetic_predictions
        written = convert(csv_path, tmp_path / "out", verbose=False)

        path = written[0]
        lines = open(path, encoding="utf-8").read().split("\n")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines[:-50]))

        ok, message = validate_frame_file(path, width, height)
        assert not ok
        assert "expected" in message

    @pytest.mark.parametrize(
        "width,height,expected_ctus",
        [(416, 240, 4 * 2), (1920, 1080, 15 * 9), (3840, 2160, 30 * 17), (128, 128, 1)],
    )
    def test_ctu_grid_accounts_for_padding(self, width, height, expected_ctus, tmp_path):
        """
        The CTU count must use ceiling division, matching validWidthInCTU in
        EncCu.cpp.
        """
        csv_path = tmp_path / "preds.csv"
        ctus_wide = (width + CTU_SIZE - 1) // CTU_SIZE
        ctus_high = (height + CTU_SIZE - 1) // CTU_SIZE
        assert ctus_wide * ctus_high == expected_ctus

        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "predicted_class"])
            for y in range(0, ctus_high * CTU_SIZE, BLOCK_SIZE):
                for x in range(0, ctus_wide * CTU_SIZE, BLOCK_SIZE):
                    writer.writerow([f"S_poc0_x{x}_y{y}.png", 0])

        written = convert(csv_path, tmp_path / "out", verbose=False)
        ok, message = validate_frame_file(written[0], width, height)
        assert ok, message


class TestIndexPredictions:
    def test_raises_on_missing_columns(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("file,klass\na.png,1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="missing column"):
            index_predictions(path)

    def test_strict_mode_raises_on_bad_filename(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("filename,predicted_class\nnonsense.png,1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="does not match"):
            index_predictions(path, strict=True)

    def test_lenient_mode_skips_bad_filename(self, tmp_path):
        path = tmp_path / "mixed.csv"
        path.write_text(
            "filename,predicted_class\nnonsense.png,1\nS_poc0_x0_y0.png,3\n",
            encoding="utf-8",
        )

        data = index_predictions(path, strict=False)
        assert data == {0: {(0, 0): [(0, 0, 3)]}}


class TestConfig:
    def test_defaults_match_reported_hyperparameters(self):
        config = Config()
        assert config.batch_size == 16
        assert config.num_epochs == 12
        assert config.lr == pytest.approx(3e-5)
        assert config.weight_decay == pytest.approx(0.05)
        assert config.mixup_alpha == pytest.approx(0.8)
        assert config.min_lr == pytest.approx(1e-6)
        assert config.grad_clip == pytest.approx(1.0)
        assert config.num_classes == NUM_CLASSES
        assert config.input_size == 224
        assert config.model_name == "swin_base_patch4_window7_224"
        assert config.use_mixup is True
        assert config.use_cutmix is False

    def test_roundtrips_through_yaml(self, tmp_path):
        original = Config(train_path="a", val_path="b", num_epochs=3)
        path = tmp_path / "config.yaml"
        original.save_yaml(path)

        assert Config.from_yaml(path).to_dict() == original.to_dict()

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown configuration keys"):
            Config.from_dict({"not_a_real_key": 1})

    def test_checkpoint_path_composition(self):
        config = Config(output_dir="runs/x", checkpoint_name="w.pth")
        assert str(config.checkpoint_path) == "runs/x/w.pth"


class TestBlockGeometry:
    def test_inference_padding_is_ctu_aligned(self, tmp_path):
        """
        Inference blocks must tile whole CTUs.

        A 416x240 frame pads to 512x256 (4x2 CTUs) -> 128 blocks per frame. If it
        padded to the block size instead (416x240 -> 416x240, 13x7.5) the HCPM
        indices would not line up with the encoder's CTU raster.
        """
        import cv2

        from swinvvc.data.blocks import extract_inference_blocks

        width, height = 416, 240
        yuv = tmp_path / "s.yuv"
        with open(yuv, "wb") as handle:
            handle.write(bytes(int(width * height * 1.5)))

        written = extract_inference_blocks(
            yuv, tmp_path / "blocks", width, height, "S", 0, 0
        )

        assert written == (512 // BLOCK_SIZE) * (256 // BLOCK_SIZE) == 128

        # Every block must be exactly 32x32, including those in the padded region.
        for path in (tmp_path / "blocks").iterdir():
            assert cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape == (
                BLOCK_SIZE,
                BLOCK_SIZE,
            )
