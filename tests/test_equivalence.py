"""
Equivalence tests: the refactored package must match the original research code.

These are the tests that matter most for this repository. The refactor was
permitted to change structure, naming and documentation, but not methodology, so
each test pins a refactored component against the verbatim original in
`tests/reference_implementation.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from reference_implementation import (
    original_bjontegaard,
    original_build_boundary_maps,
    original_csv_to_txt,
    original_get_padded_size,
    original_get_split_mode,
    original_label_sequence,
    original_read_y_frame,
)

from swinvvc.data.dtrace import build_boundary_maps
from swinvvc.data.labels import get_split_mode, label_sequence
from swinvvc.data.yuv import get_padded_size, read_y_frame
from swinvvc.inference.csv_to_txt import convert
from swinvvc.utils.bjontegaard import bd_psnr, bd_rate, time_saving


# --------------------------------------------------------------------------- #
# Measurements from the original notebook (BasketballPass_416x240_50)
# --------------------------------------------------------------------------- #
ANCHOR_PSNR = [44.2371, 40.7483, 37.416, 34.4957, 31.8203]
ANCHOR_RATE = [4962.96, 2929.6, 1632.64, 887.2, 494.16]
ANCHOR_TIME = [744.642, 560.126, 418.351, 274.357, 160.16]
TEST_PSNR = [44.2348, 40.7197, 37.3674, 34.4777, 31.7042]
TEST_RATE = [4996.88, 2958.72, 1658.72, 918.88, 518.88]
TEST_TIME = [437.315, 313.821, 222.951, 143.831, 84.323]


class TestBoundaryMaps:
    def test_matches_original(self, synthetic_trace):
        """Rasterised CU boundary maps must be identical to the original."""
        path, width, height, _ = synthetic_trace

        original = original_build_boundary_maps(path, width, height)
        refactored = build_boundary_maps(path, width, height)

        assert len(refactored) == len(original)
        for index, (expected, actual) in enumerate(zip(original, refactored)):
            assert actual == expected, f"boundary map {index} differs"


class TestLabels:
    def test_label_sequence_matches_original(self, synthetic_trace):
        """The full label table must be identical, row for row."""
        path, width, height, _ = synthetic_trace

        maps = build_boundary_maps(path, width, height)
        original = original_label_sequence(maps, width, height, 32)
        refactored = [list(row) for row in label_sequence(maps, width, height, 32)]

        assert refactored == original

    def test_recovers_ground_truth(self, synthetic_trace):
        """
        Labels recovered from the trace must match the modes used to build it.

        This is the stronger check: it validates the decision rule itself, not
        just agreement with the original code.
        """
        path, width, height, truth = synthetic_trace

        maps = build_boundary_maps(path, width, height)
        rows = label_sequence(maps, width, height, 32)

        assert len(rows) == len(truth)
        for poc, x, y, mode in rows:
            assert mode == truth[(poc, x, y)], f"wrong label at poc={poc} x={x} y={y}"

    @pytest.mark.parametrize("mode", [0, 1, 2, 3, 4, 5])
    def test_get_split_mode_matches_original(self, synthetic_trace, mode):
        """Per-region classification must agree with the original on every mode."""
        path, width, height, truth = synthetic_trace
        maps = build_boundary_maps(path, width, height)

        checked = 0
        for (poc, x, y), true_mode in truth.items():
            if true_mode != mode:
                continue
            boundary = maps[poc]
            assert get_split_mode(boundary, x, y, 32, 32) == original_get_split_mode(
                boundary, x, y, 32, 32
            )
            checked += 1

        assert checked > 0, f"no synthetic regions with mode {mode}"


class TestYuv:
    @pytest.mark.parametrize(
        "width,height,block",
        [(416, 240, 32), (416, 240, 128), (3840, 2160, 128), (1920, 1080, 128), (64, 64, 32)],
    )
    def test_padded_size_matches_original(self, width, height, block):
        assert get_padded_size(width, height, block, block) == original_get_padded_size(
            width, height, block, block
        )

    def test_read_y_frame_matches_original(self, tmp_path):
        """Luma extraction must be byte-identical, including the frame offset."""
        width, height = 64, 48
        rng = np.random.default_rng(11)
        path = tmp_path / "test.yuv"

        frames = []
        with open(path, "wb") as handle:
            for _ in range(3):
                luma = rng.integers(0, 256, (height, width), dtype=np.uint8)
                frames.append(luma)
                handle.write(luma.tobytes())
                # 4:2:0 chroma planes: half resolution each.
                handle.write(rng.integers(0, 256, (height // 2 * width // 2 * 2,),
                                          dtype=np.uint8).tobytes())

        for poc in range(3):
            expected = original_read_y_frame(path, width, height, poc)
            actual = read_y_frame(path, width, height, poc)
            assert np.array_equal(actual, expected)
            assert np.array_equal(actual, frames[poc]), f"frame {poc} content mismatch"

    def test_read_y_frame_returns_writable_array(self, tmp_path):
        """
        The returned array must be writable.

        `np.frombuffer` yields a read-only view; block extraction pads in place,
        so the copy is load-bearing.
        """
        width, height = 32, 32
        path = tmp_path / "one.yuv"
        with open(path, "wb") as handle:
            handle.write(bytes(int(width * height * 1.5)))

        frame = read_y_frame(path, width, height, 0)
        frame[0, 0] = 42  # must not raise
        assert frame[0, 0] == 42

    def test_read_y_frame_zero_fills_past_eof(self, tmp_path):
        """A truncated file must zero-fill rather than raise."""
        width, height = 32, 32
        path = tmp_path / "short.yuv"
        with open(path, "wb") as handle:
            handle.write(b"\xff" * 100)  # far less than one frame

        frame = read_y_frame(path, width, height, 0)
        assert frame.shape == (height, width)
        assert frame.flat[0] == 255
        assert frame.flat[-1] == 0


class TestCsvToTxt:
    def test_hcpm_contents_match_original(self, synthetic_predictions, tmp_path):
        """
        HCPM file contents must be identical to the original converter's.

        Only the filename differs (``POC_<n>.txt`` vs the original
        ``poc<n>.txt``), so that it matches what the patched encoder opens.
        """
        csv_path, _, _ = synthetic_predictions

        original_dir = tmp_path / "original"
        refactored_dir = tmp_path / "refactored"

        original_files = original_csv_to_txt(csv_path, str(original_dir))
        refactored_files = convert(csv_path, refactored_dir, verbose=False)

        assert set(refactored_files) == set(original_files)

        for poc in sorted(original_files):
            with open(original_files[poc], encoding="utf-8") as handle:
                expected = handle.read().split()
            with open(refactored_files[poc], encoding="utf-8") as handle:
                actual = handle.read().split()
            assert actual == expected, f"HCPM contents differ for POC {poc}"

    def test_filename_matches_encoder_expectation(self, synthetic_predictions, tmp_path):
        """
        Output must be named ``POC_<n>.txt``.

        The patched EncCu.cpp builds this name with
        ``sprintf(phFileName, ".../POC_%d.txt", poc)``.
        """
        csv_path, _, _ = synthetic_predictions
        written = convert(csv_path, tmp_path / "out", verbose=False)

        for poc, path in written.items():
            assert path.name == f"POC_{poc}.txt"


class TestBjontegaard:
    def test_bd_rate_matches_original(self):
        expected = original_bjontegaard(
            np.array(ANCHOR_RATE), np.array(ANCHOR_PSNR),
            np.array(TEST_RATE), np.array(TEST_PSNR), "rate",
        )
        actual = bd_rate(ANCHOR_RATE, ANCHOR_PSNR, TEST_RATE, TEST_PSNR)
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_bd_psnr_matches_original(self):
        expected = original_bjontegaard(
            np.array(ANCHOR_RATE), np.array(ANCHOR_PSNR),
            np.array(TEST_RATE), np.array(TEST_PSNR), "dsnr",
        )
        actual = bd_psnr(ANCHOR_RATE, ANCHOR_PSNR, TEST_RATE, TEST_PSNR)
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_time_saving_matches_original(self):
        expected = np.mean(
            (np.array(ANCHOR_TIME) - np.array(TEST_TIME)) / np.array(ANCHOR_TIME) * 100
        )
        actual = time_saving(ANCHOR_TIME, TEST_TIME)["average"]
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_reported_signs(self):
        """A slower-but-larger bitstream must show positive BD-BR, negative BD-PSNR."""
        assert bd_rate(ANCHOR_RATE, ANCHOR_PSNR, TEST_RATE, TEST_PSNR) > 0
        assert bd_psnr(ANCHOR_RATE, ANCHOR_PSNR, TEST_RATE, TEST_PSNR) < 0
        assert time_saving(ANCHOR_TIME, TEST_TIME)["average"] > 0
