"""Tests for diffpose_video.scripts.visualise — pure helpers, no video I/O."""

import numpy as np
import pytest

from pathlib import Path

from diffpose_video.scripts.visualise import draw_2d_skeleton, find_video


class TestFindVideo:
    def test_finds_direct(self, tmp_path):
        f = tmp_path / 'G07.mp4'
        f.touch()
        assert find_video('G07', str(tmp_path)) == str(f)

    def test_finds_nested(self, tmp_path):
        sub = tmp_path / 'G07'
        sub.mkdir()
        f = sub / 'G07.mp4'
        f.touch()
        assert find_video('G07', str(tmp_path)) == str(f)

    def test_prefers_non_sync(self, tmp_path):
        sub = tmp_path / 'G07'
        sub.mkdir()
        (sub / 'G07.mp4').touch()
        (sub / 'G07_sync.mp4').touch()
        result = find_video('G07', str(tmp_path))
        assert '_sync' not in Path(result).name

    def test_returns_none_when_missing(self, tmp_path):
        assert find_video('ghost', str(tmp_path)) is None

    @pytest.mark.parametrize('ext', ['.mp4', '.mov', '.avi', '.mkv'])
    def test_finds_various_extensions(self, tmp_path, ext):
        f = tmp_path / f'clip{ext}'
        f.touch()
        result = find_video('clip', str(tmp_path))
        assert result == str(f)

    def test_multiple_matches_returns_one(self, tmp_path):
        (tmp_path / 'clip.mp4').touch()
        (tmp_path / 'clip.avi').touch()
        result = find_video('clip', str(tmp_path))
        assert result is not None


class TestDraw2DSkeleton:
    def test_output_shape(self, blank_frame):
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 2] = 1.0
        out = draw_2d_skeleton(blank_frame, kps)
        assert out.shape == blank_frame.shape

    def test_returns_copy_not_inplace(self, blank_frame):
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 2] = 1.0
        original = blank_frame.copy()
        draw_2d_skeleton(blank_frame, kps)
        np.testing.assert_array_equal(blank_frame, original)

    def test_low_confidence_no_drawing(self, blank_frame):
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 2] = 0.0  # below threshold → nothing drawn
        out = draw_2d_skeleton(blank_frame, kps)
        np.testing.assert_array_equal(out, blank_frame)

    def test_high_confidence_draws_something(self, blank_frame):
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 0] = 320.0  # x centre
        kps[:, 1] = 240.0  # y centre
        kps[:, 2] = 1.0
        out = draw_2d_skeleton(blank_frame, kps)
        assert not np.array_equal(out, blank_frame)

    def test_output_dtype_preserved(self, blank_frame):
        kps = np.zeros((17, 3), dtype=np.float32)
        out = draw_2d_skeleton(blank_frame, kps)
        assert out.dtype == np.uint8

    def test_custom_confidence_threshold(self, blank_frame):
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 0] = 320.0
        kps[:, 1] = 240.0
        kps[:, 2] = 0.5
        # With default thr=0.3 → draws; with thr=0.6 → skips
        out_draws = draw_2d_skeleton(blank_frame, kps, conf_thr=0.3)
        out_skips = draw_2d_skeleton(blank_frame, kps, conf_thr=0.6)
        assert not np.array_equal(out_draws, blank_frame)
        np.testing.assert_array_equal(out_skips, blank_frame)

    def test_out_of_bounds_coords_no_crash(self, blank_frame):
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 0] = 99999.0   # way outside frame
        kps[:, 1] = 99999.0
        kps[:, 2] = 1.0
        out = draw_2d_skeleton(blank_frame, kps)
        assert out.shape == blank_frame.shape
