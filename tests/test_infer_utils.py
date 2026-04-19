"""Tests for diffpose_video.common.infer_utils — no GPU required."""

import numpy as np
import pytest
import torch

from diffpose_video.common.infer_utils import (
    RECEPTIVE_FIELD,
    build_windows,
    coco_to_h36m,
    normalise_keypoints,
    stitch_windows,
)


# ---------------------------------------------------------------------------
# coco_to_h36m
# ---------------------------------------------------------------------------

class TestCocoToH36m:
    def test_output_shape_xy(self):
        out = coco_to_h36m(np.zeros((10, 17, 2), dtype=np.float32))
        assert out.shape == (10, 17, 2)

    def test_output_shape_xyc(self, coco_kps):
        out = coco_to_h36m(coco_kps)
        assert out.shape == (10, 17, 3)

    def test_preserves_dtype(self, coco_kps):
        assert coco_to_h36m(coco_kps).dtype == np.float32

    # Direct joint mappings (H36M ← COCO)
    @pytest.mark.parametrize('h36m_idx, coco_idx', [
        (1,  12),  # right_hip
        (2,  14),  # right_knee
        (3,  16),  # right_ankle
        (4,  11),  # left_hip
        (5,  13),  # left_knee
        (6,  15),  # left_ankle
        (9,   0),  # nose
        (11,  5),  # left_shoulder
        (12,  7),  # left_elbow
        (13,  9),  # left_wrist
        (14,  6),  # right_shoulder
        (15,  8),  # right_elbow
        (16, 10),  # right_wrist
    ])
    def test_direct_mapping(self, h36m_idx, coco_idx):
        kps = np.zeros((1, 17, 2), dtype=np.float32)
        kps[0, coco_idx] = [float(coco_idx) * 10, float(coco_idx) * 5]
        out = coco_to_h36m(kps)
        np.testing.assert_array_equal(out[0, h36m_idx], kps[0, coco_idx])

    def test_hip_root_midpoint(self):
        kps = np.zeros((1, 17, 2), dtype=np.float32)
        kps[0, 11] = [0.0, 0.0]    # left_hip
        kps[0, 12] = [100.0, 0.0]  # right_hip
        out = coco_to_h36m(kps)
        assert out[0, 0, 0] == pytest.approx(50.0)

    def test_thorax_midpoint(self):
        kps = np.zeros((1, 17, 2), dtype=np.float32)
        kps[0, 5] = [0.0, 0.0]    # left_shoulder
        kps[0, 6] = [80.0, 0.0]   # right_shoulder
        out = coco_to_h36m(kps)
        assert out[0, 8, 0] == pytest.approx(40.0)

    def test_head_midpoint(self):
        kps = np.zeros((1, 17, 2), dtype=np.float32)
        kps[0, 1] = [10.0, 0.0]   # left_eye
        kps[0, 2] = [30.0, 0.0]   # right_eye
        out = coco_to_h36m(kps)
        assert out[0, 10, 0] == pytest.approx(20.0)

    def test_spine_between_hip_and_thorax(self):
        kps = np.zeros((1, 17, 2), dtype=np.float32)
        kps[0, 11] = kps[0, 12] = [0.0, 0.0]    # hips → root = (0,0)
        kps[0, 5]  = kps[0, 6]  = [0.0, 100.0]  # shoulders → thorax = (0,100)
        out = coco_to_h36m(kps)
        assert out[0, 7, 1] == pytest.approx(50.0)  # spine y midpoint


# ---------------------------------------------------------------------------
# normalise_keypoints
# ---------------------------------------------------------------------------

class TestNormaliseKeypoints:
    def test_output_shape(self):
        kps = np.random.rand(10, 17, 2).astype(np.float32) * 640
        out = normalise_keypoints(kps, width=640, height=480)
        assert out.shape == (10, 17, 2)

    def test_centre_x_maps_to_zero(self):
        w, h = 640, 480
        kps = np.full((1, 17, 2), [w / 2, h / 2], dtype=np.float32)
        out = normalise_keypoints(kps, width=w, height=h)
        assert out[0, 0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_strips_confidence_channel(self):
        kps = np.random.rand(5, 17, 3).astype(np.float32)
        kps[..., :2] *= 640
        out = normalise_keypoints(kps, width=640, height=480)
        assert out.shape == (5, 17, 2)

    def test_output_bounded(self):
        w, h = 640, 480
        kps = np.zeros((1, 17, 2), dtype=np.float32)
        kps[0, :, 0] = w   # all joints at right edge
        kps[0, :, 1] = h   # all joints at bottom edge
        out = normalise_keypoints(kps, width=w, height=h)
        assert out.min() >= -2.0 and out.max() <= 2.0


# ---------------------------------------------------------------------------
# build_windows
# ---------------------------------------------------------------------------

class TestBuildWindows:
    def test_short_sequence_single_window(self):
        kps = np.random.rand(100, 17, 2).astype(np.float32)
        windows = build_windows(kps)
        assert windows.shape == (1, RECEPTIVE_FIELD, 17, 2)

    def test_exact_one_window(self):
        kps = np.random.rand(RECEPTIVE_FIELD, 17, 2).astype(np.float32)
        assert build_windows(kps).shape == (1, RECEPTIVE_FIELD, 17, 2)

    def test_exact_two_windows(self):
        kps = np.random.rand(RECEPTIVE_FIELD * 2, 17, 2).astype(np.float32)
        assert build_windows(kps).shape == (2, RECEPTIVE_FIELD, 17, 2)

    def test_remainder_adds_overlap_window(self):
        kps = np.random.rand(RECEPTIVE_FIELD + 50, 17, 2).astype(np.float32)
        assert build_windows(kps).shape == (2, RECEPTIVE_FIELD, 17, 2)

    def test_returns_tensor(self):
        kps = np.random.rand(50, 17, 2).astype(np.float32)
        assert isinstance(build_windows(kps), torch.Tensor)

    def test_short_padding_replicates_last_frame(self):
        T = 5
        kps = np.zeros((T, 17, 2), dtype=np.float32)
        kps[-1] = 99.0
        windows = build_windows(kps)
        np.testing.assert_array_equal(windows[0, T:], 99.0)

    def test_full_windows_match_input(self):
        T = RECEPTIVE_FIELD * 2
        kps = np.random.rand(T, 17, 2).astype(np.float32)
        windows = build_windows(kps).numpy()
        np.testing.assert_array_equal(windows[0], kps[:RECEPTIVE_FIELD])
        np.testing.assert_array_equal(windows[1], kps[RECEPTIVE_FIELD:])


# ---------------------------------------------------------------------------
# stitch_windows
# ---------------------------------------------------------------------------

class TestStitchWindows:
    def test_single_window_shape(self):
        preds = np.random.rand(1, RECEPTIVE_FIELD, 17, 3).astype(np.float32)
        out = stitch_windows(preds, RECEPTIVE_FIELD)
        assert out.shape == (RECEPTIVE_FIELD, 17, 3)

    def test_single_window_values(self):
        preds = np.random.rand(1, RECEPTIVE_FIELD, 17, 3).astype(np.float32)
        out = stitch_windows(preds, RECEPTIVE_FIELD)
        np.testing.assert_array_equal(out, preds[0])

    def test_two_windows_shape(self):
        T = RECEPTIVE_FIELD * 2
        preds = np.random.rand(2, RECEPTIVE_FIELD, 17, 3).astype(np.float32)
        assert stitch_windows(preds, T).shape == (T, 17, 3)

    def test_partial_last_window(self):
        remainder = 50
        T = RECEPTIVE_FIELD + remainder
        preds = np.random.rand(2, RECEPTIVE_FIELD, 17, 3).astype(np.float32)
        out = stitch_windows(preds, T)
        assert out.shape == (T, 17, 3)
        np.testing.assert_array_equal(out[RECEPTIVE_FIELD:], preds[-1, -remainder:])

    def test_roundtrip_exact_multiple(self):
        T = RECEPTIVE_FIELD * 3
        kps = np.random.rand(T, 17, 2).astype(np.float32)
        windows = build_windows(kps).numpy()
        preds = np.concatenate([windows, np.zeros((3, RECEPTIVE_FIELD, 17, 1))], axis=-1)
        out = stitch_windows(preds, T)
        np.testing.assert_allclose(out[..., :2], kps, atol=1e-6)

    def test_single_frame(self):
        preds = np.random.rand(1, RECEPTIVE_FIELD, 17, 3).astype(np.float32)
        out = stitch_windows(preds, 1)
        assert out.shape == (1, 17, 3)
        # remainder=1 → takes last frame of last window
        np.testing.assert_array_equal(out[0], preds[0, -1])


class TestBuildWindowsBoundary:
    def test_receptive_field_minus_one(self):
        T = RECEPTIVE_FIELD - 1
        kps = np.random.rand(T, 17, 2).astype(np.float32)
        windows = build_windows(kps)
        assert windows.shape == (1, RECEPTIVE_FIELD, 17, 2)
        # Original frames preserved
        np.testing.assert_array_equal(windows[0, :T], kps)
        # Padded frames replicate last (only 1 padded frame)
        np.testing.assert_array_equal(windows[0, T], kps[-1])
