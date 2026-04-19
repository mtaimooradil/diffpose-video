import numpy as np
import pytest


@pytest.fixture
def coco_kps():
    """Random COCO-format (T, 17, 3) keypoints with unit confidence."""
    rng = np.random.default_rng(42)
    kps = rng.random((10, 17, 3)).astype(np.float32)
    kps[..., :2] *= 640
    kps[..., 2] = 1.0
    return kps


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)
