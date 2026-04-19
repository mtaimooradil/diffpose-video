"""
Utility functions for the inference pipeline.

Covers:
  - COCO-to-H36M joint remapping
  - Screen-coordinate normalisation
  - Temporal windowing (chunk a sequence into 243-frame segments)
  - Window stitching (reassemble chunks back to the original length)
  - Model loading helpers
"""

import numpy as np
import torch
from torch.nn import functional as F

from diffpose_video.common.camera import normalize_screen_coordinates
from diffpose_video.models.mixste import MixSTE2
from diffpose_video.models.gcndiff import GCNdiff, adj_mx_from_edges
from diffpose_video.common.utils_diff import get_beta_schedule


# ---------------------------------------------------------------------------
# Joint mapping
# ---------------------------------------------------------------------------

# COCO 17-joint indices
# 0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
# 5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
# 9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
# 13:left_knee  14:right_knee  15:left_ankle  16:right_ankle

# H36M 17-joint indices (after remove_joints reduction from 32→17)
# 0:hip(root)  1:right_hip  2:right_knee  3:right_ankle
# 4:left_hip   5:left_knee  6:left_ankle
# 7:spine      8:thorax     9:neck/nose  10:head
# 11:left_shoulder  12:left_elbow  13:left_wrist
# 14:right_shoulder 15:right_elbow 16:right_wrist

def coco_to_h36m(keypoints: np.ndarray) -> np.ndarray:
    """
    Remap COCO 17-joint keypoints to H36M 17-joint format.

    Args:
        keypoints: float array of shape (T, 17, 2) or (T, 17, 3) in pixel coords.
                   Channel order is (x, y) or (x, y, confidence).

    Returns:
        h36m_kps: float array of shape (T, 17, C) in H36M joint order.
                  The confidence channel (if present) is propagated via averaging
                  for synthetic joints.
    """
    T, _, C = keypoints.shape
    h36m_kps = np.zeros((T, 17, C), dtype=keypoints.dtype)

    # Direct 1-to-1 copies (H36M ← COCO)
    h36m_kps[:, 1, :]  = keypoints[:, 12, :]   # right_hip
    h36m_kps[:, 2, :]  = keypoints[:, 14, :]   # right_knee
    h36m_kps[:, 3, :]  = keypoints[:, 16, :]   # right_ankle
    h36m_kps[:, 4, :]  = keypoints[:, 11, :]   # left_hip
    h36m_kps[:, 5, :]  = keypoints[:, 13, :]   # left_knee
    h36m_kps[:, 6, :]  = keypoints[:, 15, :]   # left_ankle
    h36m_kps[:, 9, :]  = keypoints[:, 0,  :]   # nose → neck/nose
    h36m_kps[:, 11, :] = keypoints[:, 5,  :]   # left_shoulder
    h36m_kps[:, 12, :] = keypoints[:, 7,  :]   # left_elbow
    h36m_kps[:, 13, :] = keypoints[:, 9,  :]   # left_wrist
    h36m_kps[:, 14, :] = keypoints[:, 6,  :]   # right_shoulder
    h36m_kps[:, 15, :] = keypoints[:, 8,  :]   # right_elbow
    h36m_kps[:, 16, :] = keypoints[:, 10, :]   # right_wrist

    # Synthetic joints computed as midpoints
    h36m_kps[:, 0,  :] = (keypoints[:, 11, :] + keypoints[:, 12, :]) / 2  # hip root
    h36m_kps[:, 7,  :] = (h36m_kps[:, 0, :] + h36m_kps[:, 8, :]) / 2     # spine (approx)
    h36m_kps[:, 8,  :] = (keypoints[:, 5,  :] + keypoints[:, 6,  :]) / 2  # thorax (shoulder mid)
    h36m_kps[:, 10, :] = (keypoints[:, 1,  :] + keypoints[:, 2,  :]) / 2  # head (eye mid)

    # Recompute spine now that thorax is available
    h36m_kps[:, 7, :] = (h36m_kps[:, 0, :] + h36m_kps[:, 8, :]) / 2

    return h36m_kps


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_keypoints(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Normalise pixel-space 2D keypoints to [-1, 1] using the same convention
    as the training data (see common/camera.py::normalize_screen_coordinates).

    Args:
        keypoints: (T, 17, 2) array of (x, y) pixel coordinates.
        width:     frame width in pixels.
        height:    frame height in pixels.

    Returns:
        Normalised keypoints of the same shape.
    """
    return normalize_screen_coordinates(keypoints[..., :2], w=width, h=height)


# ---------------------------------------------------------------------------
# Temporal windowing
# ---------------------------------------------------------------------------

RECEPTIVE_FIELD = 243  # frames expected by MixSTE


def build_windows(keypoints_2d: np.ndarray) -> torch.Tensor:
    """
    Split a (T, 17, 2) sequence into overlapping 243-frame chunks.

    The last chunk always ends at frame T-1; if T is not a multiple of 243
    the last chunk reuses the final frames (matching eval_data_prepare logic).

    Args:
        keypoints_2d: normalised keypoints of shape (T, 17, 2).

    Returns:
        windows: float tensor of shape (n_windows, 243, 17, 2).
    """
    T = keypoints_2d.shape[0]
    kps = torch.from_numpy(keypoints_2d.astype(np.float32))  # (T, 17, 2)

    if T < RECEPTIVE_FIELD:
        # Pad the sequence by replicating the last frame on the right
        pad = RECEPTIVE_FIELD - T
        kps = kps.permute(1, 2, 0)               # (17, 2, T)
        kps = F.pad(kps, (0, pad), mode='replicate')
        kps = kps.permute(2, 0, 1)               # (243, 17, 2)
        return kps.unsqueeze(0)                   # (1, 243, 17, 2)

    n_full = T // RECEPTIVE_FIELD
    remainder = T % RECEPTIVE_FIELD
    n_windows = n_full + (1 if remainder > 0 else 0)

    windows = torch.empty(n_windows, RECEPTIVE_FIELD, 17, 2)
    for i in range(n_full):
        start = i * RECEPTIVE_FIELD
        windows[i] = kps[start : start + RECEPTIVE_FIELD]

    if remainder > 0:
        # Last window: take the final 243 frames (overlaps with previous window)
        windows[-1] = kps[-RECEPTIVE_FIELD:]

    return windows  # (n_windows, 243, 17, 2)


def stitch_windows(predictions: np.ndarray, n_frames: int) -> np.ndarray:
    """
    Reassemble per-window 3D predictions back to the original sequence length.

    For non-overlapping windows we copy directly; the last (potentially
    overlapping) window contributes only the frames not already covered.

    Args:
        predictions: (n_windows, 243, 17, 3) numpy array.
        n_frames:    original number of frames T.

    Returns:
        output: (T, 17, 3) numpy array.
    """
    output = np.zeros((n_frames, 17, 3), dtype=np.float32)

    n_full = n_frames // RECEPTIVE_FIELD
    remainder = n_frames % RECEPTIVE_FIELD

    for i in range(n_full):
        start = i * RECEPTIVE_FIELD
        output[start : start + RECEPTIVE_FIELD] = predictions[i]

    if remainder > 0:
        # Last window overlaps; only take the tail that hasn't been written yet
        output[n_full * RECEPTIVE_FIELD :] = predictions[-1, -remainder:]

    return output


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

H36M_EDGES = torch.tensor(
    [[0, 1], [1, 2], [2, 3],
     [0, 4], [4, 5], [5, 6],
     [0, 7], [7, 8], [8, 9], [9, 10],
     [8, 11], [11, 12], [12, 13],
     [8, 14], [14, 15], [15, 16]],
    dtype=torch.long,
)

# Left/right joint indices used for flip augmentation (H36M convention)
JOINTS_LEFT  = [4, 5, 6, 11, 12, 13]
JOINTS_RIGHT = [1, 2, 3, 14, 15, 16]


def load_models(config, model_pose_path: str, model_diff_path: str, device: torch.device):
    """
    Instantiate MixSTE (pose backbone) and GCNdiff (diffusion model),
    then load their pretrained weights.

    Args:
        config:          parsed YAML config namespace.
        model_pose_path: path to MixSTE checkpoint (.bin).
        model_diff_path: path to GCNdiff checkpoint (.pth).
        device:          torch device.

    Returns:
        model_pose: MixSTE2 in eval mode.
        model_diff: GCNdiff in eval mode.
        betas:      diffusion schedule tensor on device.
    """
    # --- MixSTE (2D → initial 3D) ---
    model_pose = MixSTE2(
        num_frame=RECEPTIVE_FIELD,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=512,
        depth=8,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0,
    )
    model_pose = torch.nn.DataParallel(model_pose).to(device)
    states = torch.load(model_pose_path, map_location=device, weights_only=False)
    model_pose.load_state_dict(states['model_pos'])
    model_pose.eval()

    # --- GCNdiff (diffusion refinement) ---
    adj = adj_mx_from_edges(num_pts=17, edges=H36M_EDGES, sparse=False)
    model_diff = GCNdiff(adj.to(device), config).to(device)
    model_diff = torch.nn.DataParallel(model_diff)
    states = torch.load(model_diff_path, map_location=device, weights_only=False)
    model_diff.load_state_dict(states[0])
    model_diff.eval()

    # --- Diffusion schedule ---
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)

    return model_pose, model_diff, betas
