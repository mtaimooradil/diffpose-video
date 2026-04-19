"""
Offline inference pipeline for DiffPose-Video.

Given a video file (or a folder of videos), this script:
  1. Detects 2D keypoints per frame with RTMPose (via rtmlib).
  2. Remaps COCO 17 joints → H36M 17 joints.
  3. Normalises to screen coordinates.
  4. Runs MixSTE + DiffPose to lift 2D → 3D.
  5. Saves the 3D keypoints as <output_dir>/<video_stem>.npz.

Usage
-----
Single video:
    python infer.py --input video.mp4 \\
                    --model_pose checkpoints/mixste_cpn_243f.bin \\
                    --model_diff checkpoints/diffpose_video_uvxyz_cpn.pth \\
                    --config configs/human36m_diffpose_uvxyz_cpn.yml

Folder of videos:
    python infer.py --input /path/to/videos/ \\
                    --model_pose checkpoints/mixste_cpn_243f.bin \\
                    --model_diff checkpoints/diffpose_video_uvxyz_cpn.pth \\
                    --config configs/human36m_diffpose_uvxyz_cpn.yml \\
                    --output_dir results/
"""

import argparse
import os
import sys
import types

import cv2
import numpy as np
import torch
import yaml
from rtmlib import Body, PoseTracker

from diffpose_video.common.infer_utils import (
    JOINTS_LEFT,
    JOINTS_RIGHT,
    build_windows,
    coco_to_h36m,
    load_models,
    normalise_keypoints,
    stitch_windows,
)
from diffpose_video.common.utils_diff import generalized_steps


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _default_config() -> str:
    from diffpose_video import configs_dir
    return str(configs_dir() / 'human36m_diffpose_uvxyz_cpn.yml')


def _default_checkpoint(name: str) -> str:
    from diffpose_video.download_checkpoints import DEFAULT_DIR
    return str(DEFAULT_DIR / name)


def parse_args():
    parser = argparse.ArgumentParser(description='DiffPose-Video inference on arbitrary videos')
    parser.add_argument('--input',      required=True,
                        help='Path to a video file or a folder containing video files.')
    parser.add_argument('--output_dir', default='results',
                        help='Directory where output .npz files are saved. (default: results/)')
    parser.add_argument('--config',     default=None,
                        help='Path to the YAML config file. '
                             '(default: bundled human36m_diffpose_uvxyz_cpn.yml)')
    parser.add_argument('--model_pose', default=None,
                        help='Path to the MixSTE checkpoint (.bin). '
                             '(default: ~/.cache/diffpose_video/checkpoints/mixste_cpn_243f.bin)')
    parser.add_argument('--model_diff', default=None,
                        help='Path to the GCNdiff checkpoint (.pth). '
                             '(default: ~/.cache/diffpose_video/checkpoints/diffpose_video_uvxyz_cpn.pth)')
    parser.add_argument('--det_freq',   type=int, default=1,
                        help='Run person detection every N frames to speed up inference. (default: 1 = every frame)')
    parser.add_argument('--device',     default='cuda',
                        help='Torch device: cuda or cpu. (default: cuda)')
    args = parser.parse_args()
    if args.config is None:
        args.config = _default_config()
    if args.model_pose is None:
        args.model_pose = _default_checkpoint('mixste_cpn_243f.bin')
    if args.model_diff is None:
        args.model_diff = _default_checkpoint('diffpose_video_uvxyz_cpn.pth')
    return args


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str):
    """Load YAML config and convert nested dicts to SimpleNamespace objects."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    def to_ns(d):
        if isinstance(d, dict):
            return types.SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
        return d

    return to_ns(raw)


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def collect_videos(input_path: str) -> list[str]:
    """Return a sorted list of video file paths from a file or folder."""
    if os.path.isfile(input_path):
        return [input_path]
    videos = [
        os.path.join(input_path, f)
        for f in sorted(os.listdir(input_path))
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        sys.exit(f'No video files found in {input_path}')
    return videos


def extract_frames(video_path: str) -> tuple[list[np.ndarray], int, int]:
    """
    Read all frames from a video file.

    Returns:
        frames: list of BGR numpy arrays.
        width:  frame width in pixels.
        height: frame height in pixels.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f'Cannot open video: {video_path}')

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, width, height


# ---------------------------------------------------------------------------
# 2D detection
# ---------------------------------------------------------------------------

def build_detector(device: str, det_freq: int = 1):
    """
    Instantiate an RTMPose body tracker (person detection + 2D pose estimation).

    PoseTracker wraps a Body solution (RTMDet detector + RTMPose-m estimator)
    and adds cross-frame person tracking so the dominant person stays consistent.
    We use 'performance' mode for best accuracy.
    """
    return PoseTracker(
        solution=Body,
        det_frequency=det_freq,
        mode='performance',
        backend='onnxruntime',
        device=device,
    )


def detect_keypoints(
    frames: list[np.ndarray],
    tracker: PoseTracker,
    det_freq: int,
) -> np.ndarray:
    """
    Run 2D pose detection on every frame and return the dominant person's keypoints.

    The dominant person is the one whose bounding box has the largest area;
    this heuristic works well for single-person videos and selects the main
    subject in crowded scenes.

    Args:
        frames:   list of BGR frames.
        tracker:  RTMPose PoseTracker instance.
        det_freq: run detection every `det_freq` frames (tracking fills the gaps).

    Returns:
        keypoints: float32 array of shape (T, 17, 3) — (x, y, confidence)
                   in COCO joint order, pixel coordinates.
    """
    T = len(frames)
    keypoints_all = np.zeros((T, 17, 3), dtype=np.float32)

    for i, frame in enumerate(frames):
        # rtmlib returns (keypoints, scores) arrays, one row per detected person
        kps, scores = tracker(frame)

        if kps is None or len(kps) == 0:
            # No detection: carry forward the previous frame's keypoints
            if i > 0:
                keypoints_all[i] = keypoints_all[i - 1]
            continue

        # Pick the person with the highest mean confidence (proxy for largest/closest)
        mean_scores = scores.mean(axis=1)          # (n_persons,)
        best = int(np.argmax(mean_scores))

        keypoints_all[i, :, :2] = kps[best]       # (17, 2) pixel coords
        keypoints_all[i, :, 2]  = scores[best]    # (17,)   confidence

    return keypoints_all


# ---------------------------------------------------------------------------
# 3D lifting
# ---------------------------------------------------------------------------

SRC_MASK = torch.tensor(
    [[[True] * 17]], dtype=torch.bool
)


def lift_to_3d(
    keypoints_2d: np.ndarray,
    model_pose,
    model_diff,
    betas: torch.Tensor,
    config,
    device: torch.device,
) -> np.ndarray:
    """
    Lift a sequence of normalised 2D keypoints to 3D using MixSTE + DiffPose.

    Args:
        keypoints_2d: (T, 17, 2) normalised keypoints in H36M joint order.
        model_pose:   MixSTE2 model.
        model_diff:   GCNdiff model.
        betas:        diffusion schedule.
        config:       config namespace (provides diffusion testing params).
        device:       torch device.

    Returns:
        poses_3d: (T, 17, 3) root-relative 3D joint positions.
    """
    n_frames = keypoints_2d.shape[0]
    src_mask  = SRC_MASK.to(device)

    # Build 243-frame windows; shape: (n_windows, 243, 17, 2)
    windows = build_windows(keypoints_2d).to(device)

    # Flip-augmented input (mirror x, swap left/right joints)
    windows_flip = windows.clone()
    windows_flip[:, :, :, 0] *= -1
    windows_flip[:, :, JOINTS_LEFT + JOINTS_RIGHT, :] = \
        windows_flip[:, :, JOINTS_RIGHT + JOINTS_LEFT, :]

    # Diffusion sampling parameters
    test_timesteps               = config.testing.test_timesteps
    test_num_diffusion_timesteps = config.testing.test_num_diffusion_timesteps
    skip = test_num_diffusion_timesteps // test_timesteps
    seq  = range(0, test_num_diffusion_timesteps, skip)

    all_preds = []

    with torch.no_grad():
        for win, win_flip in zip(windows, windows_flip):
            # Add batch dimension: (1, 243, 17, 2)
            inp      = win.unsqueeze(0)
            inp_flip = win_flip.unsqueeze(0)

            # --- Stage 1: MixSTE initial 3D estimate ---
            pred_3d      = model_pose(inp)        # (1, 243, 17, 3)
            pred_3d_flip = model_pose(inp_flip)

            # Un-flip and average
            pred_3d_flip[:, :, :, 0] *= -1
            pred_3d_flip[:, :, JOINTS_LEFT + JOINTS_RIGHT, :] = \
                pred_3d_flip[:, :, JOINTS_RIGHT + JOINTS_LEFT, :]
            pred_3d = (pred_3d + pred_3d_flip) / 2   # (1, 243, 17, 3)

            # --- Stage 2: DiffPose reverse diffusion ---
            # Concatenate 2D uv + 3D xyz → (1*243, 17, 5)
            B, F, J, _ = pred_3d.shape
            inp_xyz = pred_3d.reshape(B * F, J, 3)
            inp_uv  = inp.reshape(B * F, J, 2)
            input_uvxyz = torch.cat([inp_uv, inp_xyz], dim=2)  # (243, 17, 5)

            # Noise scale: small for uv (already detected), larger for xyz
            noise_scale = torch.tensor([0.01, 0.01, 1.0, 1.0, 1.0],
                                       device=device)
            noise_scale = noise_scale.repeat(B * F, J, 1)

            output_uvxyz = generalized_steps(
                input_uvxyz, src_mask, seq, model_diff, betas, eta=0.0
            )
            output_uvxyz = output_uvxyz[0][-1]   # final denoised sample

            # Extract xyz and make root-relative
            output_xyz = output_uvxyz[:, :, 2:]
            output_xyz -= output_xyz[:, :1, :]   # subtract root (joint 0)

            # Reshape back to (243, 17, 3)
            all_preds.append(output_xyz.cpu().numpy())

    # Stack windows and stitch back to the original length
    all_preds = np.stack(all_preds, axis=0)   # (n_windows, 243, 17, 3)
    poses_3d  = stitch_windows(all_preds, n_frames)  # (T, 17, 3)
    return poses_3d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    output_dir: str,
    tracker: PoseTracker,
    model_pose,
    model_diff,
    betas: torch.Tensor,
    config,
    device: torch.device,
    det_freq: int,
):
    """Run the full pipeline on a single video and save the result."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f'{stem}.npz')

    print(f'\n[{stem}] Loading frames ...')
    frames, width, height = extract_frames(video_path)
    print(f'[{stem}] {len(frames)} frames  ({width}×{height})')

    print(f'[{stem}] Detecting 2D keypoints ...')
    kps_coco = detect_keypoints(frames, tracker, det_freq)    # (T, 17, 3)

    print(f'[{stem}] Remapping COCO → H36M ...')
    kps_h36m = coco_to_h36m(kps_coco)                        # (T, 17, 3)

    print(f'[{stem}] Normalising keypoints ...')
    kps_norm = normalise_keypoints(kps_h36m, width, height)   # (T, 17, 2)

    print(f'[{stem}] Lifting 2D → 3D ...')
    poses_3d = lift_to_3d(kps_norm, model_pose, model_diff, betas, config, device)

    np.savez_compressed(
        out_path,
        poses_3d=poses_3d,          # (T, 17, 3)  root-relative, in training scale
        keypoints_2d=kps_h36m,      # (T, 17, 3)  pixel coords + confidence
        width=width,
        height=height,
    )
    print(f'[{stem}] Saved → {out_path}')


def _check_checkpoint(path: str, name: str) -> None:
    if not os.path.isfile(path):
        from diffpose_video.download_checkpoints import DEFAULT_DIR
        sys.exit(
            f'\nCheckpoint not found: {path}\n'
            f'Run the following to download pretrained weights:\n'
            f'  python download_checkpoints.py\n'
            f'Then pass --model_pose / --model_diff pointing to {DEFAULT_DIR}/'
        )


def main():
    args   = parse_args()
    _check_checkpoint(args.model_pose, 'MixSTE')
    _check_checkpoint(args.model_diff, 'GCNdiff')
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading models ...')
    model_pose, model_diff, betas = load_models(
        config, args.model_pose, args.model_diff, device
    )

    print('Initialising RTMPose detector ...')
    det_device = 'cuda' if device.type == 'cuda' else 'cpu'
    tracker = build_detector(det_device, det_freq=args.det_freq)

    videos = collect_videos(args.input)
    print(f'Found {len(videos)} video(s) to process.')

    for video_path in videos:
        process_video(
            video_path, args.output_dir,
            tracker, model_pose, model_diff, betas,
            config, device, args.det_freq,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
