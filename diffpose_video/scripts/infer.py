"""
Offline inference pipeline for DiffPose-Video.

Given one or more video files or directories, this script:
  1. Detects 2D keypoints per frame with RTMPose (via rtmlib).
  2. Remaps COCO 17 joints → H36M 17 joints.
  3. Normalises to screen coordinates.
  4. Runs MixSTE + DiffPose to lift 2D → 3D.
  5. Saves the 3D keypoints as <output_dir>/.../<video_stem>.npz,
     preserving subdirectory structure for recursive inputs.

Usage
-----
Single video:
    diffpose-infer --input video.mp4

Multiple inputs (files and/or directories):
    diffpose-infer --input clip1.mp4 clip2.mp4 /path/to/folder/

Recursive directory (preserves subdir structure in output):
    diffpose-infer --input /datasets/videos/ --recursive --output_dir results/

Skip already-processed files:
    diffpose-infer --input /datasets/videos/ --recursive --skip_existing
"""

import argparse
import fnmatch
import os
import sys
import time
import types

# Force onnxruntime to use CUDA provider before any rtmlib/onnxruntime import
os.environ.setdefault('ONNXRUNTIME_EXECUTION_PROVIDERS', 'CUDAExecutionProvider,CPUExecutionProvider')

import cv2
import numpy as np
import torch
import yaml
from rtmlib import Body, PoseTracker
from tqdm import tqdm

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
    parser.add_argument('--config',     default=None,
                        help='Path to a TOML config file (all other args become optional).')
    parser.add_argument('--input',      nargs='+', default=None,
                        help='One or more video files and/or directories to process.')
    parser.add_argument('--output_dir', default=None,
                        help='Root directory where output .npz files are saved. (default: results/)')
    parser.add_argument('--recursive',  action='store_const', const=True, default=None,
                        help='Recurse into subdirectories when --input is a directory.')
    parser.add_argument('--skip_existing', action='store_const', const=True, default=None,
                        help='Skip videos whose .npz output already exists.')
    parser.add_argument('--exclude', nargs='+', default=None, metavar='PATTERN',
                        help='Exclude files matching these patterns (substring or glob).')
    parser.add_argument('--model_config', default=None,
                        help='Path to the model YAML config file. '
                             '(default: bundled human36m_diffpose_uvxyz_cpn.yml)')
    parser.add_argument('--model_pose', default=None,
                        help='Path to the MixSTE checkpoint (.bin).')
    parser.add_argument('--model_diff', default=None,
                        help='Path to the GCNdiff checkpoint (.pth).')
    parser.add_argument('--det_freq',   type=int, default=None,
                        help='Run person detection every N frames. (default: 1)')
    parser.add_argument('--device',     default=None,
                        help='Torch device: cuda or cpu. (default: cuda)')
    args = parser.parse_args()
    _apply_config(args)
    return args


def _apply_config(args) -> None:
    from diffpose_video.common.config_loader import load_toml, merge

    cfg = load_toml(args.config) if args.config else {}

    merge(args, cfg, defaults={
        'input':        None,
        'output_dir':   'results',
        'recursive':    False,
        'skip_existing': False,
        'exclude':      [],
        'model_config': None,
        'model_pose':   None,
        'model_diff':   None,
        'det_freq':     1,
        'device':       'cuda',
    })

    if not args.input:
        raise SystemExit('Provide --input or set input in your config file.')
    if args.model_config is None:
        args.model_config = _default_config()
    if args.model_pose is None:
        args.model_pose = _default_checkpoint('mixste_cpn_243f.bin')
    if args.model_diff is None:
        args.model_diff = _default_checkpoint('diffpose_video_uvxyz_cpn.pth')


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

VIDEO_EXTENSIONS = {
    '.mp4', '.m4v', '.mov', '.qt',           # MPEG-4 / QuickTime
    '.avi',                                   # AVI
    '.mkv', '.webm',                          # Matroska / WebM
    '.wmv', '.asf',                           # Windows Media
    '.flv', '.f4v',                           # Flash
    '.mpeg', '.mpg', '.m2v', '.ts', '.mts', '.m2ts',  # MPEG-2
    '.3gp', '.3g2',                           # Mobile
    '.ogv', '.ogg',                           # Ogg
    '.vob', '.divx', '.rmvb', '.rm',          # Other
    '.dav', '.mxf',                           # Surveillance / broadcast
    '.hevc',                                  # Raw HEVC/H.265
    '.movi',                                  # MOVI container
}


def _is_video(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTENSIONS


def _is_excluded(filename: str, patterns: list[str]) -> bool:
    """
    Return True if filename matches any exclusion pattern.
    Patterns are matched case-insensitively against the bare filename.
    A pattern without wildcards is treated as a substring match.
    A pattern with '*' or '?' is treated as a glob match.
    """
    name = filename.lower()
    for pat in patterns:
        pat_lower = pat.lower()
        if '*' in pat_lower or '?' in pat_lower:
            if fnmatch.fnmatch(name, pat_lower):
                return True
        else:
            if pat_lower in name:
                return True
    return False


def collect_videos(
    inputs: list[str],
    recursive: bool = False,
    exclude: list[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Collect video files from a mixed list of files and directories.

    Returns a list of (video_path, relative_subdir) tuples.
    - video_path:      absolute path to the video file.
    - relative_subdir: path relative to the input root, used to mirror
                       the source layout under output_dir. Empty string for
                       flat inputs (single files or top-level folder scan).
    """
    exclude = exclude or []
    results: list[tuple[str, str]] = []

    for inp in inputs:
        inp = os.path.abspath(inp)
        if os.path.isfile(inp):
            fname = os.path.basename(inp)
            if not _is_video(fname):
                print(f'[warn] Skipping non-video file: {inp}')
            elif _is_excluded(fname, exclude):
                print(f'[skip] Excluded by pattern: {inp}')
            else:
                results.append((inp, ''))
        elif os.path.isdir(inp):
            if recursive:
                for dirpath, dirnames, filenames in os.walk(inp):
                    dirnames.sort()
                    rel = os.path.relpath(dirpath, inp)
                    rel = '' if rel == '.' else rel
                    for fname in sorted(filenames):
                        if not _is_video(fname):
                            continue
                        if _is_excluded(fname, exclude):
                            print(f'[skip] Excluded by pattern: {os.path.join(dirpath, fname)}')
                            continue
                        results.append((os.path.join(dirpath, fname), rel))
            else:
                for fname in sorted(os.listdir(inp)):
                    fpath = os.path.join(inp, fname)
                    if not os.path.isfile(fpath) or not _is_video(fname):
                        continue
                    if _is_excluded(fname, exclude):
                        print(f'[skip] Excluded by pattern: {fpath}')
                        continue
                    results.append((fpath, ''))
        else:
            print(f'[warn] Input not found, skipping: {inp}')

    if not results:
        sys.exit('No video files found in the specified input(s).')
    return results


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
    out_path: str,
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
    config = load_config(args.model_config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('Loading models ...')
    model_pose, model_diff, betas = load_models(
        config, args.model_pose, args.model_diff, device
    )

    print('Initialising RTMPose detector ...')
    det_device = 'cuda' if device.type == 'cuda' else 'cpu'
    tracker = build_detector(det_device, det_freq=args.det_freq)

    videos = collect_videos(args.input, recursive=args.recursive, exclude=args.exclude)
    total = len(videos)
    print(f'Found {total} video(s) to process.')

    skipped = 0
    done = 0
    bar = tqdm(
        videos,
        total=total,
        unit='video',
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        dynamic_ncols=True,
    )

    for video_path, subdir in bar:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_subdir = os.path.join(args.output_dir, subdir)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, f'{stem}.npz')

        bar.set_postfix_str(f'current={stem}  done={done}  skipped={skipped}  left={total - done - skipped - 1}')

        if args.skip_existing and os.path.exists(out_path):
            tqdm.write(f'[skip] {stem} — already exists')
            skipped += 1
            continue

        t0 = time.perf_counter()
        process_video(
            video_path, out_path,
            tracker, model_pose, model_diff, betas,
            config, device, args.det_freq,
        )
        elapsed = time.perf_counter() - t0
        done += 1
        tqdm.write(f'[done] {stem}  ({elapsed:.1f}s)  → {out_path}')
        bar.set_postfix_str(f'current={stem}  done={done}  skipped={skipped}  left={total - done - skipped}')

    bar.close()
    print(f'\nFinished — {done} processed, {skipped} skipped, {total - done - skipped} failed.')


if __name__ == '__main__':
    main()
