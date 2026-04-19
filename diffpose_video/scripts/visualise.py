"""
Visualisation script for DiffPose-Video 3D keypoint output.

Produces a side-by-side MP4 with:
  - Left panel:  original video frames with 2D skeleton overlay
  - Right panel: animated 3D skeleton (matplotlib)

Usage
-----
    python visualise.py \\
        --npz    results/IMG_0076.npz \\
        --video  /path/to/IMG_0076.MOV \\
        --output results/IMG_0076_vis.mp4

Optional flags:
    --fps   N        output frame rate (default: match source video)
    --start N        first frame to render (default: 0)
    --end   N        last frame to render  (default: all)
    --azim  DEG      initial 3-D camera azimuth (default: 70)
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# H36M 17-joint skeleton definition
# ---------------------------------------------------------------------------

# Bone connectivity: each tuple is (parent_joint, child_joint)
BONES = [
    (0, 1), (1, 2), (2, 3),        # right leg
    (0, 4), (4, 5), (5, 6),        # left leg
    (0, 7), (7, 8),                 # spine
    (8, 9), (9, 10),                # neck → nose → head
    (8, 11), (11, 12), (12, 13),   # left arm
    (8, 14), (14, 15), (15, 16),   # right arm
]

# Colours: right=blue, left=red, centre=green  (BGR for cv2, RGB for matplotlib)
_R = (0.2, 0.4, 0.8)   # right limbs – blue
_L = (0.8, 0.2, 0.2)   # left limbs  – red
_C = (0.2, 0.7, 0.3)   # centre      – green

BONE_COLORS = [
    _R, _R, _R,    # right leg
    _L, _L, _L,    # left leg
    _C, _C,        # spine
    _C, _C,        # neck/head
    _L, _L, _L,    # left arm
    _R, _R, _R,    # right arm
]

# Joint colours (same logic)
JOINT_COLORS = [
    _C,             # 0  root
    _R, _R, _R,    # 1-3  right leg
    _L, _L, _L,    # 4-6  left leg
    _C, _C,        # 7-8  spine / thorax
    _C, _C,        # 9-10 nose / head
    _L, _L, _L,    # 11-13 left arm
    _R, _R, _R,    # 14-16 right arm
]

# COCO→H36M 2D keypoint indices (for 2D overlay; confidence in channel 2)
# Already remapped to H36M order in the npz
H36M_JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle',
    'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Nose', 'Head',
    'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist',
]


# ---------------------------------------------------------------------------
# 2-D overlay helpers
# ---------------------------------------------------------------------------

def _to_bgr(rgb: tuple) -> tuple:
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


def draw_2d_skeleton(frame: np.ndarray, kps: np.ndarray, conf_thr: float = 0.3) -> np.ndarray:
    """
    Draw H36M 17-joint 2D skeleton on a BGR frame.

    Args:
        frame:    BGR image (H, W, 3).
        kps:      (17, 3) array of (x, y, confidence) in pixel coords.
        conf_thr: joints below this confidence are skipped.

    Returns:
        Annotated BGR frame (copy).
    """
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(h, w) / 1000.0   # scale stroke width to image resolution

    # Draw bones
    for (i, j), color in zip(BONES, BONE_COLORS):
        if kps[i, 2] < conf_thr or kps[j, 2] < conf_thr:
            continue
        pt1 = (int(kps[i, 0]), int(kps[i, 1]))
        pt2 = (int(kps[j, 0]), int(kps[j, 1]))
        cv2.line(out, pt1, pt2, _to_bgr(color), thickness=max(2, int(3 * scale)), lineType=cv2.LINE_AA)

    # Draw joints
    for idx, (x, y, c) in enumerate(kps):
        if c < conf_thr:
            continue
        cv2.circle(out, (int(x), int(y)), max(4, int(5 * scale)),
                   _to_bgr(JOINT_COLORS[idx]), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), max(4, int(5 * scale)),
                   (255, 255, 255), max(1, int(1.5 * scale)), lineType=cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# 3-D rendering helpers
# ---------------------------------------------------------------------------

def render_3d_frame(ax: plt.Axes, pose: np.ndarray, azim: float) -> None:
    """
    Draw a single H36M 17-joint 3D skeleton on a matplotlib 3D axes.

    The coordinate system is rotated so the skeleton looks upright:
      x_plot = x,  y_plot = -z,  z_plot = -y

    Args:
        ax:   matplotlib 3D axes (cleared before drawing).
        pose: (17, 3) root-relative xyz in H36M convention.
        azim: camera azimuth in degrees.
    """
    ax.cla()

    # Axis permutation: H36M y-up → matplotlib z-up
    xs =  pose[:, 0]
    ys = -pose[:, 2]
    zs = -pose[:, 1]

    # Draw bones
    for (i, j), color in zip(BONES, BONE_COLORS):
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                color=color, linewidth=2.5, zorder=1)

    # Draw joints
    ax.scatter(xs, ys, zs, c=[JOINT_COLORS[k] for k in range(17)],
               s=30, zorder=2, edgecolors='white', linewidths=0.5)

    # Consistent axis limits based on body scale
    radius = 0.7
    mid_x, mid_y, mid_z = xs.mean(), ys.mean(), zs.mean()
    ax.set_xlim(mid_x - radius, mid_x + radius)
    ax.set_ylim(mid_y - radius, mid_y + radius)
    ax.set_zlim(mid_z - radius, mid_z + radius)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=15, azim=azim)
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='grey', labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('grey')


def fig_to_array(fig: plt.Figure) -> np.ndarray:
    """Convert a matplotlib figure to a (H, W, 3) uint8 BGR array."""
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


# ---------------------------------------------------------------------------
# Main visualisation loop
# ---------------------------------------------------------------------------

def visualise(
    npz_path: str,
    video_path: str,
    output_path: str,
    fps: float | None,
    start: int,
    end: int | None,
    azim: float,
) -> None:
    # Load 3D predictions and 2D detections
    data       = np.load(npz_path)
    poses_3d   = data['poses_3d']    # (T, 17, 3)
    kps_2d     = data['keypoints_2d']  # (T, 17, 3) — x, y, confidence

    # Open source video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = fps if fps else src_fps

    T = len(poses_3d)
    end = min(end, T) if end is not None else T
    n_frames = end - start

    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # Read one frame to get dimensions
    ok, sample = cap.read()
    if not ok:
        raise RuntimeError('Cannot read first frame from video.')
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    vid_h, vid_w = sample.shape[:2]
    panel_w = 640
    panel_h = int(panel_w * vid_h / vid_w)
    out_w   = panel_w * 2
    out_h   = panel_h

    # Set up output video writer
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))

    # Set up matplotlib figure for 3D panel (reused each frame)
    fig = plt.figure(figsize=(panel_w / 100, panel_h / 100), dpi=100)
    fig.patch.set_facecolor('#1a1a2e')
    ax3d = fig.add_subplot(111, projection='3d')

    print(f'Rendering {n_frames} frames → {output_path}')

    for i in tqdm(range(n_frames)):
        frame_idx = start + i
        ok, frame = cap.read()
        if not ok:
            break

        # --- Left panel: video + 2D skeleton ---
        frame_2d = draw_2d_skeleton(frame, kps_2d[frame_idx])
        left = cv2.resize(frame_2d, (panel_w, panel_h))

        # --- Right panel: 3D skeleton ---
        render_3d_frame(ax3d, poses_3d[frame_idx], azim)
        right = cv2.resize(fig_to_array(fig), (panel_w, panel_h))

        # Concatenate and write
        combined = np.concatenate([left, right], axis=1)
        writer.write(combined)

    cap.release()
    writer.release()
    plt.close(fig)
    print(f'Saved → {output_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Visualise DiffPose-Video 3D keypoints')
    parser.add_argument('--npz',    required=True, help='Path to .npz output from infer.py')
    parser.add_argument('--video',  required=True, help='Path to the original source video')
    parser.add_argument('--output', required=True, help='Path to save the output .mp4')
    parser.add_argument('--fps',    type=float, default=None,
                        help='Output FPS (default: match source video)')
    parser.add_argument('--start',  type=int,   default=0,
                        help='First frame to render (default: 0)')
    parser.add_argument('--end',    type=int,   default=None,
                        help='Last frame to render, exclusive (default: all)')
    parser.add_argument('--azim',   type=float, default=70,
                        help='Initial 3D camera azimuth in degrees (default: 70)')
    return parser.parse_args()


def main():
    args = parse_args()
    visualise(
        npz_path=args.npz,
        video_path=args.video,
        output_path=args.output,
        fps=args.fps,
        start=args.start,
        end=args.end,
        azim=args.azim,
    )


if __name__ == '__main__':
    main()
