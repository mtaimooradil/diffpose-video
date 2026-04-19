# DiffPose-Video

[![PyPI version](https://img.shields.io/pypi/v/diffpose-video)](https://pypi.org/project/diffpose-video/)
[![Python](https://img.shields.io/pypi/pyversions/diffpose-video)](https://pypi.org/project/diffpose-video/)
[![CI](https://github.com/mtaimooradil/diffpose-video/actions/workflows/ci.yml/badge.svg)](https://github.com/mtaimooradil/diffpose-video/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

3D human pose estimation from arbitrary video using **MixSTE** (2D→3D lifting) and **DiffPose** (diffusion-based refinement).

This package wraps the original [DiffPose](https://github.com/GONGJIA0208/Diffpose) research code with a clean inference pipeline, an interactive visualisation dashboard, and a video renderer — all accessible as CLI commands after a single `pip install`.

> **Paper:** [DiffPose: Toward More Reliable 3D Pose Estimation](https://arxiv.org/abs/2211.16940), CVPR 2023.

---

## Install

```bash
# 1. Install PyTorch for your CUDA version first (example: CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install this package
pip install diffpose-video
```

> **Note:** `onnxruntime-gpu==1.20.1` is pinned because later versions have a broken CUDA provider on some systems.

---

## Download pretrained checkpoints

```bash
diffpose-download
```

Downloads all pretrained weights to `~/.cache/diffpose_video/checkpoints/`. Safe to re-run — skips files that already exist.

---

## Usage

### 1. Run inference on a video

```bash
diffpose-infer --input video.mp4 --output_dir results/
```

Config and checkpoint paths default to the bundled config and `~/.cache/diffpose_video/checkpoints/` (populated by `diffpose-download`). Override with `--config`, `--model_pose`, `--model_diff` if needed.

Output: `results/<video_name>.npz` containing:
- `poses_3d` — `(T, 17, 3)` root-relative 3D joint positions
- `keypoints_2d` — `(T, 17, 3)` pixel-space 2D detections + confidence

Process a whole folder of videos by passing a directory to `--input`.

### 2. Interactive dashboard

```bash
diffpose-explore \
  --npz   results/video.npz \
  --video video.mp4 \
  --fps   30
```

Opens a Plotly Dash app at `http://localhost:8050` with:
- Synchronized video playback with 2D skeleton overlay
- Animated 3D skeleton
- X / Y / Z trajectory graphs per joint
- Play/pause + frame scrubber, all linked

### 3. Render a side-by-side MP4

```bash
diffpose-visualise \
  --npz    results/video.npz \
  --video  video.mp4 \
  --output results/video_vis.mp4
```

Produces a video with the original footage (+ 2D overlay) on the left and the animated 3D skeleton on the right.

---

## Docker

The image is self-contained — checkpoints and config are baked in at build time.

```bash
# Build once
docker compose build

# Inference (outputs land in ./results/)
export VIDEOS_DIR=/path/to/your/videos
docker compose run infer --input /videos/clip.mp4

# Render side-by-side MP4
docker compose run visualise \
  --npz /results/clip.npz --video /videos/clip.mp4 --output /results/clip_vis.mp4

# Interactive dashboard — open http://localhost:8050
docker compose run --service-ports explore \
  --npz /results/clip.npz --video /videos/clip.mp4
```

---

## Citation

```bibtex
@InProceedings{gong2023diffpose,
    author    = {Gong, Jia and Foo, Lin Geng and Fan, Zhipeng and Ke, Qiuhong and Rahmani, Hossein and Liu, Jun},
    title     = {DiffPose: Toward More Reliable 3D Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
}
```
