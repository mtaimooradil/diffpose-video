# DiffPose-Video

[![PyPI version](https://img.shields.io/pypi/v/diffpose-video)](https://pypi.org/project/diffpose-video/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/diffpose-video/)
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

All three commands accept either CLI flags or a `--config <file>.toml`. Example TOML files are in [`configs/`](configs/) — copy and edit one with your own paths before use. These are separate from the bundled model config (`diffpose_video/configs/*.yml`) which is handled automatically.

### 1. Run inference

**Single video:**
```bash
diffpose-infer --input video.mp4 --output_dir results/
```

**Batch (whole directory):**
```bash
diffpose-infer --input /datasets/videos/ --recursive --skip_existing --output_dir results/
```

**Via config file:**
```bash
diffpose-infer --config infer.toml
```

Output: `results/<video_name>.npz` containing:
- `poses_3d` — `(T, 17, 3)` root-relative 3D joint positions
- `keypoints_2d` — `(T, 17, 3)` pixel-space 2D detections + confidence

Model and checkpoint paths default to the bundled config and `~/.cache/diffpose_video/checkpoints/`. Override with `--model_config`, `--model_pose`, `--model_diff` if needed.

**Key options:**
| Flag | Default | Description |
|---|---|---|
| `--input` | — | Video file(s) and/or directory paths |
| `--output_dir` | `results/` | Root output directory |
| `--recursive` | off | Recurse into subdirectories |
| `--skip_existing` | off | Skip if `.npz` already exists |
| `--exclude PATTERN` | — | Exclude files matching pattern(s) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--det_freq N` | `1` | Run person detector every N frames |

---

### 2. Interactive dashboard

**Single video:**
```bash
diffpose-explore --npz results/video.npz --video video.mp4
```

**Multi-video (browse & compare):**
```bash
diffpose-explore --results_dir results/ --videos_dir /path/to/videos/
```

**Multi-camera with directory mapping:**
```bash
diffpose-explore \
  --results_dir results/ \
  --videos_map Cam1:/data/Cam1/InputMedia Cam2:/data/Cam2/InputMedia \
  --videos_dir /data/default_videos/
```

**Via config file** (recommended for multi-camera setups):
```bash
diffpose-explore --config explore.toml
```

```toml
# explore.toml
results_dir = "results"
output_dir  = "visualisations"
fps         = 30.0
port        = 8050

[videos]
default = "/data/default_videos"
Cam1    = "/data/Cam1/InputMedia"
Cam2    = "/data/Cam2/InputMedia"
```

Opens a Plotly Dash app at `http://localhost:8050` with:
- Synchronized video playback with 2D skeleton overlay
- Side-by-side A/B comparison of any two results
- Animated 3D skeleton
- X / Y / Z trajectory graphs per joint, with joint selector
- Play/pause + frame scrubber, all linked
- "Render" buttons to generate side-by-side MP4s on demand

---

### 3. Render side-by-side MP4

**Single video:**
```bash
diffpose-visualise \
  --npz results/video.npz --video video.mp4 --output results/video_vis.mp4
```

**Batch (whole results directory):**
```bash
diffpose-visualise \
  --results_dir results/ --videos_dir /path/to/videos/ --output_dir visualisations/
```

**Via config file:**
```bash
diffpose-visualise --config visualise.toml
```

Produces a video with the original footage (+ 2D overlay) on the left and the animated 3D skeleton on the right.

**Key options:**
| Flag | Default | Description |
|---|---|---|
| `--skip_existing` | off | Skip if output MP4 already exists |
| `--fps` | source FPS | Output frame rate |
| `--start` / `--end` | full range | Frame range to render |
| `--azim` | `70` | Initial 3D camera azimuth (degrees) |

---

## Docker

The image is self-contained — checkpoints and config are baked in at build time.

```bash
# Build once
docker compose build
```

Set paths via environment variables (or export them):
```bash
export VIDEOS_DIR=/path/to/your/videos
export RESULTS_DIR=/path/to/your/results
export VIS_DIR=/path/to/your/visualisations
export CONFIGS_DIR=/path/to/your/toml/configs   # optional
```

**Inference:**
```bash
# Single video
docker compose run infer --input /videos/clip.mp4

# Batch — whole directory, recursive
docker compose run infer --input /videos/ --recursive --skip_existing

# Via config file
docker compose run infer --config /configs/infer.toml
```

**Render side-by-side MP4:**
```bash
# Single video
docker compose run visualise \
  --npz /results/clip.npz --video /videos/clip.mp4 --output /vis/clip_vis.mp4

# Batch
docker compose run visualise --results_dir /results --videos_dir /videos

# Via config file
docker compose run visualise --config /configs/visualise.toml
```

**Interactive dashboard — open http://localhost:8050:**
```bash
docker compose run --service-ports explore \
  --results_dir /results --videos_dir /videos

# Multi-camera via config file
docker compose run --service-ports explore --config /configs/explore.toml
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
