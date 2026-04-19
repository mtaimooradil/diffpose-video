# DiffPose-Video

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
python download_checkpoints.py
```

Downloads `mixste_cpn_243f.bin` and `diffpose_video_uvxyz_cpn.pth` to `~/.cache/diffpose_video/checkpoints/`.

---

## Usage

### 1. Run inference on a video

```bash
diffpose-infer \
  --input        video.mp4 \
  --config       configs/human36m_diffpose_uvxyz_cpn.yml \
  --model_pose   ~/.cache/diffpose_video/checkpoints/mixste_cpn_243f.bin \
  --model_diff   ~/.cache/diffpose_video/checkpoints/diffpose_video_uvxyz_cpn.pth \
  --output_dir   results/
```

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

```bash
docker build -t diffpose-video .

docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/results:/workspace/results \
  -v /path/to/videos:/videos \
  diffpose-video \
  diffpose-infer --input /videos/clip.mp4 \
    --config configs/human36m_diffpose_uvxyz_cpn.yml \
    --model_pose checkpoints/mixste_cpn_243f.bin \
    --model_diff checkpoints/diffpose_video_uvxyz_cpn.pth
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
