"""
Download DiffPose-Video pretrained checkpoints.

Saves to ~/.cache/diffpose_video/checkpoints/ by default.
Run once before using diffpose-infer.

Usage
-----
    python download_checkpoints.py
    python download_checkpoints.py --dest /path/to/checkpoints
"""

import argparse
import os
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "mixste_cpn_243f.bin": "https://www.dropbox.com/scl/fi/n7xq0s4f6bhbpwdph22we/mixste_cpn_243f.bin?rlkey=yetcjd9yi2mrdvbrnhb3ydpzh&dl=1",
    "diffpose_video_uvxyz_cpn.pth": "https://www.dropbox.com/scl/fi/t0p7n8ndqn4rexb1brcio/diffpose_video_uvxyz_cpn.pth?rlkey=t1x9o2efz8owp0dex3hmq8v0j&dl=1",
    "diffpose_video_uvxyz_gt.pth": "https://www.dropbox.com/scl/fi/i8o3mh84y0djrxxsrpbey/diffpose_video_uvxyz_gt.pth?rlkey=h4jfqjh9myzb3y8vjlyge87g0&dl=1",
}

DEFAULT_DIR = Path.home() / ".cache" / "diffpose_video" / "checkpoints"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {pct:5.1f}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_checkpoints(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in CHECKPOINTS.items():
        out = dest / name
        if out.exists():
            print(f"  [skip] {name} already exists")
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, out, reporthook=_progress)
        print(f"\r  done → {out}")
    print("\nAll checkpoints ready.")


def default_checkpoint_dir() -> Path:
    return DEFAULT_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DiffPose-Video checkpoints")
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DIR),
        help=f"Directory to save checkpoints (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args()
    download_checkpoints(Path(args.dest))
