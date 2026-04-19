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
import urllib.request
import zipfile
from pathlib import Path

CHECKPOINTS_URL = (
    "https://www.dropbox.com/sh/jhwz3ypyxtyrlzv/AABivC5oiiMdgPePxekzu6vga?dl=1"
)

# Maps the filename inside the zip → the name to save it as locally.
# Some entries in the zip use slightly different names than the config expects.
ZIP_TO_LOCAL = {
    "mixste_cpn_243f.bin":       "mixste_cpn_243f.bin",
    "diffpose_video_uvxyz_cpn.pth": "diffpose_video_uvxyz_cpn.pth",
    "diffpose_uvxyz_gt.pth":     "diffpose_video_uvxyz_gt.pth",
}

EXPECTED_FILES = list(ZIP_TO_LOCAL.values())

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

    missing = [f for f in EXPECTED_FILES if not (dest / f).exists()]
    if not missing:
        print("All checkpoints already present — nothing to download.")
        return

    print(f"Missing: {', '.join(missing)}")
    zip_path = dest / "_checkpoints.zip"

    print("Downloading checkpoint archive ...")
    urllib.request.urlretrieve(CHECKPOINTS_URL, zip_path, reporthook=_progress)
    print()

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            zip_name = Path(member).name
            if zip_name in ZIP_TO_LOCAL:
                local_name = ZIP_TO_LOCAL[zip_name]
                if not (dest / local_name).exists():
                    data = zf.read(member)
                    (dest / local_name).write_bytes(data)
                    print(f"  extracted → {dest / local_name}")

    zip_path.unlink()
    print("\nAll checkpoints ready.")


def default_checkpoint_dir() -> Path:
    return DEFAULT_DIR


def main():
    parser = argparse.ArgumentParser(description="Download DiffPose-Video checkpoints")
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DIR),
        help=f"Directory to save checkpoints (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args()
    download_checkpoints(Path(args.dest))


if __name__ == "__main__":
    main()
