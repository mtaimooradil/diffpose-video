"""Tests for diffpose_video.download_checkpoints."""

import sys
import zipfile
from pathlib import Path

from diffpose_video import download_checkpoints as dc


class TestProgress:
    def test_prints_progress_when_total_known(self, capsys):
        dc._progress(block_num=1, block_size=512, total_size=1024)
        out = capsys.readouterr().out
        assert "50.0%" in out

    def test_silent_when_total_unknown(self, capsys):
        dc._progress(block_num=1, block_size=512, total_size=0)
        assert capsys.readouterr().out == ""


class TestDownloadCheckpoints:
    def test_skips_download_when_all_files_exist(self, tmp_path, monkeypatch, capsys):
        for name in dc.EXPECTED_FILES:
            (tmp_path / name).write_bytes(b"ok")

        def _never_called(*args, **kwargs):  # pragma: no cover
            raise AssertionError("urlretrieve should not be called")

        monkeypatch.setattr(dc.urllib.request, "urlretrieve", _never_called)

        dc.download_checkpoints(tmp_path)
        out = capsys.readouterr().out
        assert "All checkpoints already present" in out
        assert not (tmp_path / "_checkpoints.zip").exists()

    def test_download_extracts_and_maps_files(self, tmp_path, monkeypatch):
        # Pre-existing file should not be overwritten.
        (tmp_path / "mixste_cpn_243f.bin").write_bytes(b"original")

        def _fake_urlretrieve(url, filename, reporthook=None):
            with zipfile.ZipFile(filename, "w") as zf:
                zf.writestr("nested/mixste_cpn_243f.bin", b"new-mixste")
                zf.writestr("nested/diffpose_video_uvxyz_cpn.pth", b"cpn")
                zf.writestr("nested/diffpose_uvxyz_gt.pth", b"gt")
                zf.writestr("nested/ignore.txt", b"ignore")
            if reporthook is not None:
                reporthook(1, 100, 200)
            return str(filename), None

        monkeypatch.setattr(dc.urllib.request, "urlretrieve", _fake_urlretrieve)

        dc.download_checkpoints(tmp_path)

        assert (tmp_path / "mixste_cpn_243f.bin").read_bytes() == b"original"
        assert (tmp_path / "diffpose_video_uvxyz_cpn.pth").read_bytes() == b"cpn"
        assert (tmp_path / "diffpose_video_uvxyz_gt.pth").read_bytes() == b"gt"
        assert not (tmp_path / "_checkpoints.zip").exists()

    def test_default_checkpoint_dir(self):
        assert dc.default_checkpoint_dir() == dc.DEFAULT_DIR

    def test_main_passes_dest_to_download(self, tmp_path, monkeypatch):
        seen = {}

        def _fake_download(dest: Path):
            seen["dest"] = dest

        monkeypatch.setattr(dc, "download_checkpoints", _fake_download)
        monkeypatch.setattr(sys, "argv", ["diffpose-download", "--dest", str(tmp_path)])

        dc.main()
        assert seen["dest"] == tmp_path
