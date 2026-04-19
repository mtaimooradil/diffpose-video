"""Additional tests for uncovered helper logic in scripts."""

import sys

import matplotlib.pyplot as plt
import pytest
import yaml
import diffpose_video

from diffpose_video.scripts import infer, visualise


class TestInferHelpers:
    def test_default_config_uses_package_configs_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(diffpose_video, "configs_dir", lambda: tmp_path)
        assert infer._default_config() == str(tmp_path / "human36m_diffpose_uvxyz_cpn.yml")

    def test_default_checkpoint_uses_default_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "diffpose_video.download_checkpoints.DEFAULT_DIR",
            tmp_path,
        )
        assert infer._default_checkpoint("model.bin") == str(tmp_path / "model.bin")

    def test_load_config_returns_nested_namespace(self, tmp_path):
        cfg_file = tmp_path / "cfg.yml"
        cfg_file.write_text(
            yaml.safe_dump(
                {"outer": {"inner": 7}, "name": "demo"},
                sort_keys=True,
            )
        )
        cfg = infer.load_config(str(cfg_file))
        assert cfg.outer.inner == 7
        assert cfg.name == "demo"

    def test_parse_args_parses_flags_and_calls_apply_config(self, monkeypatch):
        called = {}

        def _fake_apply_config(args):
            called["args"] = args

        monkeypatch.setattr(infer, "_apply_config", _fake_apply_config)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "diffpose-infer",
                "--input",
                "a.mp4",
                "b.mov",
                "--output_dir",
                "results2",
                "--recursive",
                "--exclude",
                "_sync",
                "neutral*",
                "--det_freq",
                "4",
                "--device",
                "cpu",
            ],
        )

        args = infer.parse_args()
        assert called["args"] is args
        assert args.input == ["a.mp4", "b.mov"]
        assert args.output_dir == "results2"
        assert args.recursive is True
        assert args.exclude == ["_sync", "neutral*"]
        assert args.det_freq == 4
        assert args.device == "cpu"


class TestVisualiseHelpers:
    def test_to_bgr_converts_rgb_float_tuple(self):
        assert visualise._to_bgr((0.1, 0.5, 1.0)) == (255, 127, 25)

    def test_render_3d_frame_draws_expected_structure(self):
        pose = (pytest.importorskip("numpy").arange(51, dtype=float)).reshape(17, 3)
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(111, projection="3d")
        visualise.render_3d_frame(ax, pose, azim=35.0)

        assert len(ax.lines) == len(visualise.BONES)
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert ax.get_zlabel() == "Z"
        assert ax.azim == pytest.approx(35.0)
        plt.close(fig)

    def test_fig_to_array_returns_uint8_bgr_image(self):
        np = pytest.importorskip("numpy")
        fig = plt.figure(figsize=(1, 1), dpi=40)
        fig.patch.set_facecolor((1.0, 0.0, 0.0, 1.0))  # red in RGB
        arr = visualise.fig_to_array(fig)
        plt.close(fig)

        assert arr.ndim == 3
        assert arr.shape[2] == 3
        assert arr.dtype == np.uint8
        # Converted to BGR, so red channel index 2 should dominate blue index 0.
        assert arr[..., 2].mean() > arr[..., 0].mean()
