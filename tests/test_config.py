"""Tests for TOML config loading and _apply_config in all three scripts."""

import argparse

import pytest

from diffpose_video.common.config_loader import merge


# ---------------------------------------------------------------------------
# config_loader.merge
# ---------------------------------------------------------------------------

def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


class TestMerge:
    def test_fills_none_from_defaults(self):
        args = _ns(foo=None)
        merge(args, {}, {'foo': 'default'})
        assert args.foo == 'default'

    def test_cli_value_takes_priority(self):
        args = _ns(foo='cli')
        merge(args, {'foo': 'cfg'}, {'foo': 'default'})
        assert args.foo == 'cli'

    def test_cfg_overrides_default(self):
        args = _ns(foo=None)
        merge(args, {'foo': 'cfg'}, {'foo': 'default'})
        assert args.foo == 'cfg'

    def test_empty_list_filled_from_cfg(self):
        args = _ns(items=[])
        merge(args, {'items': ['a', 'b']}, {})
        assert args.items == ['a', 'b']

    def test_non_empty_list_not_overwritten(self):
        args = _ns(items=['x'])
        merge(args, {'items': ['a', 'b']}, {})
        assert args.items == ['x']

    def test_videos_key_skipped(self):
        args = _ns(videos=None)
        merge(args, {'videos': 'something'}, {'videos': 'default'})
        assert args.videos is None

    def test_unknown_cfg_key_set_on_args(self):
        args = _ns()
        merge(args, {'extra': 42}, {})
        assert args.extra == 42

    def test_false_bool_default_not_overwritten_by_none(self):
        args = _ns(flag=None)
        merge(args, {}, {'flag': False})
        assert args.flag is False


# ---------------------------------------------------------------------------
# infer._apply_config
# ---------------------------------------------------------------------------

class TestInferApplyConfig:
    def _args(self, **kwargs):
        defaults = dict(
            config=None, input=None, output_dir=None, recursive=None,
            skip_existing=None, exclude=None, model_config=None,
            model_pose=None, model_diff=None, det_freq=None, device=None,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_defaults_applied(self, tmp_path):
        # Need a real input so SystemExit is not raised
        f = tmp_path / 'v.mp4'
        f.touch()
        args = self._args(input=[str(f)])
        from diffpose_video.scripts.infer import _apply_config
        _apply_config(args)
        assert args.output_dir == 'results'
        assert args.recursive is False
        assert args.device == 'cuda'
        assert args.det_freq == 1

    def test_no_input_raises(self):
        args = self._args()
        from diffpose_video.scripts.infer import _apply_config
        with pytest.raises(SystemExit):
            _apply_config(args)

    def test_toml_values_applied(self, tmp_path):
        toml = tmp_path / 'infer.toml'
        toml.write_text(
            f'input = ["{tmp_path}"]\noutput_dir = "myresults"\ndevice = "cpu"\n'
        )
        f = tmp_path / 'v.mp4'
        f.touch()
        args = self._args(config=str(toml))
        from diffpose_video.scripts.infer import _apply_config
        _apply_config(args)
        assert args.output_dir == 'myresults'
        assert args.device == 'cpu'

    def test_cli_overrides_toml(self, tmp_path):
        toml = tmp_path / 'infer.toml'
        toml.write_text(f'input = ["{tmp_path}"]\ndevice = "cpu"\n')
        (tmp_path / 'v.mp4').touch()
        args = self._args(config=str(toml), device='cuda')
        from diffpose_video.scripts.infer import _apply_config
        _apply_config(args)
        assert args.device == 'cuda'


# ---------------------------------------------------------------------------
# visualise._apply_config
# ---------------------------------------------------------------------------

class TestVisualiseApplyConfig:
    def _args(self, **kwargs):
        defaults = dict(
            config=None, npz=None, video=None, output=None,
            results_dir=None, videos_dir=None, output_dir=None,
            skip_existing=None, fps=None, start=None, end=None, azim=None,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_defaults_applied(self):
        args = self._args()
        from diffpose_video.scripts.visualise import _apply_config
        _apply_config(args)
        assert args.output_dir == 'visualisations'
        assert args.start == 0
        assert args.azim == 70
        assert args.skip_existing is False

    def test_toml_values_applied(self, tmp_path):
        toml = tmp_path / 'vis.toml'
        toml.write_text('results_dir = "results"\nvideos_dir = "/vids"\nazim = 45\n')
        args = self._args(config=str(toml))
        from diffpose_video.scripts.visualise import _apply_config
        _apply_config(args)
        assert args.results_dir == 'results'
        assert args.azim == 45

    def test_cli_overrides_toml(self, tmp_path):
        toml = tmp_path / 'vis.toml'
        toml.write_text('azim = 45\n')
        args = self._args(config=str(toml), azim=90.0)
        from diffpose_video.scripts.visualise import _apply_config
        _apply_config(args)
        assert args.azim == 90.0


# ---------------------------------------------------------------------------
# explore._apply_config
# ---------------------------------------------------------------------------

class TestExploreApplyConfig:
    def _args(self, **kwargs):
        defaults = dict(
            config=None, npz=None, video=None, results_dir=None,
            videos_dir=None, videos_map=None, fps=None, port=None,
            output_dir=None,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_defaults_applied(self):
        args = self._args()
        from diffpose_video.common.config_loader import apply_explore_config as _apply_config
        _apply_config(args)
        assert args.fps == 30.0
        assert args.port == 8050
        assert args.output_dir == 'visualisations'
        assert args.videos_map == []

    def test_videos_section_sets_videos_dir(self, tmp_path):
        toml = tmp_path / 'explore.toml'
        toml.write_text(
            'results_dir = "results"\n\n[videos]\ndefault = "/data/vids"\n'
        )
        args = self._args(config=str(toml))
        from diffpose_video.common.config_loader import apply_explore_config as _apply_config
        _apply_config(args)
        assert args.videos_dir == '/data/vids'

    def test_videos_section_builds_map(self, tmp_path):
        toml = tmp_path / 'explore.toml'
        toml.write_text(
            '[videos]\ndefault = "/data"\nCam1 = "/data/Cam1"\n'
        )
        args = self._args(config=str(toml))
        from diffpose_video.common.config_loader import apply_explore_config as _apply_config
        _apply_config(args)
        assert 'Cam1:/data/Cam1' in args.videos_map

    def test_cli_overrides_toml_port(self, tmp_path):
        toml = tmp_path / 'explore.toml'
        toml.write_text('port = 9000\n')
        args = self._args(config=str(toml), port=8051)
        from diffpose_video.common.config_loader import apply_explore_config as _apply_config
        _apply_config(args)
        assert args.port == 8051
