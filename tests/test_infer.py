"""Tests for diffpose_video.scripts.infer — file collection helpers."""

import pytest

from diffpose_video.scripts.infer import _is_excluded, _is_video, collect_videos


class TestIsVideo:
    @pytest.mark.parametrize('fname', [
        'clip.mp4', 'video.MOV', 'recording.avi', 'file.mkv', 'demo.webm',
        'clip.m4v', 'footage.mov', 'data.MP4',
    ])
    def test_known_extensions(self, fname):
        assert _is_video(fname)

    @pytest.mark.parametrize('fname', [
        'data.npz', 'image.jpg', 'report.pdf', 'model.pth', 'videofile', '',
    ])
    def test_non_video(self, fname):
        assert not _is_video(fname)


class TestIsExcluded:
    def test_substring_match(self):
        assert _is_excluded('clip_sync.mp4', ['_sync'])

    def test_no_match(self):
        assert not _is_excluded('clip.mp4', ['_sync', 'neutral'])

    def test_case_insensitive(self):
        assert _is_excluded('Neutral.mp4', ['neutral'])
        assert _is_excluded('clip_SYNC.mp4', ['_sync'])

    def test_glob_wildcard(self):
        assert _is_excluded('neutral.mp4', ['neutral*'])
        assert _is_excluded('neutral_walk.mp4', ['neutral*'])
        assert not _is_excluded('clip.mp4', ['neutral*'])

    def test_question_mark_glob(self):
        assert _is_excluded('clip1.mp4', ['clip?.mp4'])
        assert not _is_excluded('clip12.mp4', ['clip?.mp4'])

    def test_empty_patterns(self):
        assert not _is_excluded('clip.mp4', [])

    def test_empty_string_pattern_matches_everything(self):
        assert _is_excluded('clip.mp4', [''])

    def test_multiple_patterns_any_match(self):
        assert _is_excluded('neutral.mp4', ['_sync', 'neutral'])


class TestCollectVideos:
    def test_single_file(self, tmp_path):
        f = tmp_path / 'clip.mp4'
        f.touch()
        result = collect_videos([str(f)])
        assert len(result) == 1
        assert result[0][0] == str(f)
        assert result[0][1] == ''

    def test_skips_non_video_file(self, tmp_path, capsys):
        f = tmp_path / 'data.npz'
        f.touch()
        (tmp_path / 'clip.mp4').touch()
        result = collect_videos([str(tmp_path)])
        assert all(r[0].endswith('.mp4') for r in result)

    def test_directory_flat(self, tmp_path):
        (tmp_path / 'a.mp4').touch()
        (tmp_path / 'b.mp4').touch()
        (tmp_path / 'data.npz').touch()
        result = collect_videos([str(tmp_path)])
        assert len(result) == 2

    def test_directory_not_recursive_by_default(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (tmp_path / 'a.mp4').touch()
        (sub / 'b.mp4').touch()
        result = collect_videos([str(tmp_path)], recursive=False)
        assert len(result) == 1

    def test_recursive(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (tmp_path / 'a.mp4').touch()
        (sub / 'b.mp4').touch()
        result = collect_videos([str(tmp_path)], recursive=True)
        assert len(result) == 2

    def test_recursive_preserves_subdir(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'b.mp4').touch()
        result = collect_videos([str(tmp_path)], recursive=True)
        assert result[0][1] == 'sub'

    def test_exclude_pattern(self, tmp_path):
        (tmp_path / 'clip.mp4').touch()
        (tmp_path / 'clip_sync.mp4').touch()
        result = collect_videos([str(tmp_path)], exclude=['_sync'])
        assert len(result) == 1
        assert '_sync' not in result[0][0]

    def test_mixed_files_and_dirs(self, tmp_path):
        f = tmp_path / 'direct.mp4'
        f.touch()
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'nested.mp4').touch()
        result = collect_videos([str(f), str(sub)])
        stems = {r[0].split('/')[-1] for r in result}
        assert 'direct.mp4' in stems
        assert 'nested.mp4' in stems

    def test_missing_input_skipped(self, tmp_path, capsys):
        (tmp_path / 'real.mp4').touch()
        result = collect_videos([str(tmp_path / 'real.mp4'),
                                  str(tmp_path / 'ghost.mp4')])
        assert len(result) == 1

    def test_all_excluded_exits(self, tmp_path):
        (tmp_path / 'clip_sync.mp4').touch()
        with pytest.raises(SystemExit):
            collect_videos([str(tmp_path)], exclude=['_sync'])
