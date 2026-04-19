"""Tests for diffpose_video.scripts.explore — pure figure/helper functions."""

import numpy as np
import pytest
import plotly.graph_objects as go

from diffpose_video.scripts.explore import (
    NPZ_DATA,
    _rgb,
    build_skeleton_figure,
    build_trajectory_figure,
    _empty_skeleton,
    BONES,
    JOINT_COLORS,
)


@pytest.fixture
def pose():
    """A single (17, 3) pose with spread-out joints."""
    rng = np.random.default_rng(0)
    return rng.random((17, 3)).astype(np.float32)


@pytest.fixture
def poses_3d(pose):
    """(50, 17, 3) sequence."""
    return np.tile(pose, (50, 1, 1))


@pytest.fixture(autouse=True)
def populate_npz(poses_3d):
    """Populate NPZ_DATA for the duration of each test, then clean up."""
    NPZ_DATA['vid_a'] = {'poses_3d': poses_3d, 'keypoints_2d': None, 'npz_path': '/tmp/a.npz'}
    NPZ_DATA['vid_b'] = {'poses_3d': poses_3d, 'keypoints_2d': None, 'npz_path': '/tmp/b.npz'}
    yield
    NPZ_DATA.pop('vid_a', None)
    NPZ_DATA.pop('vid_b', None)


# ---------------------------------------------------------------------------
# _rgb
# ---------------------------------------------------------------------------

class TestRgb:
    def test_black(self):
        assert _rgb((0, 0, 0)) == 'rgb(0,0,0)'

    def test_white(self):
        assert _rgb((1, 1, 1)) == 'rgb(255,255,255)'

    def test_mid(self):
        assert _rgb((0.5, 0.0, 1.0)) == 'rgb(127,0,255)'

    def test_returns_string(self):
        assert isinstance(_rgb((0.2, 0.4, 0.8)), str)


# ---------------------------------------------------------------------------
# build_skeleton_figure
# ---------------------------------------------------------------------------

class TestBuildSkeletonFigure:
    def test_returns_figure(self, pose):
        fig = build_skeleton_figure(pose)
        assert isinstance(fig, go.Figure)

    def test_trace_count(self, pose):
        fig = build_skeleton_figure(pose)
        # One Scatter3d per bone + one for all joints
        assert len(fig.data) == len(BONES) + 1

    def test_bone_traces_have_two_points(self, pose):
        fig = build_skeleton_figure(pose)
        for trace in fig.data[: len(BONES)]:
            assert len(trace.x) == 2

    def test_joint_trace_has_17_points(self, pose):
        fig = build_skeleton_figure(pose)
        joint_trace = fig.data[-1]
        assert len(joint_trace.x) == 17

    def test_joint_colours_count(self, pose):
        fig = build_skeleton_figure(pose)
        joint_trace = fig.data[-1]
        assert len(joint_trace.marker.color) == len(JOINT_COLORS)

    def test_uirev_propagated(self, pose):
        fig = build_skeleton_figure(pose, uirev='my-rev')
        assert fig.layout.uirevision == 'my-rev'

    def test_dark_background(self, pose):
        fig = build_skeleton_figure(pose)
        assert fig.layout.paper_bgcolor == '#1a1a2e'


# ---------------------------------------------------------------------------
# _empty_skeleton
# ---------------------------------------------------------------------------

class TestEmptySkeleton:
    def test_returns_figure(self):
        assert isinstance(_empty_skeleton('rev'), go.Figure)

    def test_no_data_traces(self):
        assert len(_empty_skeleton('rev').data) == 0

    def test_has_annotation(self):
        fig = _empty_skeleton('rev')
        assert any('Select a video' in a.text for a in fig.layout.annotations)

    def test_uirev_propagated(self):
        fig = _empty_skeleton('my-rev')
        assert fig.layout.uirevision == 'my-rev'


# ---------------------------------------------------------------------------
# build_trajectory_figure
# ---------------------------------------------------------------------------

class TestBuildTrajectoryFigure:
    def test_returns_figure(self):
        fig = build_trajectory_figure('vid_a', None, 0, None, 30.0)
        assert isinstance(fig, go.Figure)

    def test_single_video_trace_count(self):
        # 3 dims × 1 trace each = 3 traces + 1 cursor shape
        fig = build_trajectory_figure('vid_a', None, 0, None, 30.0)
        assert len(fig.data) == 3

    def test_two_videos_trace_count(self):
        # 3 dims × 2 videos = 6 traces
        fig = build_trajectory_figure('vid_a', 'vid_b', 0, None, 30.0)
        assert len(fig.data) == 6

    def test_overlay_joint_adds_traces(self):
        fig_no_ov = build_trajectory_figure('vid_a', None, 0, None,  30.0)
        fig_ov    = build_trajectory_figure('vid_a', None, 0, 1,     30.0)
        assert len(fig_ov.data) == len(fig_no_ov.data) * 2

    def test_cursor_shape_present(self):
        fig = build_trajectory_figure('vid_a', None, 0, None, 30.0, cursor_frame=10)
        assert len(fig.layout.shapes) == 1
        assert fig.layout.shapes[0].type == 'line'

    def test_cursor_position(self):
        fps = 30.0
        frame = 15
        fig = build_trajectory_figure('vid_a', None, 0, None, fps, cursor_frame=frame)
        assert fig.layout.shapes[0].x0 == pytest.approx(frame / fps)

    def test_uirevision_changes_with_joint(self):
        fig0 = build_trajectory_figure('vid_a', None, 0, None, 30.0)
        fig1 = build_trajectory_figure('vid_a', None, 1, None, 30.0)
        assert fig0.layout.uirevision != fig1.layout.uirevision

    def test_three_subplots(self):
        fig = build_trajectory_figure('vid_a', None, 0, None, 30.0)
        # 3 subplot rows → 3 y-axes
        assert hasattr(fig.layout, 'yaxis') and hasattr(fig.layout, 'yaxis3')
