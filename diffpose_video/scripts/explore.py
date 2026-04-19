"""
Interactive joint trajectory explorer — Plotly Dash edition.

Single-video mode (backward-compatible):
    diffpose-explore --npz results/clip.npz --video clip.mp4 --fps 30

Multi-video / comparison mode:
    diffpose-explore --results_dir results/ --videos_dir /path/to/videos/ --fps 30

In comparison mode, pick Video A and Video B from the sidebar dropdowns.
Trajectories for both are overlaid in the graph; skeletons and video frames
are shown side-by-side.
"""

import argparse
import threading
from pathlib import Path

import cv2
import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html, Patch
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# H36M skeleton definitions
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    'Hip (root)',
    'R Hip', 'R Knee', 'R Ankle',
    'L Hip', 'L Knee', 'L Ankle',
    'Spine', 'Thorax', 'Nose', 'Head',
    'L Shoulder', 'L Elbow', 'L Wrist',
    'R Shoulder', 'R Elbow', 'R Wrist',
]

BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8),
    (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

_R = (0.2, 0.4, 0.8)
_L = (0.8, 0.2, 0.2)
_C = (0.2, 0.7, 0.3)

BONE_COLORS  = [_R,_R,_R, _L,_L,_L, _C,_C, _C,_C, _L,_L,_L, _R,_R,_R]
JOINT_COLORS = [_C, _R,_R,_R, _L,_L,_L, _C,_C, _C,_C, _L,_L,_L, _R,_R,_R]

# Trajectory colours: A uses solid, B uses a second palette (orange/teal/purple)
TRAJ_A = {'X': '#e74c3c', 'Y': '#27ae60', 'Z': '#2471a3'}
TRAJ_B = {'X': '#e67e22', 'Y': '#16a085', 'Z': '#8e44ad'}

DROPDOWN_OPTIONS = [{'label': n, 'value': i} for i, n in enumerate(JOINT_NAMES)]
OVERLAY_OPTIONS  = [{'label': '(none)', 'value': -1}] + DROPDOWN_OPTIONS

VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']

# Server-side data stores (populated at startup, read-only during app runtime)
NPZ_DATA: dict[str, dict] = {}          # name → {poses_3d, keypoints_2d, npz_path}
VIDEO_FRAMES: dict[str, list[bytes] | None] = {}  # name → JPEG bytes list (lazy)
VIDEO_PATHS: dict[str, str | None] = {}  # name → original video file path
VIDEOS_DIR: Path | None = None           # default fallback videos directory
VIDEOS_MAP: dict[str, Path] = {}         # result key prefix → videos directory
OUTPUT_DIR: str = 'visualisations'

# Video generation job tracking
JOBS: dict[str, str] = {}   # name → status string
_JOBS_LOCK = threading.Lock()


def _rgb(c):
    return f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def _to_bgr(rgb):
    return (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))


def draw_2d_skeleton(frame: np.ndarray, kps: np.ndarray, conf_thr: float = 0.3):
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(h, w) / 1000.0
    for (i, j), color in zip(BONES, BONE_COLORS):
        if kps[i, 2] < conf_thr or kps[j, 2] < conf_thr:
            continue
        cv2.line(out, (int(kps[i,0]), int(kps[i,1])), (int(kps[j,0]), int(kps[j,1])),
                 _to_bgr(color), thickness=max(2, int(3*scale)), lineType=cv2.LINE_AA)
    for idx, (x, y, c) in enumerate(kps):
        if c < conf_thr:
            continue
        r = max(4, int(5*scale))
        cv2.circle(out, (int(x), int(y)), r, _to_bgr(JOINT_COLORS[idx]), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), r, (255,255,255), max(1, int(1.5*scale)), cv2.LINE_AA)
    return out


def extract_video_frames(video_path: str, kps_2d=None, max_width: int = 560) -> list[bytes]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if kps_2d is not None and idx < len(kps_2d):
            frame = draw_2d_skeleton(frame, kps_2d[idx])
        h, w = frame.shape[:2]
        # Constrain so portrait height doesn't exceed landscape equivalent
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
        elif h > max_width:
            # Portrait: constrain by height instead
            new_h = max_width
            new_w = int(w * new_h / h)
            frame = cv2.resize(frame, (new_w, new_h))
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frames.append(buf.tobytes())
        idx += 1
    cap.release()
    return frames


def _videos_dir_for(name: str) -> Path | None:
    """Return the best videos directory for a result key, using VIDEOS_MAP then VIDEOS_DIR."""
    # Longest matching prefix wins
    best_prefix, best_dir = '', None
    for prefix, vdir in VIDEOS_MAP.items():
        if name.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix, best_dir = prefix, vdir
    return best_dir or VIDEOS_DIR


def get_video_frames(name: str) -> list[bytes] | None:
    """Return cached frame bytes for `name` (key may be 'Cam1/B01' or 'B01')."""
    if name in VIDEO_FRAMES:
        return VIDEO_FRAMES[name]
    vdir = _videos_dir_for(name)
    if vdir is None:
        VIDEO_FRAMES[name] = None
        return None
    stem = NPZ_DATA[name].get('stem', Path(name).name)
    # Search recursively; prefer exact stem match, skip *_sync variants
    candidates = []
    for ext in VIDEO_EXTENSIONS:
        candidates.extend(vdir.rglob(f'{stem}{ext}'))
    # Prefer non-sync files
    non_sync = [p for p in candidates if '_sync' not in p.stem]
    path = (non_sync or candidates)
    if path:
        path = path[0]
        VIDEO_PATHS[name] = str(path)
        kps_2d = NPZ_DATA[name].get('keypoints_2d')
        print(f'Loading video frames for {name} ({path}) ...')
        VIDEO_FRAMES[name] = extract_video_frames(str(path), kps_2d=kps_2d)
        return VIDEO_FRAMES[name]
    VIDEO_FRAMES[name] = None
    VIDEO_PATHS[name] = None
    return None


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_trajectory_figure(
    name_a: str,
    name_b: str | None,
    primary: int,
    overlay: int | None,
    fps: float,
    cursor_frame: int = 0,
    uirevision: str | None = None,
) -> go.Figure:
    data_a = NPZ_DATA[name_a]
    poses_a = data_a['poses_3d']
    T_a = poses_a.shape[0]
    time_a = np.arange(T_a) / fps

    poses_b = None
    time_b  = None
    if name_b and name_b in NPZ_DATA:
        poses_b = NPZ_DATA[name_b]['poses_3d']
        T_b = poses_b.shape[0]
        time_b = np.arange(T_b) / fps

    dims = ['X', 'Y', 'Z']
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f'<b>{d}</b>' for d in dims],
    )

    for row, dim in enumerate(dims, start=1):
        ca = TRAJ_A[dim]
        cb = TRAJ_B[dim]

        # Video A primary
        fig.add_trace(go.Scatter(
            x=time_a, y=poses_a[:, primary, row-1],
            mode='lines', name=f'A · {JOINT_NAMES[primary]} {dim}',
            line=dict(color=ca, width=2),
            legendgroup=f'a_{dim}', showlegend=(row == 1),
            hovertemplate=f'<b>A · {JOINT_NAMES[primary]}</b><br>t=%{{x:.2f}}s<br>{dim}=%{{y:.4f}}<extra></extra>',
        ), row=row, col=1)

        # Video A overlay joint
        if overlay is not None:
            fig.add_trace(go.Scatter(
                x=time_a, y=poses_a[:, overlay, row-1],
                mode='lines', name=f'A · {JOINT_NAMES[overlay]} {dim} (ov)',
                line=dict(color=ca, width=1.5, dash='dot'),
                opacity=0.55, legendgroup=f'a_{dim}_ov', showlegend=(row == 1),
            ), row=row, col=1)

        # Video B primary
        if poses_b is not None:
            fig.add_trace(go.Scatter(
                x=time_b, y=poses_b[:, primary, row-1],
                mode='lines', name=f'B · {JOINT_NAMES[primary]} {dim}',
                line=dict(color=cb, width=2, dash='dash'),
                legendgroup=f'b_{dim}', showlegend=(row == 1),
                hovertemplate=f'<b>B · {JOINT_NAMES[primary]}</b><br>t=%{{x:.2f}}s<br>{dim}=%{{y:.4f}}<extra></extra>',
            ), row=row, col=1)

            if overlay is not None:
                fig.add_trace(go.Scatter(
                    x=time_b, y=poses_b[:, overlay, row-1],
                    mode='lines', name=f'B · {JOINT_NAMES[overlay]} {dim} (ov)',
                    line=dict(color=cb, width=1.5, dash='dot'),
                    opacity=0.55, legendgroup=f'b_{dim}_ov', showlegend=(row == 1),
                ), row=row, col=1)

        fig.update_yaxes(
            title_text=f'{dim}  (m)', row=row, col=1,
            title_font=dict(color=ca, size=11), tickfont=dict(color=ca),
        )

    cursor_t = cursor_frame / fps
    fig.add_shape(
        type='line', xref='x', yref='paper',
        x0=cursor_t, x1=cursor_t, y0=0, y1=1,
        line=dict(color='#f39c12', width=1.5), layer='above',
    )
    fig.update_xaxes(title_text='Time  (s)', row=3, col=1)
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                     spikecolor='#888', spikethickness=1, spikedash='dot')
    fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='sans-serif', size=11, color='#333'),
        hovermode='x unified',
        legend=dict(orientation='h', x=0, y=1.04, bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ddd', borderwidth=1, font=dict(size=9)),
        margin=dict(l=70, r=30, t=60, b=50),
        uirevision=uirevision or f'traj-{name_a}-{name_b}-{primary}-{overlay}',
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eee', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#eee', zeroline=True,
                     zerolinecolor='#ddd', zerolinewidth=1)
    fig.for_each_annotation(
        lambda a: a.update(font=dict(size=12, color='#555'), x=0, xanchor='left')
    )
    return fig


def build_skeleton_figure(pose: np.ndarray, uirev: str = 'skeleton') -> go.Figure:
    xs =  pose[:, 0]
    ys = -pose[:, 2]
    zs = -pose[:, 1]
    traces = []
    for (pi, qi), color in zip(BONES, BONE_COLORS):
        traces.append(go.Scatter3d(
            x=[xs[pi], xs[qi]], y=[ys[pi], ys[qi]], z=[zs[pi], zs[qi]],
            mode='lines', line=dict(color=_rgb(color), width=6),
            showlegend=False, hoverinfo='skip',
        ))
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs, mode='markers',
        marker=dict(size=5, color=[_rgb(c) for c in JOINT_COLORS],
                    line=dict(color='white', width=1)),
        showlegend=False, hoverinfo='skip',
    ))
    radius = 0.75
    mx, my, mz = float(xs.mean()), float(ys.mean()), float(zs.mean())
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[mx-radius, mx+radius], showticklabels=False, title='', showgrid=False, zeroline=False),
            yaxis=dict(range=[my-radius, my+radius], showticklabels=False, title='', showgrid=False, zeroline=False),
            zaxis=dict(range=[mz-radius, mz+radius], showticklabels=False, title='', showgrid=False, zeroline=False),
            bgcolor='#1a1a2e', aspectmode='cube',
        ),
        paper_bgcolor='#1a1a2e',
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision=uirev,
    )
    return fig


def _empty_skeleton(uirev: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='#1a1a2e',
        scene=dict(bgcolor='#1a1a2e',
                   xaxis=dict(showticklabels=False, title='', showgrid=False),
                   yaxis=dict(showticklabels=False, title='', showgrid=False),
                   zaxis=dict(showticklabels=False, title='', showgrid=False)),
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision=uirev,
    )
    fig.add_annotation(text='Select a video', xref='paper', yref='paper',
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color='#888', size=13))
    return fig


def _start_gen_job(name: str, azim: float = 70.0) -> None:
    """Spawn a background thread to render a visualisation video for `name`."""
    from diffpose_video.scripts.visualise import visualise as render_video

    npz_path   = NPZ_DATA.get(name, {}).get('npz_path')
    video_path = VIDEO_PATHS.get(name)

    if not npz_path or not video_path:
        with _JOBS_LOCK:
            JOBS[name] = 'error: missing npz or video path'
        return

    stem = NPZ_DATA[name].get('stem', Path(name).name)
    out_dir = Path(OUTPUT_DIR) / Path(name).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f'{stem}_vis.mp4')

    def _run():
        import time as _t
        n_frames = NPZ_DATA[name]['poses_3d'].shape[0]
        with _JOBS_LOCK:
            JOBS[name] = f'rendering… (0 / {n_frames} fr)'
        try:
            t0 = _t.perf_counter()
            render_video(npz_path, video_path, out_path,
                         fps=None, start=0, end=None, azim=azim)
            elapsed = _t.perf_counter() - t0
            with _JOBS_LOCK:
                JOBS[name] = f'done ({elapsed:.0f}s) → {out_path}'
        except Exception as exc:
            with _JOBS_LOCK:
                JOBS[name] = f'error: {exc}'

    threading.Thread(target=_run, daemon=True).start()


def _frame_img(name: str | None, frame: int) -> str | None:
    if not name:
        return None
    frames = get_video_frames(name)
    if not frames:
        return None
    idx = min(frame, len(frames) - 1)
    return f'/frame/{name}/{idx}'


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def build_app(video_options: list[dict], fps: float, multi_mode: bool) -> dash.Dash:

    app = dash.Dash(__name__, title='DiffPose · Joint Explorer')

    from flask import Response

    @app.server.route('/frame/<path:name>/<int:idx>')
    def serve_frame(name: str, idx: int):
        frames = VIDEO_FRAMES.get(name)
        if not frames:
            return Response(status=404)
        idx = min(idx, len(frames) - 1)
        return Response(frames[idx], mimetype='image/jpeg')

    SIDEBAR = {
        'width': '200px', 'minWidth': '200px',
        'padding': '14px 12px',
        'backgroundColor': '#f7f7f7',
        'borderRight': '1px solid #e0e0e0',
        'display': 'flex', 'flexDirection': 'column', 'gap': '14px',
        'overflowY': 'auto',
    }
    LABEL = {
        'fontSize': '10px', 'fontWeight': '600', 'color': '#555',
        'marginBottom': '3px', 'textTransform': 'uppercase', 'letterSpacing': '0.5px',
    }
    READOUT = {
        'backgroundColor': 'white', 'border': '1px solid #e0e0e0',
        'borderRadius': '6px', 'padding': '7px 9px',
        'fontSize': '11px', 'fontFamily': 'monospace',
        'color': '#333', 'lineHeight': '1.8',
    }
    DD = {'fontSize': '12px'}
    PANEL = {
        'flex': '1', 'display': 'flex', 'flexDirection': 'column',
        'gap': '4px', 'minWidth': '0',
    }
    PANEL_LABEL = {
        'fontSize': '10px', 'fontWeight': '700', 'color': 'white',
        'padding': '2px 6px', 'borderRadius': '3px', 'alignSelf': 'flex-start',
    }

    default_a = video_options[0]['value'] if video_options else None
    default_b = video_options[1]['value'] if len(video_options) > 1 else None

    # Compute initial T for slider
    T_init = NPZ_DATA[default_a]['poses_3d'].shape[0] if default_a else 1
    interval_ms = max(50, int(1000 / fps))

    def video_panel(slot_id: str, label: str, label_color: str):
        """Returns [label_div, video_frame_or_placeholder, 3d_skeleton] in a column."""
        return html.Div(style=PANEL, children=[
            html.Div(label, style={**PANEL_LABEL, 'backgroundColor': label_color}),
            html.Div(
                style={'height': '315px', 'backgroundColor': '#0d0d1a', 'display': 'flex',
                       'alignItems': 'center', 'justifyContent': 'center',
                       'overflow': 'hidden', 'borderRadius': '6px'},
                children=[html.Img(id=f'video-frame-{slot_id}',
                                   style={'maxWidth': '100%', 'maxHeight': '315px',
                                          'objectFit': 'contain'})],
            ),
            html.Div(
                style={'height': '200px', 'borderRadius': '6px', 'overflow': 'hidden'},
                children=[dcc.Graph(id=f'skeleton-{slot_id}',
                                    config={'displayModeBar': False},
                                    style={'height': '100%'},
                                    figure=_empty_skeleton(f'skel-{slot_id}'))],
            ),
        ])

    app.layout = html.Div(
        style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh',
               'fontFamily': 'sans-serif', 'backgroundColor': 'white'},
        children=[

            # Header
            html.Div(
                style={'backgroundColor': '#2471a3', 'padding': '7px 18px',
                       'color': 'white', 'display': 'flex', 'alignItems': 'center',
                       'gap': '12px', 'flexShrink': '0'},
                children=[
                    html.H2('DiffPose · Joint Explorer',
                            style={'margin': 0, 'fontSize': '15px', 'fontWeight': '600'}),
                    html.Span(id='header-info',
                              style={'fontSize': '11px', 'opacity': '0.75'}),
                ],
            ),

            # Body
            html.Div(
                style={'display': 'flex', 'flex': '1', 'overflow': 'hidden'},
                children=[

                    # Sidebar
                    html.Div(style=SIDEBAR, children=[
                        html.Div([
                            html.Div('Video A', style={**LABEL, 'color': TRAJ_A['X']}),
                            dcc.Dropdown(id='dd-video-a', options=video_options,
                                         value=default_a, clearable=False, style=DD),
                        ]),
                        html.Div([
                            html.Div('Video B (compare)', style={**LABEL, 'color': TRAJ_B['X']}),
                            dcc.Dropdown(id='dd-video-b', options=video_options,
                                         value=default_b, clearable=True,
                                         placeholder='(none)', style=DD),
                        ]),
                        html.Hr(style={'margin': '2px 0', 'borderColor': '#ddd'}),
                        html.Div([
                            html.Div('Primary joint', style=LABEL),
                            dcc.Dropdown(id='dd-primary', options=DROPDOWN_OPTIONS,
                                         value=0, clearable=False, style=DD),
                        ]),
                        html.Div([
                            html.Div('Overlay joint', style=LABEL),
                            dcc.Dropdown(id='dd-overlay', options=OVERLAY_OPTIONS,
                                         value=-1, clearable=False, style=DD),
                        ]),
                        html.Hr(style={'margin': '2px 0', 'borderColor': '#ddd'}),
                        html.Div([
                            html.Div('Values at frame', style=LABEL),
                            html.Div(id='readout', style=READOUT, children='—'),
                        ]),
                        html.Hr(style={'margin': '2px 0', 'borderColor': '#ddd'}),
                        html.Div([
                            html.Div('Generate Video', style=LABEL),
                            html.Button('⬇ Render A', id='gen-btn-a', n_clicks=0,
                                        style={'width': '100%', 'marginBottom': '4px',
                                               'padding': '5px', 'fontSize': '11px',
                                               'cursor': 'pointer', 'borderRadius': '4px',
                                               'border': f'1px solid {TRAJ_A["X"]}',
                                               'color': TRAJ_A['X'], 'background': 'white'}),
                            html.Div(id='gen-status-a',
                                     style={'fontSize': '10px', 'color': '#888',
                                            'marginBottom': '6px', 'wordBreak': 'break-all'}),
                            html.Button('⬇ Render B', id='gen-btn-b', n_clicks=0,
                                        style={'width': '100%', 'marginBottom': '4px',
                                               'padding': '5px', 'fontSize': '11px',
                                               'cursor': 'pointer', 'borderRadius': '4px',
                                               'border': f'1px solid {TRAJ_B["X"]}',
                                               'color': TRAJ_B['X'], 'background': 'white'}),
                            html.Div(id='gen-status-b',
                                     style={'fontSize': '10px', 'color': '#888',
                                            'wordBreak': 'break-all'}),
                        ]),
                    ]),

                    # Main area
                    html.Div(
                        style={'flex': '1', 'display': 'flex', 'flexDirection': 'column',
                               'padding': '10px 12px', 'gap': '8px',
                               'overflow': 'hidden', 'minWidth': '0'},
                        children=[

                            # Video + skeleton panels (side by side if comparing)
                            html.Div(
                                style={'display': 'flex', 'gap': '10px',
                                       'flexShrink': '0'},
                                children=[
                                    video_panel('a', 'Video A', TRAJ_A['X']),
                                    video_panel('b', 'Video B', TRAJ_B['X']),
                                ],
                            ),

                            # Playback controls
                            html.Div(
                                style={'display': 'flex', 'alignItems': 'center',
                                       'gap': '10px', 'flexShrink': '0'},
                                children=[
                                    html.Button(
                                        '▶', id='play-btn',
                                        style={'fontSize': '18px', 'border': 'none',
                                               'background': 'none', 'cursor': 'pointer',
                                               'padding': '0 2px', 'color': '#333',
                                               'lineHeight': '1'},
                                        n_clicks=0,
                                    ),
                                    dcc.Slider(
                                        id='frame-slider', min=0, max=T_init - 1,
                                        step=1, value=0, marks=None,
                                        tooltip={'placement': 'bottom', 'always_visible': False},
                                        updatemode='drag', persistence=False,
                                    ),
                                    html.Span(id='frame-counter',
                                              children=f'0 / {T_init-1}',
                                              style={'fontSize': '11px', 'color': '#888',
                                                     'minWidth': '120px', 'textAlign': 'right',
                                                     'fontFamily': 'monospace', 'flexShrink': '0'}),
                                ],
                            ),

                            # Trajectory graph
                            html.Div(
                                style={'flex': '1', 'overflow': 'hidden', 'minHeight': '0'},
                                children=[dcc.Graph(
                                    id='main-graph',
                                    config={
                                        'displayModeBar': True,
                                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                        'toImageButtonOptions': {'format': 'png', 'scale': 2},
                                    },
                                    style={'height': '100%'},
                                )],
                            ),
                        ],
                    ),
                ],
            ),

            dcc.Interval(id='play-interval', interval=interval_ms,
                         disabled=True, n_intervals=0),
            dcc.Interval(id='gen-interval', interval=1500,
                         disabled=True, n_intervals=0),
        ],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────

    @callback(
        Output('frame-slider', 'max'),
        Output('header-info', 'children'),
        Input('dd-video-a', 'value'),
        Input('dd-video-b', 'value'),
    )
    def update_slider_range(name_a, name_b):
        T_a = NPZ_DATA[name_a]['poses_3d'].shape[0] if name_a else 1
        T_b = NPZ_DATA[name_b]['poses_3d'].shape[0] if name_b and name_b in NPZ_DATA else T_a
        T = max(T_a, T_b)
        info = f'{name_a or "—"}'
        if name_b:
            info += f'  vs  {name_b}'
        info += f'  ·  {T} frames  ·  {T/fps:.1f} s'
        return T - 1, info

    @callback(
        Output('frame-slider', 'value'),
        Input('play-interval', 'n_intervals'),
        State('frame-slider', 'value'),
        State('frame-slider', 'max'),
        prevent_initial_call=True,
    )
    def advance_frame(_, current, max_val):
        return (current + 1) % (max_val + 1)

    @callback(
        Output('play-interval', 'disabled'),
        Output('play-btn', 'children'),
        Input('play-btn', 'n_clicks'),
        State('play-interval', 'disabled'),
        prevent_initial_call=True,
    )
    def toggle_play(_, is_disabled):
        return (False, '⏸') if is_disabled else (True, '▶')

    @callback(
        Output('frame-counter', 'children'),
        Input('frame-slider', 'value'),
        State('frame-slider', 'max'),
    )
    def update_counter(frame, max_val):
        return f'{frame} / {max_val}  ·  {frame/fps:.2f} s'

    @callback(
        Output('video-frame-a', 'src'),
        Output('video-frame-b', 'src'),
        Input('frame-slider', 'value'),
        Input('dd-video-a', 'value'),
        Input('dd-video-b', 'value'),
    )
    def update_video_frames(frame, name_a, name_b):
        return _frame_img(name_a, frame), _frame_img(name_b, frame)

    def _patch_skeleton(slot_uirev: str, name: str | None, frame: int):
        if not name or name not in NPZ_DATA:
            return _empty_skeleton(slot_uirev)
        pose = NPZ_DATA[name]['poses_3d']
        frame = min(frame, len(pose)-1)
        p = pose[frame]
        xs, ys, zs = p[:,0], -p[:,2], -p[:,1]
        if ctx.triggered_id == 'frame-slider':
            patched = Patch()
            for i, (pi, qi) in enumerate(BONES):
                patched['data'][i]['x'] = [xs[pi], xs[qi]]
                patched['data'][i]['y'] = [ys[pi], ys[qi]]
                patched['data'][i]['z'] = [zs[pi], zs[qi]]
            j = len(BONES)
            patched['data'][j]['x'] = xs.tolist()
            patched['data'][j]['y'] = ys.tolist()
            patched['data'][j]['z'] = zs.tolist()
            return patched
        return build_skeleton_figure(p, uirev=slot_uirev)

    @callback(
        Output('skeleton-a', 'figure'),
        Input('frame-slider', 'value'),
        Input('dd-video-a', 'value'),
    )
    def update_skeleton_a(frame, name):
        return _patch_skeleton('skel-a', name, frame)

    @callback(
        Output('skeleton-b', 'figure'),
        Input('frame-slider', 'value'),
        Input('dd-video-b', 'value'),
    )
    def update_skeleton_b(frame, name):
        return _patch_skeleton('skel-b', name, frame)

    @callback(
        Output('main-graph', 'figure'),
        Input('dd-video-a', 'value'),
        Input('dd-video-b', 'value'),
        Input('dd-primary', 'value'),
        Input('dd-overlay', 'value'),
        Input('frame-slider', 'value'),
    )
    def update_figure(name_a, name_b, primary, overlay_val, frame):
        if not name_a:
            return go.Figure()
        overlay = None if overlay_val == -1 else overlay_val
        t = frame / fps
        if ctx.triggered_id == 'frame-slider':
            patched = Patch()
            patched['layout']['shapes'] = [dict(
                type='line', xref='x', yref='paper',
                x0=t, x1=t, y0=0, y1=1,
                line=dict(color='#f39c12', width=1.5), layer='above',
            )]
            return patched
        uirev = f'traj-{name_a}-{name_b}-{primary}-{overlay}'
        return build_trajectory_figure(name_a, name_b, primary, overlay, fps, frame, uirevision=uirev)

    @callback(
        Output('readout', 'children'),
        Input('frame-slider', 'value'),
        Input('dd-video-a', 'value'),
        Input('dd-video-b', 'value'),
        Input('dd-primary', 'value'),
        Input('dd-overlay', 'value'),
    )
    def update_readout(frame, name_a, name_b, primary, overlay_val):
        lines = [html.Span(f'Frame {frame}  ·  {frame/fps:.2f} s',
                           style={'color': '#888', 'fontSize': '10px'})]

        def joint_block(poses, jidx, label, colors, fade=False):
            v = poses[min(frame, len(poses)-1), jidx]
            op = {'opacity': '0.65'} if fade else {}
            return [
                html.Br(),
                html.Span(label, style={'fontWeight': '600', **op}), html.Br(),
                html.Span(f'  X = {v[0]:+.4f}', style={'color': colors["X"], **op}), html.Br(),
                html.Span(f'  Y = {v[1]:+.4f}', style={'color': colors["Y"], **op}), html.Br(),
                html.Span(f'  Z = {v[2]:+.4f}', style={'color': colors["Z"], **op}),
            ]

        if name_a and name_a in NPZ_DATA:
            pa = NPZ_DATA[name_a]['poses_3d']
            lines += joint_block(pa, primary, f'A · {JOINT_NAMES[primary]}', TRAJ_A)
            if overlay_val != -1:
                lines += joint_block(pa, overlay_val, f'A · {JOINT_NAMES[overlay_val]} (ov)', TRAJ_A, fade=True)

        if name_b and name_b in NPZ_DATA:
            pb = NPZ_DATA[name_b]['poses_3d']
            lines += [html.Br()]
            lines += joint_block(pb, primary, f'B · {JOINT_NAMES[primary]}', TRAJ_B)
            if overlay_val != -1:
                lines += joint_block(pb, overlay_val, f'B · {JOINT_NAMES[overlay_val]} (ov)', TRAJ_B, fade=True)

        return lines

    # ── Video generation callbacks ─────────────────────────────────────────

    @callback(
        Output('gen-status-a', 'children'),
        Output('gen-status-b', 'children'),
        Output('gen-interval', 'disabled'),
        Input('gen-btn-a', 'n_clicks'),
        Input('gen-btn-b', 'n_clicks'),
        State('dd-video-a', 'value'),
        State('dd-video-b', 'value'),
        prevent_initial_call=True,
    )
    def on_gen_click(n_a, n_b, name_a, name_b):
        triggered = ctx.triggered_id
        if triggered == 'gen-btn-a' and name_a:
            _start_gen_job(name_a)
        elif triggered == 'gen-btn-b' and name_b:
            _start_gen_job(name_b)
        sa = JOBS.get(name_a, '') if name_a else ''
        sb = JOBS.get(name_b, '') if name_b else ''
        return sa, sb, False   # enable polling interval

    @callback(
        Output('gen-status-a', 'children', allow_duplicate=True),
        Output('gen-status-b', 'children', allow_duplicate=True),
        Output('gen-interval', 'disabled', allow_duplicate=True),
        Input('gen-interval', 'n_intervals'),
        State('dd-video-a', 'value'),
        State('dd-video-b', 'value'),
        prevent_initial_call=True,
    )
    def poll_gen_status(_, name_a, name_b):
        sa = JOBS.get(name_a, '') if name_a else ''
        sb = JOBS.get(name_b, '') if name_b else ''
        still_running = any(v == 'rendering…' for v in JOBS.values())
        return sa, sb, not still_running

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Interactive joint explorer (Dash)')
    parser.add_argument('--config', default=None,
                        help='Path to a TOML config file (all other args become optional).')
    # Single-video mode (backward-compatible)
    parser.add_argument('--npz',   default=None, help='Path to a single .npz from infer.py')
    parser.add_argument('--video', default=None, help='Path to original video file (optional, single-video mode)')
    # Multi-video mode
    parser.add_argument('--results_dir', default=None,
                        help='Directory of .npz files to browse and compare (recursive scan)')
    parser.add_argument('--videos_dir', default=None,
                        help='Default directory for original videos (matched by stem name)')
    parser.add_argument('--videos_map', nargs='+', default=None, metavar='PREFIX:PATH',
                        help='Map result key prefixes to video directories. '
                             'E.g. --videos_map Cam1:/path/to/Cam1/InputMedia')
    # Shared
    parser.add_argument('--fps',        type=float, default=None, help='Video FPS (default: 30)')
    parser.add_argument('--port',       type=int,   default=None, help='Local port (default: 8050)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for generated videos (default: visualisations/)')
    args = parser.parse_args()
    _apply_config(args)
    return args


def _apply_config(args) -> None:
    from diffpose_video.common.config_loader import apply_explore_config
    apply_explore_config(args)


def main():
    global VIDEOS_DIR, VIDEOS_MAP, OUTPUT_DIR

    args = parse_args()

    if not args.npz and not args.results_dir:
        raise SystemExit('Provide either --npz (single video) or --results_dir (multi-video mode).')

    if args.videos_dir:
        VIDEOS_DIR = Path(args.videos_dir)
    for mapping in args.videos_map:
        if ':' not in mapping:
            raise SystemExit(f'--videos_map entry must be PREFIX:PATH, got: {mapping!r}')
        prefix, path = mapping.split(':', 1)
        VIDEOS_MAP[prefix] = Path(path)
        print(f'  Video map: {prefix!r} → {path}')
    OUTPUT_DIR = args.output_dir

    multi_mode = bool(args.results_dir)

    if args.results_dir:
        # Scan directory recursively for .npz files
        results_path = Path(args.results_dir)
        npz_files = sorted(results_path.rglob('*.npz'))
        if not npz_files:
            raise SystemExit(f'No .npz files found in {args.results_dir}')
        print(f'Found {len(npz_files)} result file(s).')
        for p in npz_files:
            key = str(p.parent.relative_to(results_path))  # e.g. "Cam1/B01" or "B01"
            data = np.load(p)
            NPZ_DATA[key] = {
                'poses_3d':    data['poses_3d'],
                'keypoints_2d': data['keypoints_2d'] if 'keypoints_2d' in data else None,
                'npz_path':    str(p),
                'stem':        p.stem,
            }
            print(f'  Loaded {key}  ({data["poses_3d"].shape[0]} frames)')
    else:
        # Single .npz
        p = Path(args.npz)
        key = p.stem
        data = np.load(p)
        NPZ_DATA[key] = {
            'poses_3d':    data['poses_3d'],
            'keypoints_2d': data['keypoints_2d'] if 'keypoints_2d' in data else None,
            'npz_path':    str(p),
            'stem':        p.stem,
        }
        if args.video:
            kps_2d = NPZ_DATA[key]['keypoints_2d']
            print('Extracting video frames ...')
            VIDEO_FRAMES[key] = extract_video_frames(args.video, kps_2d=kps_2d)
            print(f'Extracted {len(VIDEO_FRAMES[key])} frames')
        multi_mode = False

    video_options = [{'label': name, 'value': name} for name in sorted(NPZ_DATA)]

    print(f'Open http://localhost:{args.port} in your browser')
    app = build_app(video_options, args.fps, multi_mode)
    app.run(debug=False, port=args.port)


if __name__ == '__main__':
    main()
