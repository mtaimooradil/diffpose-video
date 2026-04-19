"""
Interactive joint trajectory explorer — Plotly Dash edition.

Displays video, animated 3D skeleton, and X/Y/Z trajectory graphs all linked
to a single frame slider with play/pause controls.

Usage
-----
    python explore.py --npz results/IMG_0076.npz --fps 30
    python explore.py --npz results/IMG_0076.npz --video path/to/video.mp4 --fps 30
"""

import argparse
import base64

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
    (0, 1), (1, 2), (2, 3),        # right leg
    (0, 4), (4, 5), (5, 6),        # left leg
    (0, 7), (7, 8),                 # spine
    (8, 9), (9, 10),                # neck → head
    (8, 11), (11, 12), (12, 13),   # left arm
    (8, 14), (14, 15), (15, 16),   # right arm
]

_R = (0.2, 0.4, 0.8)   # right limbs – blue
_L = (0.8, 0.2, 0.2)   # left limbs  – red
_C = (0.2, 0.7, 0.3)   # centre      – green

BONE_COLORS = [
    _R, _R, _R,    # right leg
    _L, _L, _L,    # left leg
    _C, _C,        # spine
    _C, _C,        # neck/head
    _L, _L, _L,    # left arm
    _R, _R, _R,    # right arm
]

JOINT_COLORS = [
    _C,
    _R, _R, _R,
    _L, _L, _L,
    _C, _C,
    _C, _C,
    _L, _L, _L,
    _R, _R, _R,
]

TRAJ_COLORS = {'X': '#e74c3c', 'Y': '#27ae60', 'Z': '#2471a3'}
DROPDOWN_OPTIONS = [{'label': n, 'value': i} for i, n in enumerate(JOINT_NAMES)]
OVERLAY_OPTIONS  = [{'label': '(none)', 'value': -1}] + DROPDOWN_OPTIONS


def _rgb(c):
    return f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'


# ---------------------------------------------------------------------------
# Video frame extraction (with optional 2D skeleton overlay)
# ---------------------------------------------------------------------------

def _to_bgr(rgb: tuple) -> tuple:
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


def draw_2d_skeleton(frame: np.ndarray, kps: np.ndarray, conf_thr: float = 0.3) -> np.ndarray:
    """Draw H36M 17-joint 2D skeleton on a BGR frame (pixel-coord keypoints)."""
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(h, w) / 1000.0
    for (i, j), color in zip(BONES, BONE_COLORS):
        if kps[i, 2] < conf_thr or kps[j, 2] < conf_thr:
            continue
        pt1 = (int(kps[i, 0]), int(kps[i, 1]))
        pt2 = (int(kps[j, 0]), int(kps[j, 1]))
        cv2.line(out, pt1, pt2, _to_bgr(color),
                 thickness=max(2, int(3 * scale)), lineType=cv2.LINE_AA)
    for idx, (x, y, c) in enumerate(kps):
        if c < conf_thr:
            continue
        r = max(4, int(5 * scale))
        cv2.circle(out, (int(x), int(y)), r, _to_bgr(JOINT_COLORS[idx]), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), r, (255, 255, 255),
                   max(1, int(1.5 * scale)), cv2.LINE_AA)
    return out


def extract_video_frames(
    video_path: str,
    kps_2d: np.ndarray | None = None,
    max_width: int = 640,
) -> list[str]:
    """
    Read all video frames and return them as base64-encoded JPEG strings.
    If kps_2d (T, 17, 3) is provided, the 2D skeleton is baked into each frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')

    frames_b64 = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if kps_2d is not None and idx < len(kps_2d):
            frame = draw_2d_skeleton(frame, kps_2d[idx])
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf.tobytes()).decode()
        frames_b64.append(f'data:image/jpeg;base64,{b64}')
        idx += 1

    cap.release()
    return frames_b64


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_trajectory_figure(
    poses_3d: np.ndarray,
    time_axis: np.ndarray,
    primary: int,
    overlay: int | None,
    cursor_time: float = 0.0,
) -> go.Figure:
    """Build 3-subplot trajectory figure (X / Y / Z) with a frame cursor."""
    components = ['X', 'Y', 'Z']
    ylabels    = ['X  (m)', 'Y  (m)', 'Z  (m)']

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f'<b>{c}</b>' for c in components],
    )

    for row, (dim, comp, ylabel) in enumerate(
        zip(range(3), components, ylabels), start=1
    ):
        color = TRAJ_COLORS[comp]

        fig.add_trace(
            go.Scatter(
                x=time_axis, y=poses_3d[:, primary, dim],
                mode='lines',
                name=f'{JOINT_NAMES[primary]} — {comp}',
                line=dict(color=color, width=2),
                legendgroup=comp, showlegend=(row == 1),
                hovertemplate=(
                    f'<b>{JOINT_NAMES[primary]}</b><br>'
                    f't=%{{x:.2f}} s<br>{comp}=%{{y:.4f}} m<extra></extra>'
                ),
            ),
            row=row, col=1,
        )

        if overlay is not None:
            fig.add_trace(
                go.Scatter(
                    x=time_axis, y=poses_3d[:, overlay, dim],
                    mode='lines',
                    name=f'{JOINT_NAMES[overlay]} — {comp} (overlay)',
                    line=dict(color=color, width=1.5, dash='dash'),
                    opacity=0.55,
                    legendgroup=f'{comp}_ov', showlegend=(row == 1),
                    hovertemplate=(
                        f'<b>{JOINT_NAMES[overlay]}</b><br>'
                        f't=%{{x:.2f}} s<br>{comp}=%{{y:.4f}} m<extra></extra>'
                    ),
                ),
                row=row, col=1,
            )

        fig.update_yaxes(
            title_text=ylabel, row=row, col=1,
            title_font=dict(color=color, size=11),
            tickfont=dict(color=color),
        )

    fig.update_xaxes(title_text='Time  (s)', row=3, col=1)
    fig.update_xaxes(
        showspikes=True, spikemode='across', spikesnap='cursor',
        spikecolor='#888', spikethickness=1, spikedash='dot',
    )

    # Vertical cursor line spanning all three subplots
    fig.add_shape(
        type='line', xref='x', yref='paper',
        x0=cursor_time, x1=cursor_time, y0=0, y1=1,
        line=dict(color='#f39c12', width=1.5),
        layer='above',
    )

    fig.update_layout(
        height=None,
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='sans-serif', size=11, color='#333'),
        hovermode='x unified',
        legend=dict(
            orientation='h', x=0, y=1.04,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#ddd', borderwidth=1, font=dict(size=10),
        ),
        margin=dict(l=70, r=30, t=60, b=50),
        uirevision='traj',   # preserve zoom/pan across updates
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eee', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#eee', zeroline=True,
                     zerolinecolor='#ddd', zerolinewidth=1)
    fig.for_each_annotation(
        lambda a: a.update(font=dict(size=12, color='#555'), x=0, xanchor='left')
    )
    return fig


def build_skeleton_figure(pose: np.ndarray) -> go.Figure:
    """Build a 3D Plotly figure for a single H36M pose (17, 3)."""
    xs =  pose[:, 0]
    ys = -pose[:, 2]   # H36M y-up → display z-up
    zs = -pose[:, 1]

    traces = []
    for (pi, qi), color in zip(BONES, BONE_COLORS):
        traces.append(go.Scatter3d(
            x=[xs[pi], xs[qi]],
            y=[ys[pi], ys[qi]],
            z=[zs[pi], zs[qi]],
            mode='lines',
            line=dict(color=_rgb(color), width=6),
            showlegend=False, hoverinfo='skip',
        ))

    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=5, color=[_rgb(c) for c in JOINT_COLORS],
                    line=dict(color='white', width=1)),
        showlegend=False, hoverinfo='skip',
    ))

    radius = 0.75
    mx, my, mz = float(xs.mean()), float(ys.mean()), float(zs.mean())

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[mx - radius, mx + radius],
                       showticklabels=False, title='', showgrid=False, zeroline=False),
            yaxis=dict(range=[my - radius, my + radius],
                       showticklabels=False, title='', showgrid=False, zeroline=False),
            zaxis=dict(range=[mz - radius, mz + radius],
                       showticklabels=False, title='', showgrid=False, zeroline=False),
            bgcolor='#1a1a2e',
            aspectmode='cube',
        ),
        paper_bgcolor='#1a1a2e',
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision='skeleton',   # preserve camera rotation across updates
    )
    return fig


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def build_app(
    poses_3d: np.ndarray,
    fps: float,
    frames_b64: list[str] | None,
) -> dash.Dash:
    T         = poses_3d.shape[0]
    time_axis = np.arange(T) / fps
    has_video = bool(frames_b64)

    app = dash.Dash(__name__, title='DiffPose · Joint Explorer')

    # ── Styles ───────────────────────────────────────────────────────────
    SIDEBAR = {
        'width': '190px', 'minWidth': '190px',
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
        'color': '#333', 'lineHeight': '1.8', 'minHeight': '65px',
    }
    DD = {'fontSize': '12px'}

    interval_ms = max(50, int(1000 / fps))

    # ── Video panel ──────────────────────────────────────────────────────
    if has_video:
        video_content = html.Img(
            id='video-frame',
            style={'maxWidth': '100%', 'maxHeight': '100%', 'objectFit': 'contain'},
        )
    else:
        video_content = html.Span(
            'No video provided\n(pass --video to enable)',
            style={'color': '#666', 'fontSize': '12px',
                   'whiteSpace': 'pre-line', 'textAlign': 'center'},
        )

    # ── Layout ───────────────────────────────────────────────────────────
    app.layout = html.Div(
        style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh',
               'fontFamily': 'sans-serif', 'backgroundColor': 'white'},
        children=[

            # Header bar
            html.Div(
                style={'backgroundColor': '#2471a3', 'padding': '7px 18px',
                       'color': 'white', 'display': 'flex',
                       'alignItems': 'center', 'gap': '12px', 'flexShrink': '0'},
                children=[
                    html.H2('DiffPose · Joint Explorer',
                            style={'margin': 0, 'fontSize': '15px', 'fontWeight': '600'}),
                    html.Span(f'{T} frames  ·  {T/fps:.1f} s  ·  17 joints',
                              style={'fontSize': '11px', 'opacity': '0.75'}),
                ],
            ),

            # Body
            html.Div(
                style={'display': 'flex', 'flex': '1', 'overflow': 'hidden'},
                children=[

                    # ── Sidebar ─────────────────────────────────────────
                    html.Div(style=SIDEBAR, children=[
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
                        html.Div([
                            html.Div('Values at frame', style=LABEL),
                            html.Div(id='readout', style=READOUT, children='—'),
                        ]),
                    ]),

                    # ── Main area ────────────────────────────────────────
                    html.Div(
                        style={'flex': '1', 'display': 'flex', 'flexDirection': 'column',
                               'padding': '10px 12px', 'gap': '8px', 'overflow': 'hidden',
                               'minWidth': '0'},
                        children=[

                            # Top row: video + 3D skeleton
                            html.Div(
                                style={'display': 'flex', 'gap': '10px',
                                       'height': '270px', 'flexShrink': '0'},
                                children=[
                                    # Video
                                    html.Div(
                                        style={'flex': '1', 'backgroundColor': '#0d0d1a',
                                               'display': 'flex', 'alignItems': 'center',
                                               'justifyContent': 'center', 'overflow': 'hidden',
                                               'borderRadius': '6px'},
                                        children=[video_content],
                                    ),
                                    # 3D skeleton
                                    html.Div(
                                        style={'flex': '1', 'overflow': 'hidden',
                                               'borderRadius': '6px'},
                                        children=[
                                            dcc.Graph(
                                                id='skeleton-3d',
                                                config={'displayModeBar': False},
                                                style={'height': '100%'},
                                            ),
                                        ],
                                    ),
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
                                        id='frame-slider',
                                        min=0, max=T - 1, step=1, value=0,
                                        marks=None,
                                        tooltip={'placement': 'bottom',
                                                 'always_visible': False},
                                        updatemode='drag',
                                        persistence=False,
                                    ),
                                    html.Span(
                                        id='frame-counter',
                                        children=f'0 / {T-1}',
                                        style={'fontSize': '11px', 'color': '#888',
                                               'minWidth': '100px', 'textAlign': 'right',
                                               'fontFamily': 'monospace', 'flexShrink': '0'},
                                    ),
                                ],
                            ),

                            # Trajectory graph
                            html.Div(
                                style={'flex': '1', 'overflow': 'hidden', 'minHeight': '0'},
                                children=[
                                    dcc.Graph(
                                        id='main-graph',
                                        config={
                                            'displayModeBar': True,
                                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                            'toImageButtonOptions': {'format': 'png', 'scale': 2},
                                        },
                                        style={'height': '100%'},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # Playback interval (hidden)
            dcc.Interval(
                id='play-interval',
                interval=interval_ms,
                disabled=True,
                n_intervals=0,
            ),
        ],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────

    @callback(
        Output('frame-slider', 'value'),
        Input('play-interval', 'n_intervals'),
        State('frame-slider', 'value'),
        prevent_initial_call=True,
    )
    def advance_frame(_, current):
        return (current + 1) % T

    @callback(
        Output('play-interval', 'disabled'),
        Output('play-btn', 'children'),
        Input('play-btn', 'n_clicks'),
        State('play-interval', 'disabled'),
        prevent_initial_call=True,
    )
    def toggle_play(_, is_disabled):
        if is_disabled:
            return False, '⏸'
        return True, '▶'

    @callback(
        Output('frame-counter', 'children'),
        Input('frame-slider', 'value'),
    )
    def update_counter(frame):
        return f'{frame} / {T - 1}  ·  {frame / fps:.2f} s'

    if has_video:
        @callback(
            Output('video-frame', 'src'),
            Input('frame-slider', 'value'),
        )
        def update_video(frame):
            return frames_b64[min(frame, len(frames_b64) - 1)]

    @callback(
        Output('skeleton-3d', 'figure'),
        Input('frame-slider', 'value'),
    )
    def update_skeleton(frame):
        pose = poses_3d[frame]
        # Patch-update bone/joint data without rebuilding the figure
        # (preserves camera orientation set by the user)
        if ctx.triggered_id == 'frame-slider':
            xs =  pose[:, 0]
            ys = -pose[:, 2]
            zs = -pose[:, 1]
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
        return build_skeleton_figure(pose)

    @callback(
        Output('main-graph', 'figure'),
        Input('dd-primary', 'value'),
        Input('dd-overlay', 'value'),
        Input('frame-slider', 'value'),
    )
    def update_figure(primary, overlay_val, frame):
        overlay = None if overlay_val == -1 else overlay_val
        t = frame / fps
        # Patch-update only the cursor shape during playback
        if ctx.triggered_id == 'frame-slider':
            patched = Patch()
            patched['layout']['shapes'] = [dict(
                type='line', xref='x', yref='paper',
                x0=t, x1=t, y0=0, y1=1,
                line=dict(color='#f39c12', width=1.5),
                layer='above',
            )]
            return patched
        return build_trajectory_figure(poses_3d, time_axis, primary, overlay,
                                       cursor_time=t)

    @callback(
        Output('readout', 'children'),
        Input('frame-slider', 'value'),
        Input('dd-primary', 'value'),
        Input('dd-overlay', 'value'),
    )
    def update_readout(frame, primary, overlay_val):
        v = poses_3d[frame, primary]
        lines = [
            html.Span(f'Frame {frame}  ·  {frame/fps:.2f} s',
                      style={'color': '#888', 'fontSize': '10px'}),
            html.Br(),
            html.Span(JOINT_NAMES[primary], style={'fontWeight': '600'}),
            html.Br(),
            html.Span(f'  X = {v[0]:+.4f}', style={'color': TRAJ_COLORS["X"]}),
            html.Br(),
            html.Span(f'  Y = {v[1]:+.4f}', style={'color': TRAJ_COLORS["Y"]}),
            html.Br(),
            html.Span(f'  Z = {v[2]:+.4f}', style={'color': TRAJ_COLORS["Z"]}),
        ]
        if overlay_val != -1:
            ov = poses_3d[frame, overlay_val]
            lines += [
                html.Br(), html.Br(),
                html.Span(JOINT_NAMES[overlay_val],
                          style={'fontWeight': '600', 'opacity': '0.7'}),
                html.Br(),
                html.Span(f'  X = {ov[0]:+.4f}',
                          style={'color': TRAJ_COLORS["X"], 'opacity': '0.7'}),
                html.Br(),
                html.Span(f'  Y = {ov[1]:+.4f}',
                          style={'color': TRAJ_COLORS["Y"], 'opacity': '0.7'}),
                html.Br(),
                html.Span(f'  Z = {ov[2]:+.4f}',
                          style={'color': TRAJ_COLORS["Z"], 'opacity': '0.7'}),
            ]
        return lines

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Interactive joint explorer (Dash)')
    parser.add_argument('--npz',   required=True, help='Path to .npz from infer.py')
    parser.add_argument('--video', default=None,  help='Path to original video file (optional)')
    parser.add_argument('--fps',   type=float, default=30.0,
                        help='Source video FPS (default: 30)')
    parser.add_argument('--port',  type=int, default=8050,
                        help='Local port (default: 8050)')
    return parser.parse_args()


def main():
    args     = parse_args()
    data     = np.load(args.npz)
    poses_3d = data['poses_3d']   # (T, 17, 3)
    T        = poses_3d.shape[0]
    print(f'Loaded {T} frames  ·  17 joints  ·  3 components')

    frames_b64 = None
    if args.video:
        kps_2d = data['keypoints_2d'] if 'keypoints_2d' in data else None
        print('Extracting video frames with skeleton overlay ...')
        frames_b64 = extract_video_frames(args.video, kps_2d=kps_2d)
        print(f'Extracted {len(frames_b64)} frames')

    print(f'Open http://localhost:{args.port} in your browser')
    app = build_app(poses_3d, args.fps, frames_b64)
    app.run(debug=False, port=args.port)


if __name__ == '__main__':
    main()
