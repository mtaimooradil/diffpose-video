"""
TOML config loader shared by all DiffPose-Video CLI scripts.

Usage in a script:
    args = parse_args()               # all optional args default to None
    apply_config(args)                # fills gaps from --config TOML, then hardcoded defaults
"""


def load_toml(path: str) -> dict:
    import tomllib
    with open(path, 'rb') as f:
        return tomllib.load(f)


def merge(args, cfg: dict, defaults: dict) -> None:
    """
    Fill None/missing values in args namespace.
    Priority: CLI args (non-None) > cfg > defaults.
    """
    combined = {**defaults, **cfg}
    for key, val in combined.items():
        if key == 'videos':
            continue  # handled separately by each script
        current = getattr(args, key, None)
        if current is None or current == [] and val:
            setattr(args, key, val)


def apply_explore_config(args) -> None:
    cfg = load_toml(args.config) if args.config else {}

    videos_cfg = cfg.pop('videos', {})
    videos_dir = videos_cfg.pop('default', None)
    videos_map_from_cfg = [f'{k}:{v}' for k, v in videos_cfg.items()]

    merge(args, cfg, defaults={
        'npz':          None,
        'video':        None,
        'results_dir':  None,
        'videos_dir':   videos_dir,
        'videos_map':   videos_map_from_cfg or [],
        'fps':          30.0,
        'port':         8050,
        'output_dir':   'visualisations',
    })

    if args.videos_map is None:
        args.videos_map = []
