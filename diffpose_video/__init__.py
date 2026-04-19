__version__ = "0.2.1"

from pathlib import Path


def configs_dir() -> Path:
    """Return the path to the bundled config directory."""
    return Path(__file__).parent / "configs"
