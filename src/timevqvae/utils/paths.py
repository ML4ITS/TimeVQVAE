from pathlib import Path


def get_root_dir() -> Path:
    """Return repository root from this utils package path."""
    # src/timevqvae/utils/paths.py -> repository root
    return Path(__file__).resolve().parents[3]
