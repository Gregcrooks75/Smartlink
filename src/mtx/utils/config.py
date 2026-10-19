from pathlib import Path
import yaml

def load_config(path: str | None = None) -> dict:
    """
    Load YAML config as a dictionary.
    If no path is given, default to /config/config.yaml
    """
    if path is None:
        # src/mtx/utils/config.py → go up 2 levels to project root
        path = Path(__file__).resolve().parents[3] / "config" / "config.yaml"

    with open(path, "r") as f:
        return yaml.safe_load(f)