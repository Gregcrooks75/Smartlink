from pathlib import Path
import yaml
import os

def load_paths():
    config_path = Path(__file__).resolve().parents[3] / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    def convert_to_paths(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = convert_to_paths(v)
            elif isinstance(v, str):
                result[k] = Path(v)
            else:
                result[k] = v
        return result

    return convert_to_paths(config)

# Example usage
if __name__ == "__main__":
    paths = load_paths()
    print(paths["base_dir"])
