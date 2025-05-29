# src/utils/config_loader.py

import yaml

def load_config(config_path="config/config.yaml"):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
