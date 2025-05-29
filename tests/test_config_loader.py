import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_loader import load_config

def test_load_config_keys():
    config = load_config()
    assert "data" in config
    assert "model" in config
    assert "mlflow" in config

def test_paths_exist():
    config = load_config()
    assert "processed_path" in config["data"]
    assert config["model"]["output_dir"].endswith("/")
