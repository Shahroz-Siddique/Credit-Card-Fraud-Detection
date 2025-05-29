import joblib
from src.utils.config_loader import load_config

def load_trained_model():
    config = load_config()
    model_path = config["model"]["output_dir"] + config["model"]["ensemble_model_name"]
    model = joblib.load(model_path)
    return model
