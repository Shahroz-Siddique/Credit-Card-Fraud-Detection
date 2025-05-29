import pandas as pd
import os
import joblib

base_dir = os.path.dirname(os.path.dirname(__file__))  # Goes up one level from /app/
amount_scaler = joblib.load(os.path.join(base_dir, "models", "scaler_amount.pkl"))
time_scaler = joblib.load(os.path.join(base_dir, "models", "scaler_time.pkl"))


FEATURE_COLUMNS = [
    "scaled_time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "scaled_amount"
]


def preprocess_input(input_dict):
    # Convert dict to DataFrame
    df = pd.DataFrame([input_dict])
    return df
