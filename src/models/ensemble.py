import sys
import os
sys.path.append(os.path.abspath(".."))  
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from tracking.mlflow_tracking import log_model_metrics
from src.utils.config_loader import load_config

config = load_config()
processed_path = config["data"]["processed_path"]
model_output = config["model"]["output_dir"]
model_name = config["model"]["ensemble_model_name"]

# Use:
# train_ensemble_with_smote(processed_path, model_output)
# Save model as: os.path.join(model_output, model_name)

def train_ensemble_with_smote(data_path: str, output_path: str):
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # SMOTE resampling
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    # Define base models
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
        voting="soft"
    )

    # Fit model
    ensemble.fit(X_resampled, y_resampled)

    # Predict & evaluate
    preds = ensemble.predict(X_test)
    proba = ensemble.predict_proba(X_test)[:, 1]

    print("\n✅ Ensemble with SMOTE - Evaluation")
    print(classification_report(y_test, preds, digits=4))
    print("ROC-AUC Score:", roc_auc_score(y_test, proba))

    log_model_metrics(ensemble, "Ensemble_SMOTE", y_test, preds, proba)
    # Save model
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(ensemble, os.path.join(output_path, "ensemble_smote.pkl"))
    print(f"📁 Model saved to {output_path}/ensemble_smote.pkl")

# Example usage
if __name__ == "__main__":
    train_ensemble_with_smote("data/processed/creditcard_cleaned.csv", "models")
