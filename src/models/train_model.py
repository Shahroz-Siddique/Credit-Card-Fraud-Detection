
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

def train_models(data_path: str, output_path: str):
    df = pd.read_csv(data_path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    os.makedirs(output_path, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\nModel: {name}")
        print(classification_report(y_test, preds, digits=4))
        print("ROC-AUC Score:", roc_auc_score(y_test, proba))

        joblib.dump(model, os.path.join(output_path, f"{name}.pkl"))

# Example usage
if __name__ == "__main__":
    train_models("data/processed/creditcard_cleaned.csv", "models/")
