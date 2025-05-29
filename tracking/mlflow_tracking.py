import mlflow
import os

mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

import mlflow.sklearn
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def log_model_metrics(model, model_name, y_true, y_pred, y_pred_prob):
    """Log model parameters, metrics, and artifacts to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Log model
        mlflow.sklearn.log_model(model, model_name)

        # Log metrics
        mlflow.log_metric("roc_auc", roc_auc_score(y_true, y_pred_prob))
        mlflow.log_metric("f1_score", f1_score(y_true, y_pred))
        mlflow.log_metric("precision", precision_score(y_true, y_pred))
        mlflow.log_metric("recall", recall_score(y_true, y_pred))

        print(f"✅ MLflow run logged for {model_name}")
