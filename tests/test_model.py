import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
import pytest
from sklearn.base import ClassifierMixin

from src.models.train_model import train_models
from src.models.ensemble import train_ensemble_with_smote

TEST_DATA_PATH = "data/processed/creditcard_cleaned.csv"
TEST_MODEL_DIR = "models/test_models"

@pytest.fixture(scope="module", autouse=True)
def setup_test_models():
    # Create test models
    train_models(TEST_DATA_PATH, TEST_MODEL_DIR)
    train_ensemble_with_smote(TEST_DATA_PATH, TEST_MODEL_DIR)
    yield
    # Cleanup (optional): remove test models
    # import shutil; shutil.rmtree(TEST_MODEL_DIR)

def test_logistic_regression_exists():
    path = os.path.join(TEST_MODEL_DIR, "logistic_regression.pkl")
    assert os.path.exists(path), "Logistic Regression model not saved."

def test_random_forest_exists():
    path = os.path.join(TEST_MODEL_DIR, "random_forest.pkl")
    assert os.path.exists(path), "Random Forest model not saved."

def test_xgboost_exists():
    path = os.path.join(TEST_MODEL_DIR, "xgboost.pkl")
    assert os.path.exists(path), "XGBoost model not saved."

def test_ensemble_model_exists():
    path = os.path.join(TEST_MODEL_DIR, "ensemble_smote.pkl")
    assert os.path.exists(path), "Ensemble model not saved."

def test_ensemble_model_type():
    model = joblib.load(os.path.join(TEST_MODEL_DIR, "ensemble_smote.pkl"))
    assert isinstance(model, ClassifierMixin), "Ensemble model is not a classifier."

def test_model_prediction_shape():
    model = joblib.load(os.path.join(TEST_MODEL_DIR, "logistic_regression.pkl"))
    df = pd.read_csv(TEST_DATA_PATH)
    X = df.drop("Class", axis=1)
    preds = model.predict(X[:10])
    assert len(preds) == 10, "Prediction output shape mismatch."
