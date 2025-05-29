import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import patch, MagicMock
from tracking.mlflow_tracking import log_model_metrics
from sklearn.linear_model import LogisticRegression
import numpy as np

@pytest.fixture
def dummy_data():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_pred_prob = np.array([0.1, 0.9, 0.2, 0.4])
    model = LogisticRegression()
    return model, "test_model", y_true, y_pred, y_pred_prob

@patch("mlflow.start_run")
@patch("mlflow.sklearn.log_model")
@patch("mlflow.log_metric")
def test_log_model_metrics(mock_log_metric, mock_log_model, mock_start_run, dummy_data):
    model, name, y_true, y_pred, y_pred_prob = dummy_data

    run_mock = MagicMock()
    mock_start_run.return_value.__enter__.return_value = run_mock

    log_model_metrics(model, name, y_true, y_pred, y_pred_prob)

    assert mock_log_model.called
    assert mock_log_metric.call_count == 4
    assert mock_start_run.called
