import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import math
import numpy as np
import pandas as pd
from bikeshare_model.predict import make_prediction
import pytest
from bikeshare_model.config.core import config
from bikeshare_model import __version__ as _version

def test_pipeline_trained():
    # Given
    from bikeshare_model import __version__ as _version
    
    # Constants
    pipeline_file = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    
    # Verify trained model exists with full path
    assert pipeline_file == f"bikeshare__model_output_v{_version}.pkl"

def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = 139
    expected_no_predictions = 2

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert predictions is not None, "Predictions should not be None"
    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array"
    assert len(predictions) == expected_no_predictions, f"Expected {expected_no_predictions} predictions"
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions), "Each prediction should be an integer"
    assert math.isclose(predictions[0], expected_first_prediction_value, rel_tol=1e-2), \
        f"First prediction should be close to {expected_first_prediction_value}"
    assert result.get("errors") is None, "Errors should be None"

def test_make_prediction_with_invalid_data():
    # Given
    invalid_data = {
        "dteday": ["2012-11-05", "2012-11-05"],  # Valid dates
        "season": [1, "not"],  # Invalid season value
        "hr": [6, 25],  # Invalid hour
        "holiday": [0, 1],
        "weekday": [1, 2],
        "workingday": [0, 1],
        "weathersit": [1, 5],  # Invalid weathersit
        "temp": [38.0, 15.5],
        "atemp": [35.5, 14.0],
        "hum": [0.8, 0.4],
        "windspeed": [0.3, 0.2],
        "casual": [30, 20],
        "registered": [150, 120],
    }

    # When
    result = make_prediction(input_data=invalid_data)

    # Then
    assert result.get("errors") is not None
    assert "season" in result.get("errors")
    assert "hr" in result.get("errors")
    assert "weathersit" in result.get("errors")

def test_make_prediction_with_single_record(prediction_input):
    # Given
    single_record = prediction_input.iloc[[0]]

    # When
    result = make_prediction(input_data=single_record)

    # Then
    predictions = result.get("predictions")
    assert predictions is not None, "Predictions should not be None"
    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array"
    assert len(predictions) == 1, f"Expected {1} predictions"
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions), "Each prediction should be an integer"
    assert result.get("errors") is None, "Errors should be None"