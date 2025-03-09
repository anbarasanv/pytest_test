import sys
from pathlib import Path
import math
import numpy as np
import pytest
import pandas as pd
from bikeshare_model.predict import make_prediction
from bikeshare_model.config.core import config
from bikeshare_model import __version__ as _version

# Add the root directory to the system path
file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))


def test_pipeline_trained():
    """Test if the trained model pipeline file exists with the correct version."""
    # Constants
    pipeline_file = f"{config.app_config_.pipeline_save_file}{_version}.pkl"

    # Verify trained model file path
    expected_pipeline_file = f"bikeshare__model_output_v{_version}.pkl"
    assert pipeline_file == expected_pipeline_file, \
        f"Pipeline file mismatch: expected {expected_pipeline_file}, got {pipeline_file}"


def test_make_prediction(sample_input_data):
    """Test predictions for valid input data."""
    # Expected values
    expected_first_prediction_value = 139
    expected_no_predictions = 2

    # Perform prediction
    result = make_prediction(input_data=sample_input_data)

    # Validate predictions
    predictions = result.get("predictions")
    assert predictions is not None, "Predictions should not be None"
    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array"
    assert len(predictions) == expected_no_predictions, \
        f"Expected {expected_no_predictions} predictions, got {len(predictions)}"
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions), \
        "Each prediction should be an integer"
    assert math.isclose(predictions[0], expected_first_prediction_value, rel_tol=1e-2), \
        f"First prediction should be close to {expected_first_prediction_value}, got {predictions[0]}"
    assert result.get("errors") is None, "Errors should be None"


def test_make_prediction_with_invalid_data():
    """Test predictions for invalid input data."""
    # Invalid data sample
    invalid_data = {
        "dteday": ["2012-11-05", "2012-11-05"],  # Valid dates
        "season": [1, "not"],                   # Invalid season value
        "hr": [6, 25],                          # Invalid hour
        "holiday": [0, 1],
        "weekday": [1, 2],
        "workingday": [0, 1],
        "weathersit": [1, 5],                   # Invalid weathersit
        "temp": [38.0, 15.5],
        "atemp": [35.5, 14.0],
        "hum": [0.8, 0.4],
        "windspeed": [0.3, 0.2],
        "casual": [30, 20],
        "registered": [150, 120],
    }

    # Perform prediction with invalid data
    result = make_prediction(input_data=invalid_data)

    # Validate errors in response
    errors = result.get("errors")
    assert errors is not None, "Errors should not be None for invalid data"
    assert "season" in errors, "'season' should be reported in errors"
    assert "hr" in errors, "'hr' should be reported in errors"
    assert "weathersit" in errors, "'weathersit' should be reported in errors"


def test_make_prediction_with_single_record(prediction_input):
    """Test predictions for a single record input."""
    single_record = prediction_input.iloc[[0]]  # Extract a single record

    # Perform prediction
    result = make_prediction(input_data=single_record)

    # Validate predictions for single record
    predictions = result.get("predictions")
    assert predictions is not None, "Predictions should not be None"
    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array"
    assert len(predictions) == 1, "Expected a single prediction"
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions), \
        "Each prediction should be an integer"
    assert result.get("errors") is None, "Errors should be None for valid single record"
