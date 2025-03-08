import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model import __version__ as _version

# Constants
PIPELINE_FILE = f"{config.app_config_.pipeline_save_file}{_version}.pkl"


@pytest.fixture
def sample_input_data():
    """Fixture for sample input data for testing predictions."""
    return {
    "dteday": ["2012-11-05", "2011-07-13"],
    "season": ["winter", "fall"],
    "hr": ["6am", "4am"],
    "holiday": ["No", "No"],
    "weekday": ["Mon", "Wed"],
    "workingday": ["Yes", "Yes"],
    "weathersit": ["Mist", "Clear"],
    "temp": [6.10, 26.78],
    "atemp": [3.0014, 28.9988],
    "hum": [49.0, 58.0],
    "windspeed": [19.0012, 16.9979],
    "casual": [4, 0],
    "registered": [135, 5],
    }


@pytest.fixture
def sample_input_df(sample_input_data):
    """Fixture that converts sample input data to a pandas DataFrame."""
    return pd.DataFrame(sample_input_data)


@pytest.fixture
def raw_training_data():
    """Fixture for loading a small subset of the training data for tests."""
    # You can replace this with actual loading from a test dataset file
    test_data = pd.DataFrame({
        "dteday": ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04", "2011-01-05"],
        "season": ["winter", "winter", "winter", "winter", "winter"],
        "hr": ["12pm", "1pm", "2pm", "3pm", "4pm"],
        "holiday": ["No", "No", "No", "No", "No"],
        "weekday": ["Sat", "Sun", "Mon", "Tue", "Wed"],
        "workingday": ["No", "No", "Yes", "Yes", "Yes"],
        "weathersit": ["Clear", "Clear", "Clear", "Clear", "Mist"],
        "temp": [10.0, 12.0, 11.0, 9.0, 8.0],
        "atemp": [12.0, 13.0, 12.5, 10.0, 9.0],
        "hum": [50.0, 45.0, 48.0, 52.0, 60.0],
        "windspeed": [5.0, 6.0, 7.0, 8.0, 10.0],
        "casual": [100, 120, 110, 90, 80],
        "registered": [200, 210, 190, 180, 170],
        "cnt": [300, 330, 300, 270, 250]
    })
    return test_data


@pytest.fixture()
def pipeline_trained():
    """Fixture for loading the trained model pipeline."""
    # Load the saved pipeline from the appropriate location
    pipeline_file_path = Path(config.app_config_.trained_model_dir) / PIPELINE_FILE
    return load_pipeline(file_name=pipeline_file_path)


@pytest.fixture()
def prediction_input():
    """Fixture with a minimal input example for model prediction testing."""
    return pd.DataFrame({
        "dteday": ["2012-11-05", "2011-07-13"],
    "season": ["winter", "fall"],
    "hr": ["6am", "4am"],
    "holiday": ["No", "No"],
    "weekday": ["Mon", "Wed"],
    "workingday": ["Yes", "Yes"],
    "weathersit": ["Mist", "Clear"],
    "temp": [6.10, 26.78],
    "atemp": [3.0014, 28.9988],
    "hum": [49.0, 58.0],
    "windspeed": [19.0012, 16.9979],
    "casual": [4, 0],
    "registered": [135, 5],
    })

