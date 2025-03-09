import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import (
    WeekdayImputer,
    WeathersitImputer,
    Mapper,
    OutlierHandler,
    WeekdayOneHotEncoder,
)

# Add the parent directory to sys.path
file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))


def test_weekday_imputer():
    # Given
    test_data = pd.DataFrame({
        "dteday": ["2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06"],
        "weekday": ["Mon", "Tue", np.nan, "Thu", np.nan],
        "other_col": [1, 2, 3, 4, 5],
    })
    
    # When
    imputer = WeekdayImputer(date_column="dteday", weekday_column="weekday")
    result = imputer.fit_transform(test_data)
    
    # Then
    assert not result["weekday"].isna().any()
    assert result["weekday"].iloc[2] == "Wed"  # Imputed value
    assert result["weekday"].iloc[4] == "Fri"  # Imputed value


def test_weathersit_imputer():
    # Given
    test_data = pd.DataFrame({
        "weathersit": ["clear", np.nan, "rainy", "misty", np.nan],
        "other_col": [1, 2, 3, 4, 5],
    })
    
    # When
    imputer = WeathersitImputer(column="weathersit")
    result = imputer.fit_transform(test_data)
    
    # Then
    assert not result["weathersit"].isna().any()
    assert (result["weathersit"] == "clear").sum() >= 2  # Assuming "clear" is the most frequent


def test_mapper():
    # Given
    test_data = pd.DataFrame({
        "season": ["spring", "summer", "autumn", "winter"],
        "weekday": ["monday", "tuesday", "wednesday", "thursday"],
        "weathersit": ["clear", "misty", "rainy", "clear"],
    })
    
    custom_mappings = {
        "season": {"spring": 1, "summer": 2, "autumn": 3, "winter": 4},
        "weekday": {
            "monday": 1,
            "tuesday": 2,
            "wednesday": 3,
            "thursday": 4,
            "friday": 5,
            "saturday": 6,
            "sunday": 7,
        },
    }
    
    # When
    mapper = Mapper(variables=["season", "weekday"], mappings=custom_mappings)
    result = mapper.fit_transform(test_data)
    
    # Then
    assert result["season"].tolist() == [1, 2, 3, 4]
    assert result["weekday"].tolist() == [1, 2, 3, 4]
    assert result["weathersit"].equals(test_data["weathersit"])


def test_outlier_handler():
    # Given
    test_data = pd.DataFrame({
        "temp": [0.1, 0.2, 0.8, 0.99, 1.2],
        "atemp": [0.2, 0.3, 0.7, 0.95, 1.1],
        "hum": [0.3, 0.4, 0.6, 0.85, 1.05],
    })
    
    # When
    outlier_handler = OutlierHandler(columns=["temp", "atemp", "hum"], upper_bound=1.0)
    result = outlier_handler.fit_transform(test_data)
    
    # Then
    assert (result[["temp", "atemp", "hum"]] <= 1.0).all().all()
    assert result.loc[4, ["temp", "atemp", "hum"]].tolist() == [1.0, 1.0, 1.0]


def test_weekday_one_hot_encoder():
    # Given
    train_data = pd.DataFrame({
        "weekday": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    })
    
    test_data = pd.DataFrame({
        "weekday": ["monday", "sunday", "saturday", "friday", "monday"],
    })
    
    # When
    encoder = WeekdayOneHotEncoder(column="weekday")
    encoder.fit(train_data)
    result = encoder.transform(test_data)
    
    # Then
    expected_columns = [f"weekday_{day}" for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]]
    
    # Check one-hot encoding columns exist
    assert all(col in result.columns for col in expected_columns)
    
    # Check correct encoding for some examples
    assert result.loc[0, ["weekday_monday"]].values[0] == 1
    assert result.loc[0, ["weekday_tuesday"]].values[0] == 0
    
    # Ensure dropped columns and no extra columns for unseen categories
    assert set(result.columns).issubset(expected_columns)
