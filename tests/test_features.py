import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import (
    WeekdayImputer,
    WeathersitImputer,
    Mapper,
    OutlierHandler,
    WeekdayOneHotEncoder
)


def test_weekday_imputer():
    # Given
    test_data = pd.DataFrame({
        "dteday": ["2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06"],
        "weekday": ["Mon", "Tue", np.nan, "Thu", np.nan],
        "other_col": [1, 2, 3, 4, 5]
    })
    
    # When
    imputer = WeekdayImputer(date_column="dteday", weekday_column="weekday")
    imputer.fit(test_data)
    result = imputer.transform(test_data)
    
    # Then
    assert not result["weekday"].isna().any()
    assert result["weekday"].iloc[2] == "Wed"  # Assuming the imputed value is correct
    assert result["weekday"].iloc[4] == "Fri"  # Assuming the imputed value is correct


def test_weathersit_imputer():
    # Given
    test_data = pd.DataFrame({
        "weathersit": ["clear", np.nan, "rainy", "misty", np.nan],
        "other_col": [1, 2, 3, 4, 5]
    })
    
    # When
    imputer = WeathersitImputer(column="weathersit")
    imputer.fit(test_data)
    result = imputer.transform(test_data)
    
    # Then
    assert not result["weathersit"].isna().any()
    assert result["weathersit"].iloc[1] == "clear"  # Assuming "clear" is the most frequent
    assert result["weathersit"].iloc[4] == "clear"  # Assuming "clear" is the most frequent


def test_mapper():
    # Given
    test_data = pd.DataFrame({
        "season": ["spring", "summer", "autumn", "winter"],
        "weekday": ["monday", "tuesday", "wednesday", "thursday"],
        "weathersit": ["clear", "misty", "rainy", "clear"]
    })
    
    custom_mappings = {
        "season": {
            "spring": 1,
            "summer": 2,
            "autumn": 3,
            "winter": 4
        },
        "weekday": {
            "monday": 1,
            "tuesday": 2,
            "wednesday": 3,
            "thursday": 4,
            "friday": 5,
            "saturday": 6,
            "sunday": 7
        }
    }
    
    # When using custom mappings
    mapper = Mapper(variables=["season", "weekday"], mappings=custom_mappings)
    mapper.fit(test_data)
    result = mapper.transform(test_data)
    
    # Then
    assert result["season"].tolist() == [1, 2, 3, 4]
    assert result["weekday"].tolist() == [1, 2, 3, 4]
    assert result["weathersit"].tolist() == ["clear", "misty", "rainy", "clear"]


def test_outlier_handler():
    # Given
    test_data = pd.DataFrame({
        "temp": [0.1, 0.2, 0.8, 0.99, 1.2],
        "atemp": [0.2, 0.3, 0.7, 0.95, 1.1],
        "hum": [0.3, 0.4, 0.6, 0.85, 1.05]
    })
    
    # When
    outlier_handler = OutlierHandler(columns=["temp", "atemp", "hum"], upper_bound=1.0)
    outlier_handler.fit(test_data)
    result = outlier_handler.transform(test_data)
    
    # Then
    assert result["temp"].max() <= 1.0
    assert result["atemp"].max() <= 1.0
    assert result["hum"].max() <= 1.0
    assert result["temp"].iloc[4] == 1.0
    assert result["atemp"].iloc[4] == 1.0
    assert result["hum"].iloc[4] == 1.0


def test_weekday_one_hot_encoder():
    # Given
    train_data = pd.DataFrame({
        "weekday": ["monday", "tuesday", "wednesday", "thursday", "friday"]
    })
    
    test_data = pd.DataFrame({
        "weekday": ["monday", "sunday", "saturday", "friday", "monday"]
    })
    
    # When
    encoder = WeekdayOneHotEncoder(column="weekday")
    encoder.fit(train_data)
    result = encoder.transform(test_data)
    
    # Then
    # Check that one-hot encoding was created with all original categories
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        assert f"weekday_{day}" in result.columns
    
    # Check correct encoding for first row (monday)
    assert result.loc[0, "weekday_monday"] == 1
    assert result.loc[0, "weekday_tuesday"] == 0
    
    # Check handling of new categories in test data
    assert "weekday_sunday" not in result.columns
    assert "weekday_saturday" not in result.columns
    
    # Check weekday column was dropped
    assert "weekday" not in result.columns