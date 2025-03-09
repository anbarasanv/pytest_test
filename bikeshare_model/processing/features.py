from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

class WeekdayImputer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='dteday', weekday_column='weekday'):
        self.date_column = date_column
        self.weekday_column = weekday_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert date strings to datetime objects and extract weekdays
        X[self.weekday_column] = X.apply(
            lambda row: datetime.strptime(row[self.date_column], '%Y-%m-%d').strftime('%A')[:3] 
            if pd.isna(row[self.weekday_column]) else row[self.weekday_column], axis=1
        )
        return X

class WeathersitImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column='weathersit'):
        self.column = column
        self.most_frequent = None

    def fit(self, X, y=None):
        self.most_frequent = X[self.column].mode()[0]
        return self

    def transform(self, X):
        X[self.column].fillna(self.most_frequent, inplace=True)
        return X

class Mapper(BaseEstimator, TransformerMixin):
    default_mappings = {
        'yr': {2011: 0, 2012: 1},
        'mnth': {i: i for i in range(1, 13)},
        'season': {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4},
        'weathersit': {'Clear': 1, 'Mist': 2, 'Light Rain': 3, 'Heavy Rain': 4},
        'holiday': {'No': 0, 'Yes': 1},
        'workingday': {'No': 0, 'Yes': 1},
        'hr': {f'{i}am': i for i in range(12)} | {f'{i}pm': i + 12 for i in range(1, 12)} | {'12pm': 12}
    }

    def __init__(self, variables: Optional[List[str]] = None, mappings: Optional[dict] = None):
        self.variables = variables or []
        self.mappings = mappings or self.default_mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in self.variables:
            if column in X.columns:
                X[column] = X[column].map(self.mappings.get(column, {}))
        return X

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[List[str]] = None, factor=1.5):
        self.columns = columns
        self.factor = factor
        self.bounds = {}

    def fit(self, X, y=None):
        columns_to_process = self.columns or X.select_dtypes(include=[np.number]).columns
        for column in columns_to_process:
            Q1, Q3 = X[column].quantile(0.25), X[column].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds[column] = {
                'lower': Q1 - self.factor * IQR,
                'upper': Q3 + self.factor * IQR
            }
        return self

    def transform(self, X):
        for column, bounds in self.bounds.items():
            lower_bound = bounds['lower']
            upper_bound = bounds['upper']
            X[column] = np.clip(X[column], lower_bound, upper_bound)
        return X

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column: str):
        self.column = column
        self.categories_ = None

    def fit(self, X, y=None):
        self.categories_ = X[self.column].astype(str).unique()
        return self

    def transform(self, X):
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' does not exist in the DataFrame.")
        
        one_hot_encoded = pd.get_dummies(
            pd.Categorical(X[self.column], categories=self.categories_),
            prefix=self.column
        )
        
        return pd.concat([X.drop(columns=[self.column]), one_hot_encoded], axis=1)

class DropColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[self.column_name], errors='ignore')
