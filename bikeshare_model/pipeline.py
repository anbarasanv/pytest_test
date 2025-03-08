import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer, WeathersitImputer, Mapper, OutlierHandler, WeekdayOneHotEncoder, DropColumn

bikeshare_pipe=Pipeline([
    ('weekday_imputer', WeekdayImputer(
        # weekday_column=config.model_config_.weekday_col
        )),
    ('weathersit_imputer', WeathersitImputer()),
    ('mapper', Mapper()),
    ('outlier_handler', OutlierHandler(
        # columns=config.model_config_.numeric_cols
        )),
    ('weekday_encoder', WeekdayOneHotEncoder(column=config.model_config_.weekday_col)),
    ('drop_column', DropColumn(column_name=config.model_config_.dteday_col)), #Drop the column here
    #  ('model_rf', RandomForestClassifier(n_estimators=config.model_config_.n_estimators, 
    #                                      max_depth=config.model_config_.max_depth, 
    #                                      max_features=config.model_config_.max_features,
    #                                      random_state=config.model_config_.random_state))
    ('model_rf', RandomForestClassifier())
          
     ])
