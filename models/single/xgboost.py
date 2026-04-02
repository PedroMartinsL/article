from xgboost import XGBRegressor
from services.fit_predict import FitPrediction

model_execs = 10
data_title = 'xgb'

parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1],
    'colsample_bytree': [0.6, 0.8, 1],
    'time_window': [12]
}

model = XGBRegressor(
    objective='reg:squarederror',
    verbosity=0
)

FitPrediction.train_sklearn(
    model_execs,
    data_title,
    parameters,
    model
)