from sklearn.ensemble import GradientBoostingRegressor
from services.fit_predict import FitPrediction


model_execs = 10
data_title = 'gb'

parameters = {
    'n_estimators': [100, 300, 500],
    'max_depth': [2, 3, 4],
    'max_features': [0.6, 0.8, 1],
    'subsample': [0.6, 0.8, 1],
    'learning_rate': [0.01, 0.05, 0.1],
    'time_window': [5, 10, 20, 30]
}

model = GradientBoostingRegressor()
FitPrediction.execute(
    model_execs=model_execs,
    data_title=data_title,
    parameters=parameters,
    model=model,
    normalize="min_max_scaler", 
    differencing=True
)