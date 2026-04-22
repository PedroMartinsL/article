
from sklearn.ensemble import RandomForestRegressor
from services.fit_predict import FitPrediction

model_execs = 10
data_title = 'rf'

# parameters = {'n_estimators': [50, 100, 200], 
#                   'max_depth': [5, 10, 15],
#                   'max_features': [0.6, 0.8, 1],
#                   #'max_samples' : [0.6, 0.8, 1],
#                   'time_window': [1, 7, 14]
#                  }

parameters = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, None],
    'max_features': [0.6, 0.8, 1],
    'min_samples_leaf': [1, 2],
    'time_window': [5, 10, 20]
}

model = RandomForestRegressor()
FitPrediction.train_sklearn(model_execs, data_title, parameters, model, "min_max_scaler", True)