from sklearn.ensemble import GradientBoostingRegressor
from services.fit_predict import FitPrediction


model_execs = 10
data_title = 'gb'

parameters = {'n_estimators': [50, 100, 200], 
                  'max_depth': [5, 10, 15],
                  'max_features': [0.6, 0.8, 1],
                  'subsample' : [0.6, 0.8, 1],
                  'learning_rate': [0.1, 0.3, 0.5],
                  'time_window': [12]
                 }
model = GradientBoostingRegressor()
FitPrediction.train_sklearn(model_execs, data_title, parameters, model)