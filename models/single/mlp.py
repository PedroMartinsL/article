from sklearn.neural_network import MLPRegressor
from services.fit_predict import FitPrediction

model_execs = 10
data_title = 'mlp'

parameters = {'hidden_layer_sizes': [20, 50, 100], 
                  'max_iter': [1000],
                  'tol': [0.001, 0.0001, 0.00001],
                  'time_window': [12]
                 }


model = MLPRegressor(activation='logistic', solver='lbfgs') 
FitPrediction.train_sklearn(model_execs, data_title, parameters, model)