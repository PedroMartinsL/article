from services.fit_predict import FitPrediction

from sklearn.svm import SVR

model_execs = 1
data_title = 'svr'
#     'gamma': ['scale', 'auto']

parameters = {
    'C':[10, 150, 100, 1000], 
    'gamma': [0.1, 0.01, 0.001],
    'kernel':["rbf"],
    'epsilon': [1.0, 0.5, 0.1, 0.01, 0.001], 
    'tol':[0.001],
    'time_window': [1, 7, 12]
    }

model = SVR(max_iter=100000)

FitPrediction.execute(
    model_execs=model_execs,
    data_title=data_title,
    parameters=parameters,
    model=model,
    normalize="standard_scaler", 
    differencing=True
)
