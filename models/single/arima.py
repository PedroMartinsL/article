from pmdarima import ARIMA
from services.fit_predict import FitPrediction

model_execs = 1
data_title = 'arima'

parameters = {
    'start_p': 1,
    'max_p': 3,
    'start_q': 0,
    'max_q': 2,
    'd': None,
    'seasonal': True,
    'm': 7,
}

model = ARIMA()
FitPrediction.train_sklearn(model_execs, data_title, parameters, model)