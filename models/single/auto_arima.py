from pmdarima import auto_arima
from services.fit_predict import ArimaFitPrediction

model_execs = 1
data_title = 'arima'

parameters = {
    "seasonal" : True,
    "m":7, 
    "stepwise":True,
    "suppress_warnings":True,
    "error_action":'ignore',
}

ArimaFitPrediction.train_arima(
    model_execs=model_execs,
    data_title=data_title,
    normalize=False,
    parameters=parameters
)