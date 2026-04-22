from pmdarima import auto_arima
from services.fit_predict import ArimaFitPrediction

model_execs = 1
data_title = 'arima'

parameters = {
    "seasonal": True,
    "m": 24,
    "d": 1,
    "stepwise": True,
    "suppress_warnings": True,
    "error_action": "ignore",
    "information_criterion": "aic",
    "test": "adf",
}

ArimaFitPrediction.train_arima(
    model_execs=model_execs,
    data_title=data_title,
    normalize="min_max_scaler",
    parameters=parameters,
    auto=True,
    shift=1
)