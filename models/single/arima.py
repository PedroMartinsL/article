from pmdarima import ARIMA
from services.fit_predict import ArimaFitPrediction

model_execs = 1
data_title = 'arima'

parameters = {
    "order": (1,1,1),
    "seasonal_order": (0,0,0,0),
    "trend": "c",
    "method": "lbfgs",
    "maxiter": 100
}

ArimaFitPrediction.train_arima(
    model_execs=model_execs,
    data_title=data_title,
    normalize=False,
    parameters=parameters,
    auto=False
)