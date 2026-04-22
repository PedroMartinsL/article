from pmdarima import ARIMA
from services.fit_predict import ArimaFitPrediction

model_execs = 1
data_title = 'arima'

parameters = {
    "order": (1,1,0),         
    "trend": "n",
    "method": "lbfgs",
    "maxiter": 100,
    "suppress_warnings": True,
}

ArimaFitPrediction.train_arima(
    model_execs=model_execs,
    data_title=data_title,
    normalize=False,
    parameters=parameters,
    auto=False,
    shift=1
)