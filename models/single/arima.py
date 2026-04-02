from pmdarima import auto_arima
from services.fit_predict import FitPrediction

model_execs = 10
data_title = 'auto_arima'

parameters = {
    'start_p': 1,
    'max_p': 3,
    'start_q': 0,
    'max_q': 2,
    'd': None,
    'seasonal': True,
    'm': 7,
}

def train_auto_arima(model_execs, data_title, parameters, series):
    results = []

    for _ in range(model_execs):
        model = auto_arima(
            series,
            start_p=parameters['start_p'],
            max_p=parameters['max_p'],
            start_q=parameters['start_q'],
            max_q=parameters['max_q'],
            d=parameters['d'],
            seasonal=parameters['seasonal'],
            m=parameters['m'],
            trace=False,
            error_action='ignore',
            suppress_warnings=True
        )

        forecast = model.predict(n_periods=1)

        results.append({
            'order': model.order,
            'aic': model.aic(),
            'forecast': forecast
        })

    return results

results = train_auto_arima(
    model_execs,
    data_title,
    parameters,
    series=None  # sua série (ex: pandas Series)
)