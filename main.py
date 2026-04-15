from typing import List
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import plot_acf, plot_pacf

from models.entities.EnsembleModel import EnsembleModel
from models.entities.MLModel import MLModel

from extractor import get_dataframe_by_station_and_pollutant


if __name__ == "__main__":

    pkl_files = [
        {"ARIMA": "1-arima/1-arima-090426000326.pkl"},
        {"SVR": "1-svr/1-svr(tw12)-140426163247.pkl"},
        {"GB": "1-gb/1-gb(tw12)-080426233533.pkl"},
        {"MLP": "1-mlp/1-mlp(tw12)-090426000516.pkl"},
        {"RF": "1-rf/1-rf(tw12)-090426000939.pkl"},
        {"XGB": "1-xgb/1-xgb(tw12)-080426235657.pkl"},
    ]

    pollutant = "MP10"
    station_code = "SP71"

    #Entry
    df = get_dataframe_by_station_and_pollutant(station_code, pollutant)

    ts = df['actual']
    
    models: List[MLModel] = MLModel.load_models(pkl_files)
    y_test = models[0].real_values
    X_test = np.arange(1, 21).reshape(-1, 1)

    predictions: list = MLModel.get_predictions(models=models)

    ensemble_class = EnsembleModel(name="Ensemble", predictions=predictions, station_code=station_code, ts=df, pollutant=pollutant)

    ensemble_pred = ensemble_class.predicted_values

    test_index = ts.index[-len(ensemble_pred):]

    models.append(ensemble_class)

    MLModel.plot_perfomance(test_index, y_test, ensemble_pred, pollutant, predictions)

    MLModel.plot_test_metrics_table(models)