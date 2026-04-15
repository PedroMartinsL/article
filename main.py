import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import plot_acf, plot_pacf

from models.entities.EnsembleModel import EnsembleModel
from models.entities.MLModel import MLModel

from extractor import get_dataframe_by_station_and_pollutant
from services.fit_predict import FitPrediction

def get_pkl_files(type_data, auto=True) -> list:
    base_path = "models/results"

    if not auto:
        return [
            {"ARIMA": "1-arima/1-arima-090426000326.pkl"},
            {"SVR": "1-svr/1-svr(tw12)-140426163247.pkl"},
            {"GB": "1-gb/1-gb(tw12)-080426233533.pkl"},
            {"MLP": "1-mlp/1-mlp(tw12)-090426000516.pkl"},
            {"RF": "1-rf/1-rf(tw12)-090426000939.pkl"},
            {"XGB": "1-xgb/1-xgb(tw12)-080426235657.pkl"},
        ]

    results = []

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if not os.path.isdir(folder_path):
            continue

        if not folder.startswith(f"{type_data}-"):
            continue

        model_name = folder.split("-")[1].upper()

        for file in os.listdir(folder_path):
            if file.endswith(".pkl"):
                full_path = os.path.join(folder, file)
                results.append({model_name: full_path})

    return results



if __name__ == "__main__":
    data = FitPrediction.get_configuration_by_id(1)
    test_size = data['test_size']
        
    pkl_files = get_pkl_files(type_data=1, auto=True)

    pollutant = "MP10"
    station_code = "SP71"

    #Entry
    df = get_dataframe_by_station_and_pollutant(station_code, pollutant)

    ts = df['actual']
    
    models: List[MLModel] = MLModel.load_models(pkl_files)

    predictions: list = MLModel.get_predictions(models=models)

    #Ensemble Model

    ensemble_class = EnsembleModel(name="Ensemble", predictions=predictions, ts=df)

    models.append(ensemble_class)

    #Ensemble metrics
    ensemble_pred = ensemble_class.predicted_values

    #Real values
    real_values = models[0].real_values

    y_test = real_values[-test_size:]
    
    ########

    #Windowed model
    windowed_model = MLModel.get_shift_model(lag=1, ts=ts)

    models.append(windowed_model)

    #Plotting data

    MLModel.plot_perfomance(y_test, ensemble_pred, pollutant, predictions)

    MLModel.plot_test_metrics_table(models)