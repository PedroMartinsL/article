import numpy as np
from extractor import get_dataframe_by_station_and_pollutant
from models.entities.MLModel import MLModel
from services import time_series_functions as tsf

class EnsembleModel(MLModel):
    def __init__(self, name: str, predictions: list, station_code: str, pollutant:str, **model_data: dict):
        """
            predictions: list of predictions to combine in the ensemble
            station_code / pollutant: used to fetch the actual values
            model_data: any other data to pass to MLModel (**kwargs)
        """
        y_pred = EnsembleModel.get_ensemble(predictions=predictions)
        test_size = len(y_pred)

        ts = get_dataframe_by_station_and_pollutant(
            station_code=station_code,
            pollutant=pollutant
        )
        y_true = ts.values

        y_true_test = y_true[-test_size:]
        y_pred_test = y_pred[-test_size:]

        # Add ensemble metrics into the dict
        model_data['test_metrics'] = tsf.gerenerate_metric_results(y_true_test, y_pred_test)
        model_data['real_values'] = y_true
        model_data['predicted_values'] = y_pred
        model_data['params'] = {'ensemble_size': len(predictions)}

        super().__init__(name, **model_data)


    def get_ensemble(predictions):

        # weights = [0.2, 0.5]
        # return np.mean(
        #     [pred for _, pred in predictions],
        #     axis=0
        # )
    

        preds = [pred for _, pred in predictions]

        min_len = min(len(p) for p in preds)

        preds_aligned = [p[-min_len:] for p in preds]

        return np.mean(preds_aligned, axis=0)