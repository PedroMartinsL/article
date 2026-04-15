import numpy as np
from models.entities.MLModel import MLModel
from services import time_series_functions as tsf

class EnsembleModel(MLModel):
    def __init__(self, name: str, predictions: list, ts, **model_data: dict):
        """
            predictions: list of predictions to combine in the ensemble
            station_code / pollutant: used to fetch the actual values
            model_data: any other data to pass to MLModel (**kwargs)
        """
        # Ensemble
        y_pred = np.array(EnsembleModel.get_ensemble(predictions=predictions)).reshape(-1)
        y_true = ts.values.reshape(-1)

        y_true = y_true[-len(y_pred):]

        valid_mask = ~np.isnan(y_pred)

        y_pred_test = y_pred[valid_mask]
        y_true_test = y_true[valid_mask]


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