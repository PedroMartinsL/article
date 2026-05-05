import time
import json
import os
from typing import Dict, Optional

import numpy as np
from pmdarima import ARIMA, auto_arima

from services.fit_predict import FitPrediction
from extractor import get_dataframe_by_station_and_pollutant
from services import time_series_functions as tsf

class ArimaFitPrediction(FitPrediction):

    def train_arima(
        model_execs: int,
        data_title: str,
        auto: bool = True,
        parameters: Optional[Dict] = None,
        normalize: str = None,
        shift: int = 0,
    ) -> None:

        with open(f'{FitPrediction.CONFIG_PATH}models_config.json') as f:
            data = json.load(f)

        for cfg in data:

            if cfg['activate'] != 1:
                continue

            test_size = cfg['test_size']
            val_size = 0
            horizon = cfg['horizon']
            type_data = cfg['type_data']

            pollutant = cfg['pollutant']
            station_code = cfg['station_code']

            real = get_dataframe_by_station_and_pollutant(
                station_code=station_code,
                pollutant=pollutant
            )

            ts = real['actual']

            train_size = len(ts) - test_size

            save_path = FitPrediction.get_save_path_actual(
                type_data, data_title)
            os.makedirs(save_path, exist_ok=True)

            title_temp = FitPrediction.get_title_temp(type_data, data_title)

            for _ in range(model_execs):

                if normalize:
                    scaler = FitPrediction.get_scaler(normalize)
                    ts_used = scaler.fit_transform(
                        ts.values.reshape(-1, 1)).flatten()
                else:
                    ts_used = ts.values

                preds = np.full(len(ts), np.nan)

                history = list(ts_used[:train_size])
                model = ArimaFitPrediction.build_arima(
                    history, parameters, auto)

                for i in range(test_size):
                    yhat = model.predict(n_periods=horizon)[0]

                    idx = train_size + i - shift

                    if idx >= 0:
                        preds[idx] = yhat

                    real_value = ts_used[train_size + i]
                    model.update([real_value])

                if shift:
                    for i in range(shift):
                        yhat = model.predict(n_periods=1)[0]
                        preds[-1] = yhat

                if normalize:
                    preds = scaler.inverse_transform(
                        preds.reshape(-1, 1)).flatten()

                tsf.make_metrics_avaliation(
                    y_true=ts,
                    y_pred=preds,
                    test_size=test_size,
                    val_size=val_size,
                    return_type=tsf.result_options.save_result,
                    model_params={'order': model.order},
                    title=save_path + title_temp,
                    prevs_df=None
                )

                time.sleep(1)

    def build_arima(train_used, parameters, auto):
        try:
            if auto:
                return auto_arima(train_used, **(parameters or {}))
            else:
                model = ARIMA(**(parameters or {}))
                model.fit(train_used)
                return model

        except Exception as e:
            print("Error building arima model: ", e)