
import time
import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from services.fit_predict import FitPrediction
from extractor import get_dataframe_by_station_and_pollutant
from services import time_series_functions as tsf
from sklearn.base import clone

class SklearnFitPrediction:

    def single_model(
        title: str,
        time_window: int,
        time_series: pd.DataFrame,
        model: Any,
        test_size: int,
        val_size: int,
        return_option: Any,
        normalize: str = None,
        horizon: int = 1,
        recursive: bool = False,
        differencing: bool = False
    ) -> Dict:

        train_size = len(time_series) - test_size

        if time_series.shape[1] > 1:
            raise ('Exogen')

        horizon_to_use = horizon

        if recursive:
            horizon = 1

        # normalize
        if normalize:
            scaler = FitPrediction.get_scaler(normalize)
            scaler.fit(
                time_series['actual'].values[0:train_size].reshape(-1, 1))
            ts_normalized = scaler.transform(
                time_series['actual'].values.reshape(-1, 1))
            ts_normalized = pd.DataFrame({'actual': ts_normalized.flatten()})
        else:
            ts_normalized = time_series.copy()

        ts_to_window = ts_normalized
        if differencing:

            # Target
            ts_normalized["target"] = ts_normalized["actual"].diff()

            ts_normalized = ts_normalized.dropna()

            ts_to_window = ts_normalized[["target"]]

        # windowing with delta Y
        ts_windowed = FitPrediction.get_windowing(
            ts_to_window,
            time_window,
            horizon
        )

        reg = tsf.fit_sklearn_model(ts_windowed, model, test_size, val_size)

        if recursive and (horizon_to_use > 1):
            ts_windowed_test = FitPrediction.get_windowing(
                ts_to_window, time_window, horizon_to_use)
            ts_windowed_test = ts_windowed_test.iloc[-(test_size+val_size+10):]
            pred = tsf.predict_sklearn_model_recursive(
                ts_windowed_test, reg, horizon_to_use)
        else:
            pred = tsf.predict_sklearn_model(ts_windowed, reg)

        if differencing:
            # reconstruction Delta Y to Y
            actual_series = ts_normalized["actual"]

            base = actual_series.iloc[-len(pred):]
            # Retrieving Y
            pred = base.values + pred

        if normalize:
            pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

        # ground ts - retrieving Y values
        ts_atu = time_series['actual']
        ts_atu = ts_atu[-len(pred):]

        try:
            df_prevs = model.prevs_df
        except:
            df_prevs = None

        results = tsf.make_metrics_avaliation(
            ts_atu,
            pred,
            test_size,
            val_size,
            return_option,
            model.get_params(deep=True),
            title + '(tw' + str(time_window) + ')',
            df_prevs
        )
        return results

    def do_grid_search(
        real: pd.DataFrame,
        test_size: int,
        val_size: int,
        parameters: Dict,
        model: Any,
        horizon: int,
        recursive: bool,
        model_execs: int,
        normalize: str = None,
        differencing: bool = False
    ) -> Dict:

        best_model = None
        metric = 'RMSE'
        best_result = {'time_window': 0, metric: None}
        result_type = tsf.result_options.test_result

        list_params = list(ParameterGrid(parameters))

        for params in tqdm(list_params, desc='GridSearch'):

            result = None
            params_actual = params.copy()
            del params_actual['time_window']

            forecaster = clone(model).set_params(** params_actual)

            result_atual = []
            for t in range(0, model_execs):
                retults = SklearnFitPrediction.single_model(
                    'mlp',
                    params['time_window'],
                    real,
                    forecaster,
                    test_size,
                    val_size,
                    result_type,
                    normalize,
                    horizon,
                    recursive,
                    differencing
                )[metric]
                result_atual.append(retults)

            result = np.mean(np.array(result_atual))

            if best_result[metric] == None:
                best_model = forecaster
                best_result[metric] = result
                best_result['time_window'] = params['time_window']
            else:

                if best_result[metric] > result:
                    best_model = forecaster
                    best_result[metric] = result
                    best_result['time_window'] = params['time_window']

        result_model = {'best_result': best_result, 'model': best_model}
        return result_model

    def train_sklearn(
        model_execs: int,
        data_title: str,
        parameters: Dict,
        model: Any,
        normalize: str = None,
        differencing: bool = False
    ) -> None:

        with open(f'{FitPrediction.CONFIG_PATH}models_config.json') as f:
            data = json.load(f)

        recursive = False
        # use_log = False

        for i in data:

            if i['activate'] == 1:

                # Json Vars
                test_size = i['test_size']
                val_size = i['val_size']
                type_data = i['type_data']
                horizon = i['horizon']
                # min_max = i['hour_min_max']

                pollutant = i['pollutant']
                station_code = i['station_code']

                real = get_dataframe_by_station_and_pollutant(
                    station_code=station_code, pollutant=pollutant)

                gs_result = SklearnFitPrediction.do_grid_search(
                    real=real,
                    test_size=test_size,
                    val_size=val_size,
                    parameters=parameters,
                    model=model,
                    horizon=horizon,
                    recursive=recursive,
                    model_execs=model_execs,
                    normalize=normalize,
                    differencing=differencing
                )

                save_path_actual = FitPrediction.get_save_path_actual(
                    type_data, data_title)
                os.makedirs(save_path_actual, exist_ok=True)

                title_temp = FitPrediction.get_title_temp(
                    type_data, data_title)
                for _ in range(0, model_execs):
                    SklearnFitPrediction.single_model(
                        save_path_actual+title_temp,
                        gs_result['best_result']['time_window'],
                        real,
                        gs_result['model'],
                        test_size,
                        val_size,
                        tsf.result_options.save_result,
                        normalize,
                        horizon,
                        recursive,
                        differencing
                    )
                    time.sleep(1)