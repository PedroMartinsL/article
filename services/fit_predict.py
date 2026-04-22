import pickle
import time
import json
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from pmdarima import ARIMA, auto_arima
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from extractor import get_dataframe_by_station_and_pollutant
from services import time_series_functions as tsf
from sklearn.base import clone

from sklearn import preprocessing


class FitPrediction():

    CONFIG_PATH = './'
    SAVE_PATH = './models/results/'
    
    def get_windowing(
        ts_normalized: Union[pd.Series, pd.DataFrame],
        time_window: int,
        horizon: int,
        prefix: str = ''
    ) -> pd.DataFrame:
        """
        Converts a time series into a supervised learning dataset using lag features (windowing).

        Each row in the output contains:
        - The previous `time_window` values (lags) as input features
        - The target value (`actual`) to be predicted

        Example:
            time_window = 3, horizon = 1

            Original series:
            [1, 2, 3, 4]

            Output:
            lag_3  lag_2  lag_1  actual
            1      2      3       4

        Parameters
        ----------
        ts_normalized : pd.Series or pd.DataFrame
            Input time series (normalized or not). It must contain the target values
            (usually in a column named 'actual') or be a single series.

        time_window : int
            Number of past observations (lags) used as input features.

        horizon : int
            Forecast horizon:
            - 1 → one-step ahead prediction
            - >1 → multi-step forecast (only the final step is kept as target)

        prefix : str, optional (default = '')
            Prefix added to column names (useful when working with exogenous variables).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing:
            - Input features: lag_1, lag_2, ..., lag_n
            - Target column: 'actual'

            Format:
                [lag_n, ..., lag_1, actual]

        Notes
        -----
        - Intermediate horizon columns (hor_*) are removed
        - Only the final target ('actual') is kept
        - This transformation allows traditional ML models to work with time series data
        """
        ts_windowed = tsf.create_windowing(
            lag_size=(time_window + (horizon-1)),
            df=ts_normalized
        )

        columns_lag = [f'lag_{l}{prefix}'for l in reversed(
            range(1, time_window+1))]
        columns_horizon = [
            f'hor_{l}{prefix}'for l in range(1, horizon)] + ['actual']
        ts_windowed.columns = columns_lag + columns_horizon

        ts_windowed = ts_windowed[columns_lag+['actual']]
        return ts_windowed
    
    def get_scaler(scaler: str = "min_max_scaler"):
        if scaler == "standard_scaler":
            return preprocessing.StandardScaler()
        elif scaler == "min_max_scaler":
            return  preprocessing.MinMaxScaler()
        else:
            raise ValueError("Any Scaler with this name")

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
            raise('Exogen')

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
        result_type = tsf.result_options.val_result

        list_params = list(ParameterGrid(parameters))

        for params in tqdm(list_params, desc='GridSearch'):

            result = None
            params_actual = params.copy()
            del params_actual['time_window']

            forecaster = clone(model).set_params(** params_actual)

            result_atual = []
            for t in range(0, model_execs):
                retults = FitPrediction.single_model(
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

                #Json Vars
                test_size = i['test_size']
                val_size = i['val_size']
                type_data = i['type_data']
                horizon = i['horizon']
                # min_max = i['hour_min_max']

                pollutant = i['pollutant']
                station_code = i['station_code']

                real = get_dataframe_by_station_and_pollutant(station_code=station_code, pollutant=pollutant)

                gs_result = FitPrediction.do_grid_search(
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

                save_path_actual = FitPrediction.get_save_path_actual(type_data, data_title)
                os.makedirs(save_path_actual, exist_ok=True)

                title_temp = FitPrediction.get_title_temp(type_data, data_title)
                for _ in range(0, model_execs):
                    FitPrediction.single_model(
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

    def get_save_path_actual(type_data, data_title:str):
        return FitPrediction.SAVE_PATH+str(type_data)+'-'+data_title+'/'
    
    def get_title_temp(type_data, data_title):
        return str(type_data) + '-' + data_title
    
    def get_configuration_by_id(id: int):
        with open(f'{FitPrediction.CONFIG_PATH}models_config.json') as f:
            data = json.load(f)

        for item in data:
            if item.get("type_data") == id:
                return item

        return None


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

            save_path = FitPrediction.get_save_path_actual(type_data, data_title)
            os.makedirs(save_path, exist_ok=True)

            title_temp = FitPrediction.get_title_temp(type_data, data_title)

            for _ in range(model_execs):

                if normalize:
                    scaler = FitPrediction.get_scaler(normalize)
                    ts_used = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()
                else:
                    ts_used = ts.values

                preds = np.full(len(ts), np.nan)

                history = list(ts_used[:train_size])
                model = ArimaFitPrediction.build_arima(history, parameters, auto)

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
                    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

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


    # def walk_forward_arima(
    #     ts,
    #     train_size,
    #     val_size,
    #     test_size,
    #     normalize=False,
    #     scaler=None,
    #     parameters=None,
    #     horizon=1
    # ):
    #     # -----------------------------
    #     # Preparação dos dados
    #     # -----------------------------
    #     if normalize:
    #         full_series = scaler.transform(ts.values.reshape(-1, 1)).flatten()
    #         train_used = full_series[:train_size]
    #     else:
    #         full_series = ts.values
    #         train_used = full_series[:train_size]

    #     history = list(train_used)

    #     preds = []

    #     total_steps = val_size + test_size
    #     start_index = train_size

    #     model = auto_arima(
    #         history,
    #         **(parameters or {})
    #     )

    #     # -----------------------------
    #     # Walk-forward
    #     # -----------------------------
    #     for i in range(total_steps):

    #         # pred 1 step
    #         yhat = model.predict(n_periods=horizon)[0]
    #         preds.append(yhat)
    #         real_value = full_series[start_index + i]
    #         model.update(real_value)

    #     if normalize:
    #         preds = scaler.inverse_transform(
    #             np.array(preds).reshape(-1, 1)
    #         ).flatten()

    #     y_pred_full = np.full(len(ts), np.nan)
    #     y_pred_full[-(val_size + test_size):] = preds

    #     return y_pred_full
    

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