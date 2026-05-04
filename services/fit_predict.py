import json
from typing import Union
import os

import pandas as pd
from services import time_series_functions as tsf


from sklearn import preprocessing


class FitPrediction():

    CONFIG_PATH = './'
    SAVE_PATH = './models/results/'

    def _detect_model_type(model):

        model_str = str(type(model)).lower()

        # 🔹 se for classe (NHITS, KAN...)
        if isinstance(model, type):
            if "neuralforecast" in str(model).lower():
                return "neuralforecast"

        # 🔹 instancia neuralforecast (caso raro)
        if "neuralforecast" in model_str:
            return "neuralforecast"

        # 🔹 sklearn
        if hasattr(model, "fit") and hasattr(model, "predict"):
            return "sklearn"

        # 🔹 arima
        if "arima" in model_str:
            return "arima"

        return "unknown"

    def execute(**allParams):
        """
        Executor genérico que detecta o tipo do modelo e delega os parâmetros.
        """
        from services.forecats.arima_prediction import ArimaFitPrediction
        from services.forecats.neuralforecast_prediction import NeuralForecastFitPrediction
        from services.forecats.sklearn_prediction import SklearnFitPrediction

        if 'model' not in allParams:
            raise ValueError("Você precisa passar 'model' nos parâmetros")

        model = allParams['model']

        model_type = FitPrediction._detect_model_type(model)

        # sklearn
        if model_type == "sklearn":
            return SklearnFitPrediction.train_sklearn(**allParams)

        # arima
        elif model_type == "arima":
            return ArimaFitPrediction.train_arima(**allParams)

        # neuralforecast
        elif model_type == "neuralforecast":
            # aqui ajusta automaticamente
            return NeuralForecastFitPrediction.train_neuralforecast(**allParams)

        else:
            raise ValueError(f"Tipo de modelo não suportado: {type(model)}")

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
            return preprocessing.MinMaxScaler()
        else:
            raise ValueError("Any Scaler with this name")

    def get_save_path_actual(type_data, data_title: str):
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




