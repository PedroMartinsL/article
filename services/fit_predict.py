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


    def single_model(
        title: str,
        time_window: int,
        time_series: pd.DataFrame,
        model: Any,
        test_size: int,
        val_size: int,
        return_option: Any,
        normalize: bool,
        horizon: int = 1,
        recursive: bool = False,
        use_exo_future: bool = True
    ) -> Dict:
        """
        Trains, predicts, and evaluates a single time series model using windowing.

        This function performs a complete pipeline:
        - Optional normalization (MinMaxScaler)
        - Conversion to supervised format (windowing / lag features)
        - Train/validation/test split
        - Model training
        - Prediction (standard or recursive)
        - Metric evaluation

        Parameters
        ----------
        title : str
            Identifier/name of the model (used for logging and results).

        time_window : int
            Number of lagged observations used as input features.

        time_series : pd.DataFrame
            Time series data. Must contain a column named 'actual'.
            Can also include exogenous variables (extra columns).

        model : Any
            Machine learning model (e.g., sklearn estimator).

        test_size : int
            Number of samples reserved for testing.

        val_size : int
            Number of samples reserved for validation.

        return_option : Any
            Defines which dataset (train/val/test) metrics should be returned.

        normalize : bool
            If True, applies MinMax scaling to the data (fit only on training set).

        horizon : int, optional (default = 1)
            Forecast horizon:
            - 1 → one-step ahead prediction
            - >1 → multi-step forecasting

        recursive : bool, optional (default = False)
            If True, performs recursive forecasting (step-by-step prediction).
            Not supported when using exogenous variables.

        use_exo_future : bool, optional (default = True)
            If True, uses future values of exogenous variables.
            Otherwise, uses only past values.

        Returns
        -------
        Dict
            A dictionary containing:
            - Evaluation metrics (RMSE, MAE, MAPE, etc.)
            - Real values (ground truth)
            - Predicted values
            - Model parameters
            - Optional additional outputs

        Notes
        -----
        - Avoids data leakage by fitting scalers only on training data
        - Converts time series into supervised format using lag features
        - Supports exogenous variables (additional predictors)
        - Supports recursive forecasting (except with exogenous data)
        - Designed to work with sklearn-like models
        """
        train_size = len(time_series) - test_size

        is_exogen = False

        if time_series.shape[1] > 1:
            is_exogen = True
            exogens = time_series.drop(
                columns=['actual', 'Data'], errors='ignore')
        horizon_to_use = horizon
        if recursive:
            if is_exogen:
                raise NotImplementedError(
                    'RECUSIVE IS NOT SUPPORTED WITH EXOGENS')
            horizon = 1

        # normalize
        if normalize:
            min_max_scaler = preprocessing.MinMaxScaler()
            min_max_scaler.fit(
                time_series['actual'].values[0:train_size].reshape(-1, 1))
            ts_normalized = min_max_scaler.transform(
                time_series['actual'].values.reshape(-1, 1))
            ts_normalized = pd.DataFrame({'actual': ts_normalized.flatten()})

            if is_exogen:
                min_max_scaler_x = preprocessing.MinMaxScaler()
                min_max_scaler_x.fit(exogens.values[0:train_size])
                exogens_norm = min_max_scaler_x.transform(exogens)
                exogens_norm = pd.DataFrame(
                    exogens_norm, columns=exogens.columns)

        else:
            ts_normalized = time_series
            if is_exogen:
                exogens_norm = exogens
        # ________________

        ts_windowed = FitPrediction.get_windowing(
            ts_normalized, time_window, horizon)
        if is_exogen:
            exgen_windowed = pd.DataFrame()
            for c in exogens.columns:
                if use_exo_future:
                    df_exogen = FitPrediction.get_windowing(
                        exogens_norm[c], time_window, horizon, f'_{c}')[['actual']]
                    df_exogen.rename(columns={'actual': c}, inplace=True)
                else:
                    df_exogen = FitPrediction.get_windowing(
                        exogens_norm[c], time_window, horizon, f'_{c}').drop(columns=['actual'])

                exgen_windowed = pd.concat([exgen_windowed, df_exogen], axis=1)

            ts_windowed = pd.concat([exgen_windowed, ts_windowed], axis=1)

        reg = tsf.fit_sklearn_model(ts_windowed, model, test_size, val_size)

        if recursive and (horizon_to_use > 1):
            ts_windowed_test = FitPrediction.get_windowing(
                ts_normalized, time_window, horizon_to_use)
            ts_windowed_test = ts_windowed_test.iloc[-(test_size+val_size+10):]
            pred = tsf.predict_sklearn_model_recursive(
                ts_windowed_test, reg, horizon_to_use)
        else:
            pred = tsf.predict_sklearn_model(ts_windowed, reg)

        if (normalize):
            pred = min_max_scaler.inverse_transform(
                pred.reshape(-1, 1)).flatten()

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
        use_exegen_future: bool,
        model_execs: int
    ) -> Dict:
        """
        Performs a manual grid search over hyperparameters for a time series model.

        This function evaluates multiple combinations of hyperparameters using
        a predefined parameter grid. For each combination, the model is trained
        multiple times and the average performance (RMSE) is used to select
        the best configuration.

        Parameters
        ----------
        real : pd.DataFrame
            Time series dataset. Must contain a column named 'actual' and optionally
            exogenous variables.

        test_size : int
            Number of samples reserved for testing.

        val_size : int
            Number of samples reserved for validation.

        parameters : Dict
            Dictionary defining the hyperparameter search space.
            Example:
                {
                    'max_depth': [5, 10],
                    'n_estimators': [100, 200],
                    'time_window': [12, 24]
                }

        model : Any
            Base machine learning model (e.g., sklearn estimator).

        horizon : int
            Forecast horizon (number of steps ahead to predict).

        recursive : bool
            If True, uses recursive forecasting strategy.

        use_exegen_future : bool
            If True, allows the use of future values of exogenous variables.

        model_execs : int
            Number of times each parameter configuration is executed.
            The final score is the average across executions.

        Returns
        -------
        Dict
            A dictionary containing:
            - 'best_result': dict with:
                - 'time_window': best lag size used
                - 'RMSE': best (lowest) average RMSE achieved
            - 'model': the best trained model (with optimal hyperparameters)

        Notes
        -----
        - Uses RMSE as the optimization metric (lower is better)
        - Performs a manual grid search (not sklearn GridSearchCV)
        - Repeats each experiment multiple times for robustness
        - Uses validation set (not test set) for model selection
        - `time_window` is treated separately from model hyperparameters
        - Does NOT perform true cross-validation (fixed split is used)
        """

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
                    True, 
                    horizon, 
                    recursive, 
                    use_exegen_future
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
        model: Any
    ) -> None:
        """
        Executes the full training pipeline for a sklearn-based model,
        including hyperparameter tuning via grid search and result persistence.

        This function acts as a high-level orchestrator that:
        - Iterates over hyperparameter combinations
        - Calls the grid search procedure
        - Trains models multiple times for robustness
        - Selects the best configuration
        - Saves results (e.g., metrics, predictions, model artifacts)

        Parameters
        ----------
        model_execs : int
            Number of times each model configuration is executed.
            Used to average performance and reduce randomness effects.

        data_title : str
            Identifier for the experiment (e.g., 'svr', 'mlp', 'rf').
            Typically used for naming output files and logs.

        parameters : Dict
            Dictionary defining the hyperparameter search space.
            Must include 'time_window' for time series lag configuration.

            Example:
                {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10],
                    'time_window': [12]
                }

        model : Any
            Machine learning model following sklearn interface
            (must implement fit() and predict()).

        Returns
        -------
        None
            This function does not return values directly.
            Instead, it saves results to disk (e.g., .pkl files),
            including:
            - Best model parameters
            - Evaluation metrics
            - Predictions
            - Real values

        Notes
        -----
        - Internally uses `do_grid_search` to find the best hyperparameters
        - Uses `single_model` to train and evaluate each configuration
        - Designed for time series forecasting with lag features
        - Supports multiple executions per configuration for stability
        - Typically used for experiments and batch model training
        """

        with open(f'{FitPrediction.CONFIG_PATH}models_config.json') as f:
            data = json.load(f)

        recursive = False
        use_exegen_future = False
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

                real = get_dataframe_by_station_and_pollutant(station_code=station_code, pollutant=pollutant, save_cv=False)

                gs_result = FitPrediction.do_grid_search(
                    real=real, 
                    test_size=test_size,
                    val_size=val_size,
                    parameters=parameters,
                    model=model,
                    horizon=horizon,
                    recursive=recursive,
                    use_exegen_future=use_exegen_future,
                    model_execs=model_execs
                )

                save_path_actual = FitPrediction.get_save_path_actual(type_data, title_temp)
                os.makedirs(save_path_actual, exist_ok=True)

                title_temp = FitPrediction.get_title_temp(type_data, title_temp)
                for _ in range(0, model_execs):
                    FitPrediction.single_model(
                        save_path_actual+title_temp, 
                        gs_result['best_result']['time_window'],
                        real,
                        gs_result['model'], 
                        test_size, 
                        val_size, 
                        tsf.result_options.save_result, 
                        True, 
                        horizon,
                        recursive, 
                        use_exegen_future
                    )
                    time.sleep(1)

    def get_save_path_actual(type_data, data_title:str):
        return FitPrediction.SAVE_PATH+str(type_data)+'-'+data_title+'/'
    
    def get_title_temp(type_data, data_title):
        return str(type_data) + '-' + data_title


class ArimaFitPrediction(FitPrediction):

    def train_arima(
        model_execs: int,
        data_title: str,
        auto: bool = True,
        parameters: Optional[Dict] = None,
        normalize: bool = False,
    ) -> None:
        """
        Training pipeline for ARIMA / AutoARIMA with optional normalization.

        Parameters
        ----------
        model_execs : int
            Number of executions per experiment.

        data_title : str
            Name for saving results (e.g., 'arima').

        parameters : dict, optional
            Parameters for auto_arima (optional).

        normalize : bool, default=False
            If True, applies MinMax scaling to the series (fit only on train).
        """

        with open(f'{FitPrediction.CONFIG_PATH}models_config.json') as f:
            data = json.load(f)

        for i in data:

            if i['activate'] == 1:

                # Config
                test_size = i['test_size']
                val_size = i['val_size']
                type_data = i['type_data']
                horizon = i['horizon']

                pollutant = i['pollutant']
                station_code = i['station_code']

                # Load data
                real = get_dataframe_by_station_and_pollutant(
                    station_code=station_code,
                    pollutant=pollutant
                )

                ts = real['actual']

                # Split
                train = ts[:-test_size]
                # test = ts[-test_size:]

                save_path_actual = FitPrediction.get_save_path_actual(type_data, data_title)
                os.makedirs(save_path_actual, exist_ok=True)

                title_temp = FitPrediction.get_title_temp(type_data, data_title)

                for _ in range(model_execs):

                    if normalize:
                        scaler = preprocessing.MinMaxScaler()
                        train_used = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
                    else:
                        train_used = train.values

                    model = ArimaFitPrediction.build_arima(train_used, parameters, auto)

                    train_size = len(ts) - (val_size + test_size)

                    y_pred_full = ArimaFitPrediction.walk_forward_arima(
                        ts=ts,
                        train_size=train_size,
                        val_size=val_size,
                        test_size=test_size,
                        normalize=normalize,
                        scaler=scaler if normalize else None,
                        parameters=parameters,
                        horizon=horizon
                    )

                    # EVALUATION
                    tsf.make_metrics_avaliation(
                        y_true=ts,
                        y_pred=y_pred_full,
                        test_size=test_size,
                        val_size=val_size,
                        return_type=tsf.result_options.save_result,
                        model_params={'order': model.order},
                        title=save_path_actual+title_temp,
                        prevs_df=None
                    )

                    time.sleep(1)


    def walk_forward_arima(
        ts,
        train_size,
        val_size,
        test_size,
        normalize=False,
        scaler=None,
        parameters=None,
        horizon=1
    ):
        # -----------------------------
        # Preparação dos dados
        # -----------------------------
        if normalize:
            full_series = scaler.transform(ts.values.reshape(-1, 1)).flatten()
            train_used = full_series[:train_size]
        else:
            full_series = ts.values
            train_used = full_series[:train_size]

        history = list(train_used)

        preds = []

        total_steps = val_size + test_size
        start_index = train_size

        model = auto_arima(
            history,
            **(parameters or {})
        )

        # -----------------------------
        # Walk-forward
        # -----------------------------
        for i in range(total_steps):

            # pred 1 step
            yhat = model.predict(n_periods=horizon)[0]
            preds.append(yhat)
            real_value = full_series[start_index + i]
            model.update(real_value)

        if normalize:
            preds = scaler.inverse_transform(
                np.array(preds).reshape(-1, 1)
            ).flatten()

        y_pred_full = np.full(len(ts), np.nan)
        y_pred_full[-(val_size + test_size):] = preds

        return y_pred_full
    

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