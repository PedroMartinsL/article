import time
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from services.fit_predict import FitPrediction
from extractor import get_dataframe_by_station_and_pollutant
from services import time_series_functions as tsf
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss

class NeuralForecastFitPrediction:

    def run_single_exec(
            df: pd.DataFrame,
            model,
            horizon: int,
            test_size: int,
            differencing: bool,
            param_set: dict,
            save_path: str,
            title_temp: str,
            return_option: Any,
        ):

        df_used = df.copy()

        if differencing:
            df_used['y_orig'] = df_used['y']
            df_used['y'] = df_used['y'].diff()
            df_used = df_used.dropna().reset_index(drop=True)

        try:
            model_instance = model(
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                h=horizon,
                loss=DistributionLoss(distribution='StudentT'),
                **param_set
            )
        except TypeError:
            raise Exception("Invalid model")

        fcst = NeuralForecast(models=[model_instance], freq='D')

        cv_df = fcst.cross_validation(
            df=df_used,
            n_windows=max(1, test_size // horizon),
            step_size=horizon
        )

        model_name = model_instance.__class__.__name__

        full_y = df_used['y'].values
        full_pred = np.full(len(full_y), np.nan)
        full_ds = df_used['ds'].values

        cv_last = cv_df.groupby('ds')[f'{model_name}-median'].last().reset_index()

        for _, row in cv_last.iterrows():
            idx = np.where(full_ds == row['ds'])[0]
            if len(idx) > 0:
                full_pred[idx[0]] = row[f'{model_name}-median']

        full_y_masked = full_y
        full_pred_masked = full_pred

        if differencing:
            # precisamos alinhar com série original
            original_y = df['y'].values

            # índice correspondente (shift de 1 por causa do diff)
            valid_idx = np.where(full_pred_masked)[0] + 1

            base = original_y[valid_idx - 1]

            full_pred_masked = base + full_pred_masked
            full_y_masked = original_y[valid_idx]

        response = tsf.make_metrics_avaliation(
            y_true=full_y_masked,
            y_pred=full_pred_masked,
            test_size=test_size,
            val_size=0,
            return_type=return_option,
            model_params=param_set,
            title=save_path + title_temp,
            prevs_df=cv_df
        )

        print("response \n\n", response)

        return response


    def do_grid_search(
        df: pd.DataFrame,
        model,
        model_execs: int,
        horizon: int,
        test_size: int,
        differencing: bool,
        param_grid: list,
        save_path: str,
        title_temp: str,
    ) -> dict:
        metric = 'RMSE'
        best_param_set = None
        best_result = {metric: None}
        result_type = tsf.result_options.test_result

        for param_set in tqdm(param_grid, desc='GridSearch'):

            result_atual = []

            for _ in range(model_execs):
                response = NeuralForecastFitPrediction.run_single_exec(
                    df=df,
                    model=model,
                    horizon=horizon,
                    test_size=test_size,
                    differencing=differencing,
                    return_option=result_type,
                    param_set=param_set,
                    save_path=save_path,
                    title_temp=title_temp,
                )
                result_atual.append(response[metric])

            result = np.mean(np.array(result_atual))

            if best_result[metric] is None:
                best_result[metric] = result
                best_param_set = param_set
            else:
                if best_result[metric] > result:
                    best_result[metric] = result
                    best_param_set = param_set

        print(f'Best params: {best_param_set} | {metric}: {best_result[metric]:.4f}')

        result_model = {'best_result': best_result, 'param_set': best_param_set}
        return result_model


    def train_neuralforecast(
        model_execs: int,
        data_title: str,
        parameters: dict,
        model,
        differencing: bool = False
    ):
        import itertools

        with open(f'{FitPrediction.CONFIG_PATH}models_config.json') as f:
            data = json.load(f)

        keys, values = zip(*parameters.items())
        param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for cfg in data:

            if cfg['activate'] != 1:
                continue

            test_size = cfg['test_size']
            horizon = cfg['horizon']
            type_data = cfg['type_data']
            pollutant = cfg['pollutant']
            station_code = cfg['station_code']

            real = get_dataframe_by_station_and_pollutant(
                station_code=station_code,
                pollutant=pollutant
            )

            ts = real['actual'].reset_index(drop=True)

            df = pd.DataFrame({
                'unique_id': 'series_1',
                'ds': real.index[:len(ts)],
                'y': ts.values
            })

            save_path = FitPrediction.get_save_path_actual(type_data, data_title)
            os.makedirs(save_path, exist_ok=True)
            title_temp = FitPrediction.get_title_temp(type_data, data_title)

            best = NeuralForecastFitPrediction.do_grid_search(
                df=df,
                model=model,
                model_execs=model_execs,
                horizon=horizon,
                test_size=test_size,
                differencing=differencing,
                param_grid=param_grid,
                save_path=save_path,
                title_temp=title_temp,
            )
            best_param_set = best['param_set']

            for _ in range(model_execs):
                NeuralForecastFitPrediction.run_single_exec(
                    df=df,
                    model=model,
                    horizon=horizon,
                    test_size=test_size,
                    differencing=differencing,
                    return_option=tsf.result_options.save_result,
                    param_set=best_param_set,
                    save_path=save_path,
                    title_temp=title_temp,
                )
                time.sleep(1)
