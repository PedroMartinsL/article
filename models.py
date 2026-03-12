from pmdarima.arima import auto_arima
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import pandas as pd
from extractor import get_dataframe_by_station_and_pollutant
import time_series_functions as tsf
import matplotlib.pyplot as plt

#Code retrieved from https://github.com/domingos108/time_series_functions

def mean_square_error(y_true, y_pred):
    y_true = np.asmatrix(y_true).reshape(-1)
    y_pred = np.asmatrix(y_pred).reshape(-1)

    return np.square(np.subtract(y_true, y_pred)).mean()

def root_mean_square_error(y_true, y_pred):

    return mean_square_error(y_true, y_pred)**0.5


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(np.where(y_true == 0)[0]) > 0:
        return np.inf
    else:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.mean(np.abs(y_true - y_pred))

def create_windowing(df, lag_size):
    final_df = None
    for i in range(0, (lag_size + 1)):
        serie = df.shift(i)
        if (i == 0):
            serie.columns = ['actual']
        else:
            serie.columns = [str('lag' + str(i))]
        final_df = pd.concat([serie, final_df], axis=1)

    return final_df.dropna()

def gerenerate_metric_results(y_true, y_pred):
    return {'MSE': mean_square_error(y_true, y_pred),
            'RMSE':root_mean_square_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
    }

def get_windowing(ts_normalized , time_window, horizon , prefix=''):
    ts_windowed = create_windowing(lag_size=(time_window + (horizon-1)),
                                        df=ts_normalized)

    columns_lag = [f'lag_{l}{prefix}'for l in reversed(range(1,time_window+1))]
    columns_horizon = [f'hor_{l}{prefix}'for l in range(1,horizon)] + ['actual']
    ts_windowed.columns= columns_lag + columns_horizon

    ts_windowed = ts_windowed[columns_lag+['actual']]
    return ts_windowed

def extract_arima_preview(ts, test_size):
    ts_train = ts.iloc[:-test_size]
    ts_test  = ts.iloc[-test_size:]
    arima_model = auto_arima(
        ts_train,
        seasonal=True,
        m=12,          # sazonalidade mensal
        trace=False,
        suppress_warnings=True
    )
    arima_prevs = []

    for t in ts_test:
        pred = arima_model.predict(n_periods=1).item()
        arima_prevs.append(pred)
        arima_model.update(t)

    return np.array(arima_prevs)

def extract_svr(x_train, y_train):
    model = SVR(C=1000, max_iter=100000, epsilon = 0.001) # falta encontrar params
    model.fit(x_train, y_train)

    prevs = model.predict(x_test)
    return min_max_scaler.inverse_transform(prevs.reshape(-1, 1)).flatten()

def extract_mlp(x_train, x_test, y_train):
    mlp = MLPRegressor(
        hidden_layer_sizes=(100,),
        max_iter=500,
        random_state=42
    )

    mlp.fit(x_train, y_train)

    # Previsões normalizadas
    preds_norm = mlp.predict(x_test)

    # Converter para escala real
    preds_real = min_max_scaler.inverse_transform(
        preds_norm.reshape(-1, 1)
    ).flatten()

    return preds_real

def extract_random_forest_regressor(x_train, x_test, y_train):
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    rf.fit(x_train, y_train)
    
    rf = rf.predict(x_test)

    preds_real = min_max_scaler.inverse_transform(
            rf.reshape(-1, 1)
        ).flatten()

    return preds_real

def extract_gradient_boosting(x_train, x_test, y_train):
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

    gb.fit(x_train, y_train)
    gb = gb.predict(x_test)

    preds_real = min_max_scaler.inverse_transform(
            gb.reshape(-1, 1)
        ).flatten()

    return preds_real

#-------------------------------
pollutant = "MP10"
ts = get_dataframe_by_station_and_pollutant(station_code="SP71", pollutant=pollutant)

#Interpolate values
ts = ts.interpolate()

time_window = 12
horizon = 1
test_size = 180 #3 Months

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(ts.values[0:-test_size].reshape(-1, 1))

ts_normalized = min_max_scaler.transform(ts.values.reshape(-1, 1))
ts_normalized = pd.DataFrame({'actual': ts_normalized.flatten()})

df_wind = get_windowing(ts_normalized , time_window, horizon , prefix='')

x_train = df_wind.iloc[0:-test_size].drop(columns=['actual'])
y_train = df_wind.iloc[0:-test_size]['actual']
x_test = df_wind.iloc[-test_size:].drop(columns=['actual'])
y_test = df_wind.iloc[-test_size:]['actual']

svr_pred = extract_svr(x_train=x_train, y_train=y_train)
arima_pred = extract_arima_preview(ts, test_size)
mlp_pred = extract_mlp(x_test=x_test,x_train=x_train,y_train=y_train)
rf_pred = extract_random_forest_regressor(x_test=x_test,x_train=x_train,y_train=y_train)
gb_pred = extract_gradient_boosting(x_test=x_test,x_train=x_train,y_train=y_train)

print("MLP:", gerenerate_metric_results(y_test, mlp_pred))
print("SVR:", gerenerate_metric_results(y_test, svr_pred))
print("ARIMA:", gerenerate_metric_results(y_test, arima_pred))
print("RF:", gerenerate_metric_results(y_test, rf_pred))
print("GB:", gerenerate_metric_results(y_test, gb_pred))

ensemble_pred = np.mean(
    [mlp_pred, svr_pred, arima_pred, rf_pred, gb_pred],
    axis=0
)

print("Ensemble Predictions: ", ensemble_pred)

y_test_real = min_max_scaler.inverse_transform(
    y_test.values.reshape(-1, 1)
).flatten()

plt.figure(figsize=(12,6))

plt.plot(y_test_real, label="Real", linewidth=2)
plt.plot(mlp_pred, label="MLP")
plt.plot(svr_pred, label="SVR")
plt.plot(arima_pred, label="ARIMA")
plt.plot(rf_pred, label="Random Forest")
plt.plot(gb_pred, label="Gradient Boosting")
plt.plot(ensemble_pred, label="Ensemble", linestyle="--", linewidth=2)

plt.title("Models Comparison")
plt.xlabel("Time Step - D")
plt.ylabel(pollutant)
plt.legend()

plt.show()