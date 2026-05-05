import pandas as pd
import matplotlib.pyplot as plt

# carregar o arquivo
# df = pd.read_csv("Dados_SP_AR_Qualidade.csv")

#Juntando os dados dos CSVs
import glob

from pmdarima import plot_acf, plot_pacf

def count_stations():
    # pegar estações únicas (2ª coluna)
    # estacoes = df.iloc[:, 1].dropna().unique().tolist()

    # print("Total de estações:", len(estacoes))
    # print(estacoes)
    pass

def rank_missing_by_time_pollutant(df, freq='D', poluente_filtro=None, top_n=10):
    df = df.copy()
    df['Data'] = pd.to_datetime(df['Data'])

    # 🔥 filtra o poluente se for passado
    if poluente_filtro is not None:
        df = df[df['Poluente'] == poluente_filtro]

    results = []

    for (codigo, poluente), group in df.groupby(['Codigo', 'Poluente']):
        group = group.sort_values('Data')

        unique_dates = group['Data'].dt.floor(freq).nunique()

        full_range = pd.date_range(
            start=group['Data'].min().floor(freq),
            end=group['Data'].max().floor(freq),
            freq=freq
        )

        expected = len(full_range)
        missing = expected - unique_dates

        results.append({
            'Codigo': codigo,
            'Poluente': poluente,
            'expected': expected,
            'actual': unique_dates,
            'missing': missing,
            'missing_pct': missing / expected if expected > 0 else 0
        })

    result_df = pd.DataFrame(results)

    # 🔥 melhores = menor percentual de missing
    result_df = result_df.sort_values('missing_pct', ascending=True).head(top_n)

    return result_df


def get_dataframe_by_station_and_pollutant(station_code: str, pollutant: str, save_cv: bool = False):

    files = glob.glob("dataset/IEMA/SP*.csv")

    dfs = []

    for file in files:
        df = pd.read_csv(file, encoding="latin1")
        dfs.append(df)

    df_final = pd.concat(dfs, ignore_index=True)

    dataframe = df_final[
        (df_final["Codigo"] == station_code) &
        (df_final["Poluente"] == pollutant)
    ].copy()

    dataframe["Hora"] = dataframe["Hora"].replace("24:00", "00:00")

    dataframe["datetime"] = pd.to_datetime(
        dataframe["Data"] + " " + dataframe["Hora"],
        errors="coerce"
    )

    #Removing unecessary columns
    dataframe = dataframe.drop(columns=["Data", "Hora", "Codigo", "Tipo", "Unidade", "Poluente", "Estacao"])

    #Taking the time as index
    dataframe = dataframe.sort_values("datetime").reset_index(drop=True)
    dataframe = dataframe.set_index("datetime")[["Valor"]]
    dataframe = dataframe.rename(columns={"Valor": "actual"})
    
    #Retrieving the day average
    dataframe = dataframe.resample("D").mean()

    print(dataframe)
    #Interpolation null values with akima
    dataframe = dataframe.interpolate(method="akima")
    dataframe = dataframe.clip(lower=0)

    if save_cv:
        dataframe.to_csv(f"pollution_{station_code}_{pollutant}.csv")

    return dataframe

def print_example(station_code, pollutant):
    dataframe = get_dataframe_by_station_and_pollutant(station_code=station_code, pollutant=pollutant)

    plt.figure(figsize=(12,6))

    plt.plot(dataframe.index, dataframe)
    print("Null values:", dataframe.isna().sum())

    plt.title(f"{pollutant} - {station_code}")
    plt.xlabel("Data")
    plt.ylabel(f"{pollutant} (µg/m³)")

    plt.show()

from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


if __name__ == "__main__":
    from statsmodels.tsa.seasonal import seasonal_decompose  
    # pollutant = "NO2" #Has
    # pollutant = "CO" #Has
    pollutant = "NO2"
    # pollutant = "MP10"
    # pollutant = "MP2.5"
    # pollutant = "O3"
    station_code = "SP37"

    print_example(station_code, pollutant)

    #Entry
    df = get_dataframe_by_station_and_pollutant(station_code, pollutant)

    ts = df['actual']

    import numpy as np
    from statsmodels.tsa.stattools import acf

    acf_vals = acf(ts, nlags=100)

    # pega os maiores picos (ignorando lag 0)
    lags = np.argsort(acf_vals[1:])[-5:] + 1
    print("lags",lags)

    result = seasonal_decompose(ts, model='add')
    result.plot()

    from statsmodels.stats.diagnostic import acorr_ljungbox

    # Se p-valor > 0.05, resíduos são ruído branco (série difícil de prever)
    result = acorr_ljungbox(ts, lags=[10, 20], return_df=True)
    print(result)

    adf_test(ts, "Pollution")

    plot_acf(df, lags=40)
    plot_pacf(df, lags=40)