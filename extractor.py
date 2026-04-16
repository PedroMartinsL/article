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

def print_example():
    dataframe = get_dataframe_by_station_and_pollutant(station_code="SP71", pollutant="MP10")
    print(dataframe)

    plt.figure(figsize=(12,6))

    plt.plot(dataframe.index, dataframe)
    print("Null values:", dataframe.isna().sum())

    plt.title("MP10 - São José dos Campos - Vila Santa Maria")
    plt.xlabel("Data")
    plt.ylabel("MP10 (µg/m³)")

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
    pollutant = "MP10"
    station_code = "SP71"

    #Entry
    df = get_dataframe_by_station_and_pollutant(station_code, pollutant)

    ts = df['actual']

    result = seasonal_decompose(ts, model='add')
    result.plot()

    # from statsmodels.stats.diagnostic import acorr_ljungbox

    # resultado = acorr_ljungbox(ts, lags=[10], return_df=True)
    # print(resultado)

    adf_test(ts, "Pollution")

    plot_acf(df, lags=40)
    plot_pacf(df, lags=40)