import pandas as pd
import matplotlib.pyplot as plt

# carregar o arquivo
# df = pd.read_csv("Dados_SP_AR_Qualidade.csv")

#Juntando os dados dos CSVs
import glob

def count_stations():
    # pegar estações únicas (2ª coluna)
    # estacoes = df.iloc[:, 1].dropna().unique().tolist()

    # print("Total de estações:", len(estacoes))
    # print(estacoes)
    pass

def get_dataframe_by_station_and_pollutant(station_code: str, pollutant: str):

    files = glob.glob("dataset/IEMA/SP*.csv")

    files = [
        f for f in files
        if any(year in f for year in ["2018", "2019", "2020"])
    ]

    dfs = []

    for file in files:
        df = pd.read_csv(file, encoding="latin1")
        dfs.append(df)

    df_final = pd.concat(dfs, ignore_index=True)

    dataframe = df_final[
        (df_final["Codigo"] == station_code) &
        (df_final["Poluente"] == pollutant)
    ]

    dataframe["Hora"] = dataframe["Hora"].replace("24:00", "00:00")

    dataframe["datetime"] = pd.to_datetime(
        dataframe["Data"] + " " + dataframe["Hora"],
        errors="coerce"
    )

    #Removing unecessary columns
    dataframe = dataframe.drop(columns=["Data", "Hora", "Codigo", "Tipo", "Unidade", "Poluente", "Estacao"])

    #Taking the time as index
    dataframe = dataframe.sort_values("datetime").reset_index(drop=True)
    dataframe = dataframe.set_index("datetime")["Valor"]
    
    #Retrieving the day average
    dataframe = dataframe.resample("D").mean()

    print(dataframe)

    return dataframe

def print_example():
    dataframe = get_dataframe_by_station_and_pollutant(station_code="SP71", pollutant="MP10")
    print(dataframe)

    plt.figure(figsize=(12,6))

    plt.plot(dataframe["datetime"], dataframe["Valor"])

    plt.title("MP10 - São José dos Campos - Vila Santa Maria")
    plt.xlabel("Data")
    plt.ylabel("MP10 (µg/m³)")

    plt.show()

