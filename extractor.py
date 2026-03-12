import pandas as pd
import matplotlib.pyplot as plt

# carregar o arquivo
# df = pd.read_csv("Dados_SP_AR_Qualidade.csv")

#Juntando os dados dos CSVs
import glob

files = glob.glob("dataset/IEMA/SP*02.csv")
print(files)

dfs = []

for file in files:
    df = pd.read_csv(file, encoding="latin1")
    dfs.append(df)

df_final = pd.concat(dfs, ignore_index=True)

# pegar estações únicas (2ª coluna)
# estacoes = df.iloc[:, 1].dropna().unique().tolist()

# print("Total de estações:", len(estacoes))
# print(estacoes)

dataframe = df_final[
    (df_final["Codigo"] == "SP71") &
    (df_final["Poluente"] == "MP10")
]

dataframe["Hora"] = dataframe["Hora"].replace("24:00", "00:00")

dataframe["datetime"] = pd.to_datetime(
    dataframe["Data"] + " " + dataframe["Hora"],
    errors="coerce"
)

dataframe = dataframe.drop(columns=["Data", "Hora", "Codigo", "Tipo", "Unidade"])

dataframe = dataframe.sort_values("datetime").reset_index(drop=True)

plt.figure(figsize=(12,6))

plt.plot(dataframe["datetime"], dataframe["Valor"])

plt.title("MP10 - São José dos Campos - Vila Santa Maria")
plt.xlabel("Data")
plt.ylabel("MP10 (µg/m³)")

plt.show()

