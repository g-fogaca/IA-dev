# importando bibliotecas
import pandas as pd

#importando dados
music_df = pd.read_csv("music_clean.csv")

# Criando variáveis dummy
music_dummies = pd.get_dummies(music_df["genre"], drop_first = True)

# Concatenando
music_dummies = pd.concat([music_df, music_dummies], axis=1)

# Removendo repetição
music_dummies = music_dummies.drop("genre", axis=1)
