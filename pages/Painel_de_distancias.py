import streamlit as st
import pandas as pd
from data import tratamento_dados
import numpy as np
pokemon_df = pd.read_parquet("data/pokemon.parquet")


# caixas para selecionar pokémon 1 e 2
pokemon_a = st.selectbox('Escolha um Pokémon', pokemon_df['name'], help='Selecione um Pokémon que você gosta')
pokemon_b = st.selectbox('Escolha um Pokémon', pokemon_df['name'], help='Selecione outro Pokémon que você gosta')
pokemon_features = tratamento_dados.tratar_df()
# pegar dados dos 2 pokémon comparados e armazenalos em listas
pokemon1 = pokemon_features[pokemon_df['name'] == pokemon_a]
pokemon1 = pokemon1.values.tolist()
pokemon1 = pokemon1[0]
pokemon2 = pokemon_features[pokemon_df['name'] == pokemon_b]
pokemon2 = pokemon2.values.tolist()
pokemon2 = pokemon2[0]

#chamar função de distancia euclediana e printar na tela
distancia = tratamento_dados.distancia_euclediana(pokemon1,pokemon2)
st.text(distancia)
