import streamlit as st
import pandas as pd
from data import tratamento_dados
import numpy as np
from pathlib import Path

current_path = Path(__file__).resolve().parent.parent
parquet = str(current_path / "data" / "pokemon.parquet")
logo1 = current_path / "assets" / "icons" / "logo1.png"
css_path = str(current_path / "assets" / "css" / "style.css")

st.set_page_config(
    page_title="POKECODE",
    page_icon="assets/icons/logo1.png",
    initial_sidebar_state="collapsed",
)

with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

pokemon_df = pd.read_parquet("data/pokemon.parquet")

st.image("assets/icons/logo2.png")
st.markdown('<h1 class="site-title">Ánalise Comparativa</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="site-subt">Escolha dois Pokémons para comparar:</h3>', unsafe_allow_html=True)

# caixas para selecionar pokémon 1 e 2
pokemon_a = st.selectbox('Escolha o primeiro Pokémon', pokemon_df['name'])
pokemon_b = st.selectbox('Escolha o segundo Pokémon', pokemon_df['name'])
pokemon_features = tratamento_dados.tratar_df()

if st.button('Calcular Distância Euclidiana'):
    # pegar dados dos 2 pokémon comparados e armazená-los em listas
    pokemon1 = pokemon_features[pokemon_df['name'] == pokemon_a]
    pokemon1 = pokemon1.values.tolist()
    pokemon1 = pokemon1[0]
    pokemon2 = pokemon_features[pokemon_df['name'] == pokemon_b]
    pokemon2 = pokemon2.values.tolist()
    pokemon2 = pokemon2[0]

    # chamar função de distancia euclediana e printar na tela
    distancia = tratamento_dados.distancia_euclediana(pokemon1,pokemon2)
    st.text(distancia)
