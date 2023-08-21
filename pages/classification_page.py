import streamlit as st
import pandas as pd

pokemon_df = pd.read_csv("data/pokemon.csv")

st.subheader("Testes de Classificação")