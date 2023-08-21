from pathlib import Path

import streamlit as st
import pandas as pd

current_path = Path(__file__).resolve().parent.parent
file_path = current_path / "data" / "pokemon.csv"
pokemon_df = pd.read_csv(file_path)

st.subheader("Testes de Classificação")