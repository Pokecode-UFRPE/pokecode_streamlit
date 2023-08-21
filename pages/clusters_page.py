from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px


current_path = Path(__file__).resolve().parent.parent
file_path = current_path / "data" / "pokemon.csv"
pokemon_df = pd.read_csv(file_path)

st.subheader("Testes de clusters")
# CODIGO QUE RETIRA OS DUPICADAS DE NÚMERO DA POKEDEX
pokemon_df = pokemon_df.drop_duplicates(subset='pokedex_number')

# selecionando os atributos para clusterização
attributes = ['attack', 'defense', 'hp','speed','height','weight']

# Criando o modelo K-means com 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(pokemon_df[attributes])

pokemon_df['cluster_label'] = kmeans.labels_

graph_clusters = px.scatter(pokemon_df, x='attack', y='defense',color='cluster_label',
                 title='Gráfico de Dispersão com Clusters')



st.plotly_chart(graph_clusters)