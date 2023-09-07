from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import tratamento_dados

st.set_page_config(
    page_title="POKECODE",
    page_icon="assets\icons\logo1.png",
    initial_sidebar_state="collapsed",
)

with open('assets/css/style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

current_path = Path(__file__).resolve().parent.parent
file_path = current_path / "data" / "pokemon.parquet"

pokemon_df = pd.read_parquet(file_path)
pokemon_df['image'] = ''
pokemon_df['image'] = pokemon_df['pokedex_number'].apply(
    lambda x: f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{x}.png')

pokemon_features = tratamento_dados.tratar_df()
pokemon_features_clusters, _, _ = tratamento_dados.clusterizar_df()
pokemon_df['image'] = pokemon_df['pokedex_number'].apply(
    lambda x: f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{x}.png')

selected_columns = ['typing', 'hp', 'speed', 'height', 'weight', 'shape', 'primary_color']
pokemon_features = pokemon_df[selected_columns].copy()


encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_columns = pd.DataFrame(encoder.fit_transform(pokemon_features[['typing', 'shape', 'primary_color']]))
encoded_columns.columns = encoder.get_feature_names_out(['typing', 'shape', 'primary_color'])

pokemon_features = pd.concat([pokemon_features, encoded_columns], axis=1)
pokemon_features.drop(columns=['typing', 'shape', 'primary_color'], inplace=True)

scaler = StandardScaler()
pokemon_features[['hp', 'speed', 'height', 'weight']] = scaler.fit_transform(
    pokemon_features[['hp', 'speed', 'height', 'weight']])



# Coeficiente de silhueta
silhouette_avg = silhouette_score(pokemon_features_clusters, pokemon_features_clusters['cluster_label'])
silhouette_values = silhouette_samples(pokemon_features_clusters, pokemon_features_clusters['cluster_label'])
pokemon_features['silhouette_score'] = silhouette_values
#st.write(f"Coeficiente de Silhueta Médio no KNN com cluster: {silhouette_avg}")

k_neighbors = 20
knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
knn_model.fit(pokemon_features)

knn_modelC = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
knn_modelC.fit(pokemon_features_clusters)

# MAIN PAGE START --
st.image("assets\icons\logo2.png")
st.markdown('<h1 class="site-title">Sistema de Recomendação</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="site-subt">Implementação de Machine Learning', unsafe_allow_html=True)

st.markdown('<p class="site-subt"><b>K-Nearest Neighbors Puro</b></p>', unsafe_allow_html=True)

with st.expander("Recomendações de Pokémon"):
    pokemon_choose = st.selectbox('Escolha um Pokémon', pokemon_df['name'], help='Selecione um Pokémon que você gosta')

    if pokemon_choose:
        selected_pokemon_index = pokemon_df[pokemon_df['name'] == pokemon_choose].index[0]
        distances, indices = knn_model.kneighbors(pokemon_features.iloc[selected_pokemon_index].values.reshape(1, -1))
        st.write(distances)
        st.subheader("Pokémon semelhantes:")
        if pokemon_choose:
            selected_pokemon_index = pokemon_df[pokemon_df['name'] == pokemon_choose].index[0]
            distances, indices = knn_model.kneighbors(
                pokemon_features.iloc[selected_pokemon_index].values.reshape(1, -1))

            colunas = st.columns(10)
            for i in range(10):
                with colunas[i]:
                    st.header(f"{i + 1}º")
                    st.image(pokemon_df.loc[indices[0][i], 'image'], caption=pokemon_df.loc[indices[0][i], 'name'],
                             width=100)

st.markdown('<p class="site-subt"><b>K-Nearest Neighbors com Clusterização</b></p>', unsafe_allow_html=True)
with st.expander("Recomendações de Pokémon"):
    pokemon_choose = st.selectbox('Selecione um Pokémon', pokemon_df['name'],
                                  help='Selecione um Pokémon que você gosta')

    if pokemon_choose:
        selected_pokemon_index = pokemon_df[pokemon_df['name'] == pokemon_choose].index[0]
        distances, indices = knn_modelC.kneighbors(
            pokemon_features_clusters.iloc[selected_pokemon_index].values.reshape(1, -1))
        st.write(distances)
        st.subheader("Pokémon semelhantes:")
        colunas = st.columns(10)

        for i in range(10):
            with colunas[i]:
                st.header(f"{i + 1}º")
                st.image(pokemon_df.loc[indices[0][i], 'image'], caption=pokemon_df.loc[indices[0][i], 'name'],
                         width=100)


st.markdown('<p class="site-subt"><b>DBSCAN</b></p>', unsafe_allow_html=True)

with st.expander("Recomendações de Pokémon"):
    st.write("DBSCAN")
    #Colocar aqui o Funcionamento do DBSCAN no mesmo estilo das outros algoritmos
    
st.markdown('<h3 class="site-subt"><b>Comparação de Algoritmos</b></h3>', unsafe_allow_html=True)
# Escrever resumo com informações sobre as métricas utilizadas

st.markdown('<p class="site-subt"><b>K-Nearest Neighbors Puro x K-Nearest Neighbors com Clusterização</b></p>', unsafe_allow_html=True)
with st.expander("KNN Puro x KNN Cluster"):
    st.write("")
    # Colocar aqui a comparação
    
st.markdown('<p class="site-subt"><b>K-Nearest Neighbors com Clusterização x DBSCAN</b></p>', unsafe_allow_html=True)
with st.expander("KNN Cluster x DBSCAN"):
    st.write("")
    # Colocar aqui a comparação
    