import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from data import tratamento_dados

pokemon_df = pd.read_parquet('data/pokemon.parquet')
pokemon_df['image'] = ''
pokemon_df['image'] = pokemon_df['pokedex_number'].apply(lambda x: f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{x}.png')

pokemon_features = tratamento_dados.tratar_df()
pokemon_features_clusters, _, _ = tratamento_dados.clusterizar_df()

k_neighbors = 4
knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
knn_model.fit(pokemon_features)

knn_modelC = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
knn_modelC.fit(pokemon_features_clusters)


st.header("Teste de ML")
st.subheader("Implementando aprendizado por semelhança")
    
with st.expander("Buscar recomendações por um Pokémon"):
    pokemon_choose = st.selectbox('Escolha um Pokémon', pokemon_df['name'], help='Selecione um Pokémon que você gosta')

    if pokemon_choose:
        selected_pokemon_index = pokemon_df[pokemon_df['name'] == pokemon_choose].index[0]
        distances, indices = knn_model.kneighbors(pokemon_features.iloc[selected_pokemon_index].values.reshape(1, -1))

        st.subheader("Pokémon semelhantes de acordo com:")
        
        st.image(pokemon_df.iloc[selected_pokemon_index]['image'], caption=pokemon_df.iloc[selected_pokemon_index]['name'])        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("1º")
            st.image(pokemon_df.loc[indices[0][1], 'image'], caption=pokemon_df.loc[indices[0][1], 'name'], width=100)

        with col2:
            st.header("2º")
            st.image(pokemon_df.loc[indices[0][2], 'image'], caption=pokemon_df.loc[indices[0][2], 'name'], width=100)

        with col3:
            st.header("3º")
            st.image(pokemon_df.loc[indices[0][3], 'image'], caption=pokemon_df.loc[indices[0][3], 'name'], width=100)

with st.expander("Buscar recomendações por um Pokémon com cluster"):
    pokemon_choose = st.selectbox('Selecione um Pokémon', pokemon_df['name'], help='Selecione um Pokémon que você gosta')

    if pokemon_choose:
        selected_pokemon_index = pokemon_df[pokemon_df['name'] == pokemon_choose].index[0]
        distances, indices = knn_modelC.kneighbors(pokemon_features_clusters.iloc[selected_pokemon_index].values.reshape(1, -1))

        st.subheader("Pokémon semelhantes de acordo com:")
        
        st.image(pokemon_df.iloc[selected_pokemon_index]['image'], caption=pokemon_df.iloc[selected_pokemon_index]['name'])        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("1º")
            st.image(pokemon_df.loc[indices[0][1], 'image'], caption=pokemon_df.loc[indices[0][1], 'name'], width=100)

        with col2:
            st.header("2º")
            st.image(pokemon_df.loc[indices[0][2], 'image'], caption=pokemon_df.loc[indices[0][2], 'name'], width=100)

        with col3:
            st.header("3º")
            st.image(pokemon_df.loc[indices[0][3], 'image'], caption=pokemon_df.loc[indices[0][3], 'name'], width=100)
