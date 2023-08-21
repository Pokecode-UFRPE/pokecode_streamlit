from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder

current_path = Path(__file__).resolve().parent.parent
file_path = current_path / "data" / "pokemon.parquet"

pokemon_df = pd.read_parquet(file_path)
pokemon_df['image'] = ''
pokemon_df['image'] = pokemon_df['pokedex_number'].apply(
    lambda x: f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{x}.png')

selected_columns = ['typing', 'hp', 'speed', 'height', 'weight', 'shape', 'primary_color']
pokemon_features = pokemon_df[selected_columns].copy()

# st.write(pokemon_df)

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_columns = pd.DataFrame(encoder.fit_transform(pokemon_features[['typing', 'shape', 'primary_color']]))
encoded_columns.columns = encoder.get_feature_names_out(['typing', 'shape', 'primary_color'])

pokemon_features = pd.concat([pokemon_features, encoded_columns], axis=1)
pokemon_features.drop(columns=['typing', 'shape', 'primary_color'], inplace=True)

scaler = StandardScaler()
pokemon_features[['hp', 'speed', 'height', 'weight']] = scaler.fit_transform(
    pokemon_features[['hp', 'speed', 'height', 'weight']])
# st.write(pokemon_features)


st.header("Teste de ML")
st.subheader("Implementando aprendizado por semelhança")


class RecomendacaoPokemon:
    def __init__(self):
        self.k_neighbors = 4
        self.knn_model = NearestNeighbors(n_neighbors=self.k_neighbors, metric='euclidean')
        self.knn_model.fit(pokemon_features)

    def renderize(self):
        with st.expander("Buscar recomendações por um Pokémon"):
            pokemon_choose = st.selectbox(
                'Escolha o Pokémon que mais gosta',
                pokemon_df['name'],
                help='Selecione um Pokémon que você gosta',
            )

            if pokemon_choose:
                selected_pokemon_index = pokemon_df[pokemon_df['name'] == pokemon_choose].index[0]
                distances, indices = self.knn_model.kneighbors(
                    pokemon_features.iloc[selected_pokemon_index].values.reshape(1, -1))

                st.subheader("Pokémon semelhantes de acordo com:")

                st.image(pokemon_df.iloc[selected_pokemon_index]['image'],
                         caption=pokemon_df.iloc[selected_pokemon_index]['name'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.header("1º")
                    st.image(pokemon_df.loc[indices[0][1], 'image'], caption=pokemon_df.loc[indices[0][1], 'name'],
                             width=100)

                with col2:
                    st.header("2º")
                    st.image(pokemon_df.loc[indices[0][2], 'image'], caption=pokemon_df.loc[indices[0][2], 'name'],
                             width=100)

                with col3:
                    st.header("3º")
                    st.image(pokemon_df.loc[indices[0][3], 'image'], caption=pokemon_df.loc[indices[0][3], 'name'],
                             width=100)


RecomendacaoPokemon().renderize()
