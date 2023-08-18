import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pokemon_df = pd.read_parquet('data/pokemon.parquet')
pokemon_df['image'] = ''
pokemon_df['image'] = pokemon_df['pokedex_number'].apply(lambda x: f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{x}.png')

selected_columns = ['typing', 'hp', 'speed', 'height', 'weight', 'shape', 'primary_color']
pokemon_features = pokemon_df[selected_columns].copy()

# GERANDO DUAS COLUNAS A PARTIR DA QUEBRA DA COLUNA DE TIPO
pokemon_features[['tipe1', 'tipe2']] = pokemon_features['typing'].str.split('~', n=1, expand=True)
pokemon_features.drop(columns=['typing'], inplace=True)

# PASSANDO O ONEHOT PARA GERAR AS COLUNAS BOOL DOS DADOS TEXTUAIS
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_columns_tipos = pd.DataFrame(encoder.fit_transform(pokemon_features[['tipe1', 'tipe2']]))
encoded_columns_tipos.columns = encoder.get_feature_names_out(['tipe1', 'tipe2'])
encoded_columns = pd.DataFrame(encoder.fit_transform(pokemon_features[['shape', 'primary_color']]))
encoded_columns.columns = encoder.get_feature_names_out(['shape', 'primary_color'])

# REMOVO A DIVISAO DE TIPOS PRIMARIOS E SECUNDARIOS CRIANDO UMA COLUNA GENERICA PARA CADA TIPO BOOL
colunas_tipos_genericas = set([col.split('_')[1] for col in encoded_columns_tipos.columns if 'tipe' in col])
tipos_bool = pd.DataFrame()
for tipo in colunas_tipos_genericas:
    tipo_cols = [col for col in encoded_columns_tipos.columns if f'tipe1_{tipo}' in col or f'tipe2_{tipo}' in col]
    tipos_bool[tipo] = encoded_columns_tipos[tipo_cols].any(axis=1).astype(int)

# DESCARTO OS DADOS QUE USEI PARA GERAR O ONEHOT E OS TIPOS GENERICOS
pokemon_features.drop(columns=['tipe1', 'tipe2', 'shape', 'primary_color'], inplace=True)

# CONCATENO OS TIPOS GENERICOS A FORMA E A COR
pokemon_features = pd.concat([pokemon_features, encoded_columns, tipos_bool], axis=1)
st.write(pokemon_features)

scaler = StandardScaler()
pokemon_features[['hp', 'speed', 'height', 'weight']] = scaler.fit_transform(pokemon_features[['hp', 'speed', 'height', 'weight']])

k_neighbors = 4
knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
knn_model.fit(pokemon_features)

st.header("Teste de ML")
st.subheader("Implementando aprendizado por semelhança")
    
with st.expander("Buscar recomendações por um Pokémon"):
    pokemon_choose = st.selectbox(
        'Escolha o Pokémon que mais gosta',
        pokemon_df['name'],
        help='Selecione um Pokémon que você gosta',
    )

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


            
