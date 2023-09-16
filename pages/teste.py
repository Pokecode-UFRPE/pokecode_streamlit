import streamlit as st
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from scipy.spatial.distance import cdist
from pathlib import Path

# Carregue o seu DataFrame de Pokémon, substitua 'pokemon.csv' pelo name do seu arquivo CSV
arquivo_csv = 'data/novoDataFrame.csv'

# Título da aplicação
st.title('Análise de Clusters de Pokémon')

# Carregamento dos dados
@st.cache
def load_data():
    df = pd.read_csv(arquivo_csv)
    return df

df = load_data()

# Exibir seleção de colunas numéricas e categóricas na parte principal do aplicativo
st.header('Configurações')
selected_numeric_columns = st.multiselect('Selecione colunas numéricas', ['hp', 'speed', 'height', 'weight', 'defense', 'attack', 'cluster_label'])
categorical_columns = st.selectbox('Selecione uma coluna categórica', df.columns)

# Combine as colunas numéricas e categóricas em um novo DataFrame
df_combined = df[selected_numeric_columns + [categorical_columns]]

# Aplicar codificação one-hot à coluna categórica
df_combined_encoded = pd.get_dummies(df_combined, columns=[categorical_columns])


# Número de clusters
K = st.slider('Número de Clusters (K)', min_value=2, max_value=20, value=5)

# Criação do modelo K-Prototypes
kproto = KPrototypes(n_clusters=K, init='Cao', verbose=2, n_init=1)

# Botão para executar a clusterização
if st.button('Executar Clusterização'):
    # Ajuste o modelo aos seus dados
    clusters = kproto.fit_predict(df_combined_encoded.values, categorical=[len(numeric_columns)])
    df['cluster_label'] = clusters

    # Encontre os N clusters mais próximos do cluster alvo (por exemplo, cluster 0)
    target_cluster = st.slider('Selecione o cluster alvo', min_value=0, max_value=K-1, value=0)
    n_closest_clusters = st.slider('Número de clusters mais próximos', min_value=1, max_value=K-1, value=5)

    # Calcular as distâncias entre os clusters
    cluster_distances = []
    for i in range(K):
        if i != target_cluster:
            target_data = df_combined_encoded[clusters == target_cluster].values.astype(float)
            cluster_data = df_combined_encoded[clusters == i].values.astype(float)
            num_distances = cdist(target_data[:, :len(numeric_columns)], cluster_data[:, :len(numeric_columns)], metric='euclidean')

            # Calcular as distâncias categóricas
            cat_distances = np.zeros(target_data.shape[0])
            for j in range(target_data.shape[0]):
                cat_distances[j] = np.sum(target_data[j, len(numeric_columns):] != cluster_data[:, len(numeric_columns):])

            # Calcular a distância total como a média das distâncias numéricas e categóricas
            total_distances = num_distances.mean(axis=1) + cat_distances

            # Adicionar à lista de distâncias
            cluster_distances.append((i, np.mean(total_distances)))

    # Ordene os clusters com base nas distâncias médias
    cluster_distances.sort(key=lambda x: x[1])

    # Obtenha os índices dos N clusters mais próximos (excluindo o próprio cluster)
    closest_cluster_indices = [cluster_index for cluster_index, _ in cluster_distances[:n_closest_clusters]]

    # Exibir os resultados na interface Streamlit
    st.subheader(f"Os {n_closest_clusters} clusters mais próximos do cluster {target_cluster} são:")
    for idx, dist in cluster_distances[:n_closest_clusters]:
        st.write(f"Cluster {idx}: Distância Média = {dist:.2f}")

# Seção para seleção de Pokémon
st.header('Seleção de Pokémon')

# Adicione um widget selectbox para permitir que o usuário escolha um Pokémon
pokemon_choosen = st.selectbox('Escolha um Pokémon', df['name'])

# Quando o usuário escolher um Pokémon, encontre os 5 Pokémon mais próximos com base em sua similaridade
if st.button('Encontrar Pokémon Semelhantes'):
    if pokemon_choosen:
        selected_pokemon_index = df[df['name'] == pokemon_choosen].index[0]
        distances = cdist(df_combined_encoded.values[selected_pokemon_index].reshape(1, -1), df_combined_encoded.values, metric='euclidean')
        distances = distances[0]

        # Ordene os Pokémon com base nas distâncias (os 5 mais próximos)
        closest_pokemon_indices = np.argsort(distances)[:6]  # +1 para incluir o próprio Pokémon

        # Exibir os 5 Pokémon mais próximos
        st.subheader(f"Os 5 Pokémon mais próximos de {pokemon_choosen} são:")
        for idx in closest_pokemon_indices[1:]:
            st.write(f"- {df['name'].iloc[idx]}")
