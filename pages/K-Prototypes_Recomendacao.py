# Importe as bibliotecas necessárias
import streamlit as st
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregue o seu DataFrame de Pokémon a partir de um arquivo CSV (substitua 'novoDataFrame.csv' pelo nome do seu arquivo)
arquivo_csv = '/home/gabrielcafe/Documents/UFRPE/PSI3/pokecode_streamlit/data/novoDataFrame.csv'


# Título da aplicação
st.title('Análise de Agrupamento de Dados com K-Prototypes: Um Estudo de Caso com Pokémon')

# Função para carregar os dados com cache
@st.cache_data
def load_data():
    df = pd.read_csv(arquivo_csv)
    return df

# Carregamento dos dados
df = load_data()

# Seção para seleção do número de clusters (K)
st.header('Seleção do Número de Clusters (K)')
K = st.slider('Escolha o número de clusters (K)', min_value=11, max_value=20, value=11)

# Exibir seleção de coluna categórica na parte principal do aplicativo
st.header('Configurações')

# Remova a coluna 'name' da seleção do usuário
categorical_columns = [col for col in df.columns if col != 'name']
default_column = 'hp'  # Defina 'hp' como a coluna padrão
categorical_column = st.selectbox('Selecione uma coluna categórica', categorical_columns, index=categorical_columns.index(default_column))

# Seção para seleção de Pokémon
st.header('Seleção de Pokémon')

# Adicione um widget selectbox para permitir que o usuário escolha um Pokémon
pokemon_choosen = st.selectbox('Escolha um Pokémon', df['name'])

# Botão para executar clusterização e mostrar ambas as opções
if st.button('Executar Clusterização'):
    # Combine as colunas categóricas em um novo DataFrame
    df_combined = df[[categorical_column]]

    # Aplicar codificação one-hot à coluna categórica selecionada
    df_combined_encoded = pd.get_dummies(df_combined, columns=[categorical_column])

    # Criação do modelo K-Prototypes
    kproto = KPrototypes(n_clusters=K, init='Cao', verbose=2, n_init=1)

    # Ajuste o modelo aos seus dados
    clusters = kproto.fit_predict(df_combined_encoded.values, categorical=[0])
    df['cluster_label'] = clusters

    # Quando o usuário escolher um Pokémon, encontre os 5 Pokémon mais próximos com base em sua similaridade
    if pokemon_choosen:
        selected_pokemon_index = df[df['name'] == pokemon_choosen].index[0]
        distances = cdist(df_combined_encoded.values[selected_pokemon_index].reshape(1, -1), df_combined_encoded.values, metric='euclidean')
        distances = distances[0]

        # Ordene os Pokémon com base nas distâncias (os 5 mais próximos)
        closest_pokemon_indices = np.argsort(distances)[:6]  # +1 para incluir o próprio Pokémon

        # Exibir os 5 Pokémon mais próximos com base na categoria selecionada
        st.subheader(f"Os 5 Pokémon mais próximos de {pokemon_choosen} de acordo com {categorical_column}:")
        for idx in closest_pokemon_indices[1:]:
            st.write(f"- {df['name'].iloc[idx]}")

        # Calcule o coeficiente de Silhouette após a clusterização
        silhouette_avg = silhouette_score(df_combined_encoded, clusters)

        # Exiba o valor do coeficiente de Silhouette ao usuário
        st.subheader('Coeficiente de Silhouette:')
        st.write(silhouette_avg)

        # Gráfico de barras da distribuição dos Pokémon nos clusters
        st.header('Distribuição dos Pokémon nos Clusters')
        cluster_counts = df['cluster_label'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
        plt.xlabel('Clusters')
        plt.ylabel('Número de Pokémon')
        st.pyplot(plt)
