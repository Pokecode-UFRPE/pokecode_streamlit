from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from data import tratamento_dados

# Configuração da página Streamlit
st.set_page_config(
    page_title="POKECODE",
    page_icon="assets\icons\logo1.png",
    initial_sidebar_state="collapsed",
)

# Carregar o CSS personalizado
with open('assets/css/style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Definir o caminho do arquivo de dados
current_path = Path(__file__).resolve().parent.parent
file_path = current_path / "data" / "pokemon.parquet"

# Ler os dados dos Pokémon
pokemon_df = pd.read_parquet(file_path)
pokemon_df['image'] = ''
pokemon_df['image'] = pokemon_df['pokedex_number'].apply(
    lambda x: f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{x}.png')

# Tratamento dos dados dos Pokémon
pokemon_features = tratamento_dados.tratar_df()
pokemon_features_clusters, _, _ = tratamento_dados.clusterizar_df()
pokemon_df['image'] = pokemon_df['pokedex_number'].apply(
    lambda x: f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{x}.png')

# Selecionar colunas relevantes
selected_columns = ['typing', 'hp', 'speed', 'height', 'weight', 'shape', 'primary_color']
pokemon_features = pokemon_df[selected_columns].copy()

# Aplicar codificação one-hot às colunas categóricas
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_columns = pd.DataFrame(encoder.fit_transform(pokemon_features[['typing', 'shape', 'primary_color']]))
encoded_columns.columns = encoder.get_feature_names_out(['typing', 'shape', 'primary_color'])
pokemon_features = pd.concat([pokemon_features, encoded_columns], axis=1)
pokemon_features.drop(columns=['typing', 'shape', 'primary_color'], inplace=True)

# Escalar as features numéricas
scaler = StandardScaler()
pokemon_features[['hp', 'speed', 'height', 'weight']] = scaler.fit_transform(
    pokemon_features[['hp', 'speed', 'height', 'weight']])

# Configuração do modelo KNN
k_neighbors = 20
knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
knn_model.fit(pokemon_features)

# Configuração do modelo KNN com Clusterização
knn_modelC = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
knn_modelC.fit(pokemon_features_clusters)

# Configuração do modelo DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=4, metric='euclidean')
pokemon_clusters = dbscan.fit_predict(scaled_features_dbscan)

# Início da página principal
st.image("assets\icons\logo2.png")
st.markdown('<h1 class="site-title">Sistema de Recomendação</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="site-subt">Implementação de Machine Learning', unsafe_allow_html=True)

# Seção para K-Nearest Neighbors Puro
st.markdown('<p class="site-subt"><b>K-Nearest Neighbors Puro</b></p>', unsafe_allow_html=True)
# ...
# Adicionar mais comentários explicativos se necessário

# Seção para K-Nearest Neighbors com Clusterização
with st.expander("Recomendações de Pokémon"):
    st.selectbox('Escolha um Pokémon', pokemon_df['name'], help='Selecione um Pokémon que você gosta')
    # ...
    # Adicionar mais comentários explicativos se necessário

# Seção para DBSCAN
st.markdown('<p class="site-subt"><b>DBSCAN</b></p>', unsafe_allow_html=True)
with st.expander("Recomendações de Pokémon (DBSCAN)"):
    st.selectbox('Selecione um Pokémon (DBSCAN)', pokemon_df['name'], help='Selecione um Pokémon que você gosta')
    # ...
    # Adicionar mais comentários explicativos se necessário

# Seção de comparação de algoritmos
st.markdown('<h3 class="site-subt"><b>Comparação de Algoritmos</b></h3>', unsafe_allow_html=True)

# Seção de comparação entre K-Nearest Neighbors Puro e K-Nearest Neighbors com Clusterização
st.markdown('<p class="site-subt"><b>K-Nearest Neighbors Puro x K-Nearest Neighbors com Clusterização</b></p>', unsafe_allow_html=True)
with st.expander("KNN Puro x KNN Cluster"):
    st.write("")
    # ...
    # Adicionar mais comentários explicativos se necessário

# Seção de comparação entre K-Nearest Neighbors com Clusterização e DBSCAN
st.markdown('<p class="site-subt"><b>K-Nearest Neighbors com Clusterização x DBSCAN</b></p>', unsafe_allow_html=True)
with st.expander("KNN Cluster x DBSCAN"):
    st.write("")
    # ...
    # Adicionar mais comentários explicativos se necessário
