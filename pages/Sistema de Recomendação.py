from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

from data import tratamento_dados

# Configuração da página Streamlit
current_path = Path(__file__).resolve().parent.parent
logo1 = current_path / "assets" / "icons" / "logo1.png"
logo2 = current_path / "assets" / "icons" / "logo2.png"
file_path = current_path / "data" / "pokemon.parquet"
css_path = str(current_path / "assets" / "css" / "style.css")

st.set_page_config(
    page_title="POKECODE",
    page_icon="assets\icons\logo1.png",
    initial_sidebar_state="collapsed",
)

# Carregar o CSS personalizado
with open(css_path) as f:
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

# Label Encoding para as colunas 'shape', 'typing' e 'primary_color'
label_encoder = LabelEncoder()
pokemon_df['shape'] = label_encoder.fit_transform(pokemon_df['shape'])
pokemon_df['typing'] = label_encoder.fit_transform(pokemon_df['typing'])
pokemon_df['primary_color'] = label_encoder.fit_transform(pokemon_df['primary_color'])

# Selecionar colunas relevantes
selected_columns = ['typing', 'hp', 'speed', 'height', 'weight', 'shape', 'primary_color']
pokemon_features = pokemon_df[selected_columns].copy()

# Aplicar codificação one-hot às colunas categóricas
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_columns = pd.DataFrame(encoder.fit_transform(pokemon_features[['typing', 'shape', 'primary_color']]))
encoded_columns.columns = encoder.get_feature_names_out(['typing', 'shape', 'primary_color'])
pokemon_features = pd.concat([pokemon_features, encoded_columns], axis=1)
pokemon_features.drop(columns=['typing', 'shape', 'primary_color'], inplace=True)

# Coeficiente de silhueta
silhouette_avg = silhouette_score(pokemon_features_clusters, pokemon_features_clusters['cluster_label'])
silhouette_values = silhouette_samples(pokemon_features_clusters, pokemon_features_clusters['cluster_label'])
pokemon_features['silhouette_score'] = silhouette_values

# Configuração do modelo KNN
k_neighbors = 20
knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
knn_model.fit(pokemon_features)

# Configuração do modelo KNN com Clusterização
knn_modelC = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
knn_modelC.fit(pokemon_features_clusters)

# Configuração do modelo Random Forest
X = pokemon_features
y = pokemon_df['typing']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Configuração DBSCAN
scaler = StandardScaler()
scaled_numeric_df = scaler.fit_transform(pokemon_features)

dbscan = DBSCAN(eps=5, min_samples=5, metric='euclidean')
pokemon_clusters = dbscan.fit_predict(scaled_numeric_df)

# ...

# KNN
with st.expander('KNN'):
    # Calcular as previsões do modelo KNN
    _, indices_knn = knn_model.kneighbors(X_test)
    y_pred_knn = y.iloc[indices_knn[:, 0]].values.flatten()
    
    # Calcular a acurácia
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    st.write(f"Acurácia do Modelo KNN: {accuracy_knn:.2f}")

    # Relatório de Classificação
    knn_classification_report = classification_report(y_test, y_pred_knn)
    st.subheader("Relatório de Classificação do Modelo KNN:")
    st.text(knn_classification_report)

# KNN com Clusterização
with st.expander('KNN com Clusterização'):

    # Calcular o coeficiente de silhueta
    silhouette_avg = silhouette_score(scaled_numeric_df, pokemon_clusters)
    st.write(f"Coeficiente de Silhueta do Modelo DBSCAN: {silhouette_avg:.2f}")

# DBSCAN
with st.expander('DBSCAN'):
    # Avaliação do modelo DBSCAN
    st.subheader("Avaliação do Modelo DBSCAN:")
    # Aqui você pode incluir qualquer métrica de avaliação apropriada para o DBSCAN, já que ele não produz previsões de classe.
    # Por exemplo, você pode calcular a pontuação de silhueta para avaliar a qualidade dos clusters.
    silhouette_avg_dbscan = silhouette_score(X, dbscan.labels_)
    st.write(f"Silhouette Score do DBSCAN: {silhouette_avg_dbscan}")

# Random Forest
with st.expander('Random Forest'):
    # Calcular a acurácia
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    st.write(f"Acurácia do Modelo Random Forest: {accuracy_rf:.2f}")
    
    # Relatório de Classificação
    classification_report_rf = classification_report(y_test, y_pred_rf)
    st.subheader("Relatório de Classificação do Modelo Random Forest:")
    st.text(classification_report_rf)

st.markdown('<hr>', unsafe_allow_html=True)

st.markdown('<h3 class="site-subt"><b>Implementação de Machine Learning</b></h3>', unsafe_allow_html=True)
# Seção para K-Nearest Neighbors Puro
st.markdown('<p class="site-subt"><b>K-Nearest Neighbors</b></p>', unsafe_allow_html=True)

with st.expander("Recomendações de Pokémon"):
    pokemon_choose = st.selectbox('Escolha um Pokémon', pokemon_df['name'], help='Selecione um Pokémon que você gosta')

    if pokemon_choose:
        selected_pokemon_index = pokemon_df[pokemon_df['name'] == pokemon_choose].index[0]
        distances, indices = knn_model.kneighbors(pokemon_features.iloc[selected_pokemon_index].values.reshape(1, -1))

        # st.write(distances)
        st.subheader("Pokémon semelhantes:")

        if pokemon_choose:
            selected_pokemon_index = pokemon_df[pokemon_df['name'] == pokemon_choose].index[0]

            distances, indices = knn_model.kneighbors(
                pokemon_features.iloc[selected_pokemon_index].values.reshape(1, -1))
            colunas = st.columns(8)

            for i in range(8):
                with colunas[i]:
                    st.header(f"{i + 1}º")
                    st.image(pokemon_df.loc[indices[0][i + 1], 'image'],
                             caption=pokemon_df.loc[indices[0][i + 1], 'name'],
                             width=100)

# Seção para K-Nearest Neighbors com Clusterização
st.markdown('<p class="site-subt"><b>K-Nearest Neighbors com Clusterização</b></p>', unsafe_allow_html=True)

with st.expander("Recomendações de Pokémon"):
    pokemon_choose = st.selectbox('Selecione um Pokémon', pokemon_df['name'],
                                  help='Selecione um Pokémon que você gosta')

    if pokemon_choose:
        selected_pokemon_index = pokemon_df[pokemon_df['name'] == pokemon_choose].index[0]

        distances, indices = knn_modelC.kneighbors(
            pokemon_features_clusters.iloc[selected_pokemon_index].values.reshape(1, -1))

        # st.write(distances)
        st.subheader("Pokémon semelhantes:")
        colunas = st.columns(8)

        for i in range(8):
            with colunas[i]:
                st.header(f"{i + 1}º")
                st.image(pokemon_df.loc[indices[0][i + 1], 'image'], caption=pokemon_df.loc[indices[0][i + 1], 'name'],
                         width=100)

# Seção para Random Forest
st.markdown('<p class="site-subt"><b>Random Forest</b></p>', unsafe_allow_html=True)

with st.expander("Recomendações de Pokémon"):
    pokemon_choose_rf = st.selectbox('Opte por um Pokémon', pokemon_df['name'],
                                     help='Selecione um Pokémon que você gosta')

    if pokemon_choose_rf:
        # Fazer previsão com o modelo Random Forest
        selected_pokemon_index_rf = pokemon_df[pokemon_df['name'] == pokemon_choose_rf].index[0]
        selected_pokemon_features_rf = X.iloc[selected_pokemon_index_rf].values.reshape(1, -1)
        predicted_typing_rf = rf_model.predict(selected_pokemon_features_rf)

        # Filtrar Pokémon com o mesmo tipo previsto
        similar_pokemon_indices_rf = pokemon_df[pokemon_df['typing'] == predicted_typing_rf[0]].index
        similar_pokemon_indices_rf = similar_pokemon_indices_rf[similar_pokemon_indices_rf != selected_pokemon_index_rf]

        st.subheader("Pokémon semelhantes:")
        colunas_rf = st.columns(8)

        for i in range(8):
            with colunas_rf[i]:
                st.header(f"{i + 1}º")
                st.image(pokemon_df.loc[similar_pokemon_indices_rf[i], 'image'],
                         caption=pokemon_df.loc[similar_pokemon_indices_rf[i], 'name'], width=100)

# Seção para DBSCAN
st.markdown('<p class="site-subt"><b>DBSCAN</b></p>', unsafe_allow_html=True)

with st.expander("Recomendações de Pokémon"):
    pokemon_choose_dbscan = st.selectbox('Selete um Pokémon', pokemon_df['name'],
                                         help='Selecione um Pokémon que você gosta')

    if pokemon_choose_dbscan:
        selected_pokemon_index_dbscan = pokemon_df[pokemon_df['name'] == pokemon_choose_dbscan].index[0]

        # Encontrar o cluster do Pokémon de referência
        selected_pokemon_cluster = pokemon_clusters[selected_pokemon_index_dbscan]

        # Encontrar índices dos Pokémon no mesmo cluster
        similar_pokemon_indices = [index for index, cluster in enumerate(pokemon_clusters) if
                                   cluster == selected_pokemon_cluster and index != selected_pokemon_index_dbscan]

        st.subheader("Pokémon semelhantes:")

        colunas_dbscan = st.columns(8)
        for i in range(8):
            with colunas_dbscan[i]:
                st.header(f"{i + 1}º")
                st.image(pokemon_df.iloc[similar_pokemon_indices[i]]['image'],
                         caption=pokemon_df.iloc[similar_pokemon_indices[i]]['name'], width=100)

st.markdown('<hr>', unsafe_allow_html=True)
