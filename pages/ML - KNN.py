from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
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

# Preparação KNNs
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

# Preparação Random Forest
X = pokemon_features 
y = pokemon_df['typing']

# Dividir os dados em conjuntos de treinamento e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar uma instância do modelo RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Treinar o modelo
rf_model.fit(X_train, y_train)

# Fazer previsões
y_pred = rf_model.predict(X_test)

# Avaliar o desempenho
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

# Preparação DBSCAN
# Pré-processamento dos dados
scaler = StandardScaler()
scaled_numeric_df = scaler.fit_transform(pokemon_features)

# Aplicar DBSCAN para agrupamento
dbscan = DBSCAN(eps=6, min_samples=4, metric='euclidean')
pokemon_clusters = dbscan.fit_predict(scaled_numeric_df)

# Preparação do modelo Random Forest
X = pokemon_features  # Seus recursos
y = pokemon_df['typing']  # Sua coluna de destino (classes)

# Dividir os dados em conjuntos de treinamento e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar uma instância do modelo RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Treinar o modelo
rf_model.fit(X_train, y_train)

# Fazer previsões
y_pred = rf_model.predict(X_test)

# Avaliar o desempenho (vamos usar a precisão como exemplo)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)


# Início da página principal
st.image("assets\icons\logo2.png")
st.markdown('<h1 class="site-title">Sistema de Recomendação</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="site-subt">Implementação de Machine Learning', unsafe_allow_html=True)

# Seção para K-Nearest Neighbors Puro
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

# Seção para K-Nearest Neighbors com Clusterização
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
        colunas_rf = st.columns(10)
        
        for i in range(10):
            with colunas_rf[i]:
                st.header(f"{i + 1}º")
                st.image(pokemon_df.loc[similar_pokemon_indices_rf[i], 'image'],
                         caption=pokemon_df.loc[similar_pokemon_indices_rf[i], 'name'], width=100)

# Seção para DBSCAN
st.markdown('<p class="site-subt"><b>DBSCAN</b></p>', unsafe_allow_html=True)

with st.expander("Recomendações de Pokémon"):
    pokemon_choose_dbscan = st.selectbox('Selete um Pokémon', pokemon_df['name'], help='Selecione um Pokémon que você gosta')

    if pokemon_choose_dbscan:
        selected_pokemon_index_dbscan = pokemon_df[pokemon_df['name'] == pokemon_choose_dbscan].index[0]

        # Encontrar o cluster do Pokémon de referência
        selected_pokemon_cluster = pokemon_clusters[selected_pokemon_index_dbscan]

        # Encontrar índices dos Pokémon no mesmo cluster
        similar_pokemon_indices = [index for index, cluster in enumerate(pokemon_clusters) if
                                   cluster == selected_pokemon_cluster and index != selected_pokemon_index_dbscan]

        st.subheader("Pokémon semelhantes:")

        colunas_dbscan = st.columns(10)
        for i in range(10):
            with colunas_dbscan[i]:
                st.header(f"{i + 1}º")
                st.image(pokemon_df.iloc[similar_pokemon_indices[i]]['image'],
                         caption=pokemon_df.iloc[similar_pokemon_indices[i]]['name'], width=100)

st.markdown('<hr>', unsafe_allow_html=True) 

# # Seção de comparação entre K-Nearest Neighbors Puro e K-Nearest Neighbors com Clusterização
# st.markdown('<p class="site-subt"><b>Qual algoritmo é mais adequado ao sistema?</b></p>', unsafe_allow_html=True)
# with st.expander("KNN Puro x KNN com Clusterização x Random Forest x DBSCAN "):
#     st.write("")
#     # ...
#     # Adicionar mais comentários explicativos se necessário
    
# st.markdown('<hr>', unsafe_allow_html=True) 