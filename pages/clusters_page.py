from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler

current_path = Path(__file__).resolve().parent.parent
file_path = current_path / "data" / "pokemon.csv"
pokemon_df = pd.read_csv(file_path)

st.subheader("Testes de clusters")

# atributos que serão usados na clusterização
attributes = ['typing','legendary','mythical','primary_color','shape','attack', 'defense','hp', 'speed', 'height', 'weight']
# copia dos atributos usados
pokemon_features = pokemon_df[attributes].copy()
# criar a variavel que vai aplicar a binarização
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# fazer a binarização
encoded_columns = pd.DataFrame(encoder.fit_transform(pokemon_features[['legendary','mythical','typing', 'shape', 'primary_color']]))
encoded_columns.columns = encoder.get_feature_names_out(['legendary','mythical','typing', 'shape', 'primary_color'])
# adicionar as novas colunas
pokemon_features = pd.concat([pokemon_features, encoded_columns], axis=1)
#retirar as antigas colunas
pokemon_features.drop(columns=['typing', 'shape','legendary','mythical', 'primary_color'], inplace=True)



scaler = StandardScaler()
pokemon_features[['attack', 'defense','hp', 'speed', 'height', 'weight']] = scaler.fit_transform(pokemon_features[['attack', 'defense','hp', 'speed', 'height', 'weight']])

# Criando o modelo K-means com 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(pokemon_features[pokemon_features.columns])

pokemon_features['cluster_label'] = kmeans.labels_

graph_clusters = px.scatter(pokemon_features, x='attack', y='defense',color='cluster_label',
                 title='Gráfico de Dispersão com Clusters')

st.plotly_chart(graph_clusters)

# aqui eu separei os pokemon de cada cluster em listas (PS: são 5 da am e eu to com preguiça de fazer um laço de repetição aqui -_-)
pokemon_cluster_1 = pokemon_df[pokemon_features['cluster_label'] == 1]
pokemon_cluster_2 = pokemon_df[pokemon_features['cluster_label'] == 2]
pokemon_cluster_3 = pokemon_df[pokemon_features['cluster_label'] == 3]
pokemon_cluster_4 = pokemon_df[pokemon_features['cluster_label'] == 4]

st.subheader("Pokémons do Cluster 1:")
for pokemon_name in pokemon_cluster_1['name']:
    st.write(pokemon_name)

for pokemon_image in pokemon_cluster_1['image'][:6]:
    st.image(pokemon_image)