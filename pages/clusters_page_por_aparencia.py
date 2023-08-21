import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors

pokemon_df = pd.read_csv("pokemonParaClusters.csv")
pokemon_df['image'] = pokemon_df['pokedex_number'].apply(lambda x: f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{x}.png')
st.subheader("Testes de clusters")

# atributos que serão usados na clusterização
attributes = ['typing','legendary','mythical','primary_color','shape']
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



# Criando o modelo K-means com 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(pokemon_features[pokemon_features.columns])

pokemon_features['cluster_label'] = kmeans.labels_
pokemon_df['cluster_label'] = kmeans.labels_

# aqui eu separei os pokemon de cada cluster em listas (PS: são 5 da am e eu to com preguiça de fazer um laço de repetição aqui -_-)
pokemon_cluster_1 = pokemon_df[pokemon_features['cluster_label'] == 1]
pokemon_cluster_2 = pokemon_df[pokemon_features['cluster_label'] == 2]
pokemon_cluster_3 = pokemon_df[pokemon_features['cluster_label'] == 3]
pokemon_cluster_4 = pokemon_df[pokemon_features['cluster_label'] == 4]


clusters = [pokemon_cluster_1,pokemon_cluster_2,pokemon_cluster_3,pokemon_cluster_4]

# percorrer os clusters e exibir 10 imagens de cada 1
for i in range(0,len(clusters)):
    st.write(f"Pokémon cluster {i+1}")
    for pokemon_image in clusters[i]['image'][:10]:
        st.image(pokemon_image)

