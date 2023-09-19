# Importe as bibliotecas necessárias
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

# Carregue o seu DataFrame de Pokémon a partir de um arquivo CSV  no spyder foi colocado na mesma pasta o arquivo CSV e o arquivo spyder.py
arquivo_csv = 'novoDataFrame.csv'
df = pd.read_csv(arquivo_csv)

# Escolha as colunas numéricas que serão usadas como características
numeric_columns = ['hp', 'speed', 'height', 'weight', 'defense', 'attack']

# Escolha a coluna categórica que você deseja incluir na análise (substitua 'typing' pelo nome da sua coluna categórica)
categorical_columns = ['typing']

# Combine as colunas numéricas e categóricas em um novo DataFrame
df_combined = df[numeric_columns + categorical_columns]

# Aplique a codificação one-hot à coluna categórica 'typing'
df_combined_encoded = pd.get_dummies(df_combined, columns=categorical_columns)

# Crie uma instância do modelo K-Prototypes
K = 11  # Substitua pelo número desejado de clusters
kproto = KPrototypes(n_clusters=K, init='Cao', verbose=2, n_init=1)  # Adicione n_init=1 para tentar uma inicialização

# Ajuste o modelo aos seus dados
clusters = kproto.fit_predict(df_combined_encoded.values, categorical=list(range(len(numeric_columns), len(df_combined_encoded.columns))))
df['cluster_label'] = clusters

# Encontre os N clusters mais próximos do cluster alvo 
target_cluster = 0
n_closest_clusters = 5

# Calcule as distâncias entre os clusters
cluster_distances = []
for i in range(K):
    if i != target_cluster:
        target_data = df_combined_encoded[clusters == target_cluster].values.astype(float)
        cluster_data = df_combined_encoded[clusters == i].values.astype(float)
        num_distances = cdist(target_data[:, :len(numeric_columns)], cluster_data[:, :len(numeric_columns)], metric='euclidean')
        
        # Calcule as distâncias categóricas
        cat_distances = np.zeros(target_data.shape[0])
        for j in range(target_data.shape[0]):
            cat_distances[j] = np.sum(target_data[j, len(numeric_columns):] != cluster_data[:, len(numeric_columns):])
        
        # Calcule a distância total como a média das distâncias numéricas e categóricas
        total_distances = num_distances.mean(axis=1) + cat_distances
        
        # Adicione à lista de distâncias
        cluster_distances.append((i, np.mean(total_distances)))

# Ordene os clusters com base nas distâncias médias
cluster_distances.sort(key=lambda x: x[1])

# Obtenha os índices dos N clusters mais próximos (excluindo o próprio cluster)
closest_cluster_indices = [cluster_index for cluster_index, _ in cluster_distances[:n_closest_clusters]]

# Agora closest_cluster_indices contém os índices dos N clusters mais próximos ao cluster alvo
print(f"Os {n_closest_clusters} clusters mais próximos do cluster {target_cluster} são: {closest_cluster_indices}")

# Calcule o coeficiente de Silhouette para avaliar a qualidade dos clusters
silhouette_avg = silhouette_score(df_combined_encoded.values, clusters)

# Exiba o coeficiente de Silhouette
print(f"Coeficiente de Silhouette: {silhouette_avg}")
