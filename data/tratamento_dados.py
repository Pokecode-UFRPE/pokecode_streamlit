import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def tratar_df():
    pokemon_df = pd.read_parquet('data/pokemon.parquet')

    selected_columns = ['typing', 'hp', 'speed', 'height', 'weight', 'shape', 'primary_color', 'attack', 'defense', 'base_happiness']
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
    scaler = StandardScaler()
    pokemon_features[['hp', 'speed', 'height', 'weight', 'attack', 'defense', 'base_happiness']] = scaler.fit_transform(pokemon_features[['hp', 'speed', 'height', 
                                                                                                                                          'weight', 'attack', 'defense', 
                                                                                                                                          'base_happiness']])
    return pokemon_features

def clusterizar_df():
    pokemon_features = tratar_df()
    # Aplicação do PCA para redução de dimensionalidade
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pokemon_features)

    # Método do Cotovelo para determinar o número ideal de clusters
    inertia_values = []
    for num_clusters in range(1, 11):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(pca_result)
        inertia_values.append(kmeans.inertia_)

    # gráfico de visualização do Elbow para selecionar o ponto de possivel melhor caso
    # graph_elbow = px.line(x=range(1, 11), y=inertia_values, title='Gráfico método do Cotovelo')
    # st.plotly_chart(graph_elbow.update_layout(xaxis_title='Número de Clusters', yaxis_title='Inércia'))

    # ponto escolhido com base no gráfico
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(pca_result)

    # Adição dos rótulos dos clusters aos dados
    pokemon_features['cluster_label'] = kmeans.labels_
    return pokemon_features, pca_result[:, 0], pca_result[:, 1]
