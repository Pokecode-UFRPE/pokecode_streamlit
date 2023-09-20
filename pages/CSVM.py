# Importe as bibliotecas necessárias
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
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

# Selecione os atributos relevantes para o SVM
selected_columns = ['typing', 'hp', 'speed', 'height', 'weight', 'shape', 'primary_color']

# Tratamento dos dados dos Pokémon
pokemon_features = tratamento_dados.tratar_df()

# Label Encoding para as colunas 'shape', 'typing' e 'primary_color'
label_encoder = LabelEncoder()
pokemon_df['shape'] = label_encoder.fit_transform(pokemon_df['shape'])
pokemon_df['typing'] = label_encoder.fit_transform(pokemon_df['typing'])
pokemon_df['primary_color'] = label_encoder.fit_transform(pokemon_df['primary_color'])

# Inclua a clusterização K-means como atributo
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=0)  # Altere o número de clusters conforme necessário
pokemon_df['cluster'] = kmeans.fit_predict(pokemon_df[selected_columns])

# Selecione os atributos relevantes para o SVM
pokemon_features = pokemon_df[selected_columns].copy()

# Crie uma matriz de recursos e um vetor de destino
X = pokemon_features.values
y = pokemon_df['cluster'].values

# Padronize os atributos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Treine um modelo SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X, y)

# Interface Streamlit para recomendação de Pokémon
st.markdown('<hr>', unsafe_allow_html=True)
st.image("assets\icons\logo2.png")
st.markdown('<h1 class="site-title">Support Vector Machines com Clusterização (CSVM)</h1>', unsafe_allow_html=True)

with st.expander("Conversão de variáveis categóricas com LabelEncoder"):
    st.subheader('Tabela Categórica')
    st.write(pokemon_features)
    st.subheader('Tabela Numérica')
    st.write(pokemon_features)
    
with st.expander("Gráfico do Cotovelo"):
    # Lista para armazenar os valores de inércia
    inertia = []

    # Testar diferentes números de clusters
    for i in range(1, 15):  
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)  # X são seus dados
        inertia.append(kmeans.inertia_)

    # Plotar o gráfico de cotovelo
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 15), inertia, marker='o', linestyle='--')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inércia')
    plt.title('Método Elbow para Determinar o Número de Clusters')
    plt.grid(True)
    plt.show()
    st.pyplot(plt)
    
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(X, pokemon_df['cluster'])

    st.write(f"Coeficiente de Silhueta: {silhouette_avg}")
    
with st.expander("Avaliação do SVM em pokemon_df"):
    # Teste de acurácia com cross-validation
    cv_result = cross_val_score(svm_model, X, y, cv=10, scoring="accuracy")

    # Retorna a acurácia em porcentagem do modelo
    st.write("Acurácia com cross-validation:", cv_result.mean() * 100)

    # Fazer previsões usando o modelo treinado
    y_pred = svm_model.predict(X)

    # Matriz de confusão
    confusion = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Verdadeiros')
    plt.title('Matriz de Confusão')
    st.pyplot(plt)

    # Relatório de classificação
    classification_rep = classification_report(y, y_pred, output_dict=True)
    classification_df = pd.DataFrame(classification_rep).transpose()
    st.subheader('Relatório de Classificação')
    st.dataframe(classification_df)

with st.expander("Recomendações de Pokémon"):
    pokemon_choose = st.selectbox('Selecione um Pokémon', pokemon_df['name'],
                                  help='Selecione um Pokémon que você gosta')

    if pokemon_choose:
        # Encontre o cluster do Pokémon escolhido
        selected_pokemon_cluster = pokemon_df.loc[pokemon_df['name'] == pokemon_choose, 'cluster'].values[0]

        # Selecione Pokémon do mesmo cluster
        recommended_pokemon_data = pokemon_df[pokemon_df['cluster'] == selected_pokemon_cluster]

        num_recommendations = min(8, len(recommended_pokemon_data))

        # Mostra a imagem do Pokémon escolhido pelo usuário
        selected_pokemon_data = pokemon_df.loc[pokemon_df['name'] == pokemon_choose]
        # Mostra a imagem do Pokémon escolhido pelo usuário centralizada
        st.markdown(
            f"<div style='display: flex; justify-content: center;'>"
            f"<img src='{selected_pokemon_data['image'].values[0]}' width='100' alt='{selected_pokemon_data['name'].values[0]}' />"
            f"</div>",
            unsafe_allow_html=True
        )

        st.subheader("Pokémon recomendados:")
        colunas = st.columns(num_recommendations)

        for i in range(num_recommendations):
            with colunas[i]:
                st.header(f"{i + 1}º")
                st.image(recommended_pokemon_data.iloc[i+1]['image'],
                        caption=recommended_pokemon_data.iloc[i+1]['name'],
                        width=100)
