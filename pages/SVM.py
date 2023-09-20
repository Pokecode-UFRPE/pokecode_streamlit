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

# Tratamento dos dados dos Pokémon
pokemon_features = tratamento_dados.tratar_df()
pokemon_df['image'] = pokemon_df['pokedex_number'].apply(
    lambda x: f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{x}.png')

selectedddd_columns = ['typing', 'hp', 'speed', 'height', 'weight', 'shape', 'primary_color']
pokemon_featuresss = pokemon_df[selectedddd_columns].copy()

# Label Encoding para as colunas 'shape', 'typing' e 'primary_color'
label_encoder = LabelEncoder()
pokemon_df['shape'] = label_encoder.fit_transform(pokemon_df['shape'])
pokemon_df['typing'] = label_encoder.fit_transform(pokemon_df['typing'])
pokemon_df['primary_color'] = label_encoder.fit_transform(pokemon_df['primary_color'])
pokemon_df['classes'] = label_encoder.fit_transform(pokemon_df['shape'])

# Selecione os atributos relevantes para o SVM
selected_columns = ['typing', 'hp', 'speed', 'height', 'weight', 'shape', 'primary_color']
pokemon_features = pokemon_df[selected_columns].copy()

# Crie uma matriz de recursos e um vetor de destino
X = pokemon_features.values
y = pokemon_df['classes'].values

# Padronize os atributos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Treine um modelo SVM
svm_model = SVC(kernel='linear')
# svm_model = SVC(kernel='rbf')
# svm_model = SVC(kernel='poly', degree=3)

svm_model.fit(X, y)

# Interface Streamlit para recomendação de Pokémon
st.markdown('<hr>', unsafe_allow_html=True)
st.image("assets\icons\logo2.png")
st.markdown('<h1 class="site-title">Support Vector Machines (SVM)</h1>', unsafe_allow_html=True)

with st.expander("Conversão de variáveis categóricas com LabelEncoder"):
    st.subheader('Tabela Categórica')
    st.write(pokemon_featuresss)
    st.subheader('Tabela Numérica')
    st.write(pokemon_features)

with st.expander("Exploração do SVM"):
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pokemon_features)
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Escolha as características numéricas que você deseja usar no gráfico de violino
    features = ['hp', 'speed', 'height', 'weight']

    # Selecione os dados relevantes
    data_to_plot = pokemon_df[['classes'] + features]

    # Crie um gráfico de violino
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    for feature in features:
        plt.subplot(2, 2, features.index(feature) + 1)
        sns.violinplot(x='classes', y=feature, data=data_to_plot, palette='viridis')
        plt.title(f'Distribuição de {feature} por Classe')

    plt.tight_layout()
    plt.show()
    st.pyplot(plt)
    
    # Crie um DataFrame com os dados padronizados
    data_std = pd.DataFrame(X, columns=selected_columns)

    # Adicione a coluna 'classes' de volta aos dados
    data_std['classes'] = y

    # Crie o gráfico de distribuição usando Seaborn (Violin Plot)
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='classes', y='typing', data=data_std)
    plt.xlabel("Classe")
    plt.ylabel("Tipo")
    plt.title("Distribuição dos tipos de Pokémon por Classes")
    st.pyplot(plt)
    
    # Padronizar os dados (média zero e desvio padrão um)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pokemon_df[selected_columns])

    # Aplicar PCA para redução de dimensionalidade para 2 componentes principais
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)

    # Criar um DataFrame com os componentes principais
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Adicionar as classes de Pokémon de volta ao DataFrame
    pca_df['Class'] = pokemon_df['shape']

    # Crie o gráfico de distribuição usando Seaborn (Violin Plot)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Class', data=pca_df, palette='viridis')
    plt.xlabel("Classe")
    plt.ylabel("Pokemon")
    plt.title("Distribuição dos Pokémon por Classes")
    st.pyplot(plt)

with st.expander("Avaliação do SVM em pokemon_df"):
    #Teste de acurácia com cross-validation
    cv_result = cross_val_score(svm_model, X, y, cv=14, scoring="accuracy")

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
        selected_pokemon_features = pokemon_df.loc[pokemon_df['name'] == pokemon_choose, selected_columns].values
        selected_pokemon_features = scaler.transform(selected_pokemon_features)
        
        # Calcule a distância de cosseno entre o Pokémon escolhido e todos os outros Pokémon
        cosine_dists = cosine_distances(selected_pokemon_features, X)
        
        # Ordene os Pokémon pelo grau de similaridade (menor distância de cosseno é mais similar)
        similar_pokemon_indices = cosine_dists.argsort(axis=1).flatten()
        
        # Exclua o Pokémon de referência da lista de recomendações
        similar_pokemon_indices = similar_pokemon_indices[similar_pokemon_indices != 0]
        
        num_recommendations = min(8, len(similar_pokemon_indices))
        
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
                recommended_pokemon_data = pokemon_df.iloc[similar_pokemon_indices[i+1]]
                
                st.header(f"{i + 1}º")
                st.image(recommended_pokemon_data['image'],
                        caption=recommended_pokemon_data['name'],
                        width=100)
                