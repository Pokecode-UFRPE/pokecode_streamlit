from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

current_path = Path(__file__).resolve().parent.parent
pokemon_csv = str(current_path / "data" / "pokemon.csv")
logo1 = str(current_path / "assets" / "icons" / "logo1.png")
logo2 = str(current_path / "assets" / "icons" / "logo2.png")
css_path = str(current_path / "assets" / "css" / "style.css")

st.set_page_config(
    page_title="POKECODE",
    page_icon=logo1,
    initial_sidebar_state="collapsed",
)

with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.image(logo2)
st.markdown('<h1 class="site-title">Ánalise Comparativa</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="site-subt">Escolha dois Pokémons para comparar:</h3>', unsafe_allow_html=True)

data = pd.read_csv(pokemon_csv)

st.title("K-Means Clustering com Streamlit")
k = st.sidebar.slider("Número de Clusters (k)", min_value=2, max_value=50, value=3)

selected_pokemon = st.sidebar.selectbox("Escolha um Pokémon", data["name"])
selected_cols = st.sidebar.multiselect("Selecione as colunas para clusterização", data.columns)
if selected_cols and selected_pokemon:
    pokemon_data = data[data["name"] == selected_pokemon]
    X = data[selected_cols]
    data_encoded = pd.get_dummies(X,
                                  columns=selected_cols)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(data_encoded)
    data["cluster"] = kmeans.labels_
    fig = px.scatter(data, x=selected_cols[0], y=selected_cols[1], color="cluster", title="K-Means Clustering")
    st.plotly_chart(fig)

    cluster_selected_pokemon = kmeans.predict(data_encoded)
    data["cluster"] = kmeans.labels_
    similar_pokemon = data[data["cluster"] == cluster_selected_pokemon[0]]
    st.write("Pokémon com características semelhantes:")
    st.write(similar_pokemon)

    silhouette_avg = silhouette_score(data_encoded, kmeans.labels_)
    st.write(f"Coeficiente de Silhueta: {silhouette_avg:.2f}")
    st.write("O coeficiente de silhueta varia de -1 a 1, onde:")
    st.write("- Valores próximos de 1 indicam uma clusterização bem definida.")
    st.write("- Valores próximos de 0 indicam sobreposição entre clusters.")
    st.write("- Valores próximos de -1 indicam uma clusterização incorreta.")
    if silhouette_avg >= 0.7:
        st.markdown('<p class="silhoute-badget-good">A clusterização é muito boa</p>', unsafe_allow_html=True)
    elif 0.5 <= silhouette_avg < 0.7:
        st.markdown('<p class="silhoute-badget-ok">A clusterização é razoável</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="silhoute-badget-bad">A clusterização pode não ser ideal</p>', unsafe_allow_html=True)

else:
    st.warning("Selecione pelo menos duas colunas para clusterização na barra lateral e escolha um Pokémon.")
