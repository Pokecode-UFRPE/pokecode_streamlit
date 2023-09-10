import random
from pathlib import Path

import pandas as pd
import streamlit as st
from st_pages import Page, show_pages

current_path = Path(__file__).resolve().parent
file_path = str(current_path / "data" / "pokemon.csv")
parquet = str(current_path / "data" / "pokemon.parquet")
css_path = str(current_path / "assets" / "css" / "style.css")
#logo1 = current_path / "assets" / "icons" / "logo1.png"
st.set_page_config(
    page_title="POKECODE",
    #page_icon=str(logo1),
    page_icon="assets\icons\logo1.png",
    initial_sidebar_state="collapsed",
)

show_pages(
    [
        Page("main.py", " POKECODE", "📌"),
        Page("pages/AED - Painel de Gráficos.py", " Análise de Dados", "📈"),
        Page("pages/AED - Painel de Tabelas.py", " Visualização do DataFrame", "📊"),
        Page("pages/Sistema de Recomendação.py", " Sistemas de Recomendação", "💬"),
        Page("pages/Painel_de_distancias.py", " Análise Comparativa", "📏"),
    ]
)

with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

pokemon_df = pd.read_parquet(parquet)
pokemon_df = pokemon_df.drop_duplicates(subset='pokedex_number')
pokemon_df['image'] = ''
pokemon_df['image'] = pokemon_df['pokedex_number'].apply(
    lambda x: f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{x}.png')


def generate_data_editor(result):
    return st.data_editor(
        result,
        column_config={
            "image": st.column_config.ImageColumn(
                "Image"
            )
        },
        hide_index=True, width=1000
    )


filtered_df = pokemon_df
columns_to_keep = [
    "name", "pokedex_number", "image", "typing",
    "hp", "speed", "height", "weight",
    "shape", "primary_color", "attack",
    "defense", "base_happiness"
]

filtered_and_cut_df = filtered_df[columns_to_keep]


def generate_data_editor(result):
    return st.data_editor(
        result,
        column_config={
            # precisa criar id diferente se não dá conflito quando chama o mesmo banco na busca com mais de um parametro
            "id": st.column_config.TextColumn(
                str(random.randint(1, 1000))
            ),
            "image": st.column_config.ImageColumn(
                "Image"
            )
        },
        hide_index=True, width=1000
    )


def dataset():
    return filtered_and_cut_df


# MAIN PAGE START --
# logo2 = current_path / "assets" / "icons" / "logo2.png"
# st.image(str(logo2))

st.image("assets\icons\logo2.png")

st.markdown('<h2 class="site-subt">Conheça o PokeCode</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="site-subt">O <b>POKECODE</b> tem como objetivo desenvolver uma solução que simplifique a pesquisa, aprendizado e acesso a informações sobre os Pokémon de interesse do usuário, utilizando dados de um dataset como base, a fim de aprimorar a experiência de jogo.</p>',
    unsafe_allow_html=True)

st.markdown('<h3 class="site-subt">Conjunto de Dados</h3>', unsafe_allow_html=True)
st.markdown('<p><b>Complete Pokemon Data Set</b><br>Por Kyle Kohnen</p>', unsafe_allow_html=True)

st.markdown(
    '<p>O conjunto de dados utilizado na aplicação contém 1118 Pokémon diferentes e 49 colunas distintas contendo informações distintas sobre as criaturas do. A maioria dos dados foi extraída do PokeAPI. Outras fontes de dados incluem PokemonDB, Serebii, Bulbapedia e Pokemon Wiki.</p>',
    unsafe_allow_html=True)

with st.expander("Visualizar o DataSet:"):
    conjdados = dataset()
    generate_data_editor(conjdados)

st.markdown('<p class="site-subt"><b>Usabilidade</b><br>10.00</p>', unsafe_allow_html=True)
st.markdown('<p class="site-subt"><b>Tags</b><br>Games, Anime, Video Games, Anime and Manga, Popular Culture</p>',
            unsafe_allow_html=True)

st.markdown('<h3 class="site-subt">Aplicações</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    button1 = st.markdown("""
        <style>
        .custom-button {
            background-color: #5171FD;
            color: white;
            padding: 14px 20px;
            margin: 8px 0;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
        }
        .custom-button:hover {
            opacity: 0.8;
        }
        </style><button class="custom-button">Análise de Dados</button>
        """, unsafe_allow_html=True)

with col2:
    button2 = st.markdown("""
        <style>
        .custom-button {
            background-color: #5171FD;
            color: white;
            padding: 14px 20px;
            margin: 8px 0;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
        }
        .custom-button:hover {
            opacity: 0.8;
        }
        </style><button class="custom-button">Visualização do Dataframe</button>
        """, unsafe_allow_html=True)

with col3:
    button3 = st.markdown("""
        <style>
        .custom-button {
            background-color: #5171FD;
            color: white;
            padding: 14px 20px;
            margin: 8px 0;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
        }
        .custom-button:hover {
            opacity: 0.8;
        }
        </style><button class="custom-button">Sistema de Recomendação</button>
        """, unsafe_allow_html=True)

st.markdown('<br><hr>', unsafe_allow_html=True)

st.markdown('<h3 class="site-subt">Sobre o projeto</h3>', unsafe_allow_html=True)

st.markdown(
    '<p class="site-subt"><b>Equipe PokeCode</b><br> Alberson Alison de Araújo, André Filipe de Oliveira Figueiredo, Enzo Ferro Kretli, Gabriel Café Nunes de Souza, Isis Maria Oliveira Nilo de Souza, Pedro Henrique Correia da Silva.</p>',
    unsafe_allow_html=True)

st.markdown(
    '<p class="site-subt"><b>Universidade Federal Rural de Pernambuco (UFRPE)</b><br><b>Departamento de Estatística e Informática</b><br>Projeto para as disciplinas de Projeto Interdisciplinar para Sistemas da Informação III e Desenvolvimento de Sistemas da Informação.</p>',
    unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)
