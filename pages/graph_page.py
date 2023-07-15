import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
###
with open('assets\css\style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
pokemon_df = pd.read_csv("data\pokemon.csv")
# CODIGO QUE RETIRA OS DUPICADAS DE NÚMERO DA POKEDEX
pokemon_df = pokemon_df.drop_duplicates(subset='pokedex_number')

type_information = pokemon_df['typing'].value_counts()
format_information = pokemon_df['shape'].value_counts()
gen_information = pokemon_df['gen_introduced'].value_counts()

st.subheader("Filtros do DF")
st.text("Cada uma das opções irá retornar um gráfico distinto com base em um filtro de dados")
option = st.selectbox(
    'Selecione um dos filtros para mostrar os dados:',
    ['Quantidade de espécies por tipo', 'Quantidade de espécies por formato', 'Quantidade de espécies por geração', 'Gráfico de dispersão'])

# Exibe os dados correspondentes ao filtro selecionado
if option == 'Quantidade de espécies por tipo':
    st.text("Apresenta um gráfico contando a quantidade de pokémon por cada tipo existente")
    st.text("Ele inclui os pokemon que possuem dois tipos, e os subdivide")
    st.markdown("<br>", unsafe_allow_html=True)
    graph_type = go.Figure(data=go.Scatter(x=type_information.index, y=type_information.values, mode='lines'))
    st.plotly_chart(graph_type)
elif option == 'Quantidade de espécies por formato':
    st.text("Apresenta um gráfico que contabiliza a quantidade de pokemon definidos pelo formato")
    st.markdown("<br>", unsafe_allow_html=True)
    graph_format = go.Figure(data=go.Scatter(x=format_information.index,y=format_information.values,mode='lines'))
    st.plotly_chart(graph_format)
elif option == 'Quantidade de espécies por geração':
    st.text("Apresenta um gráfico que contabiliza a quantidade de pokemon por geração")
    st.markdown("<br>",unsafe_allow_html=True)
    gen_information = gen_information.sort_index()
    graph_gen = go.Figure(data=go.Scatter(x=gen_information.index,y=gen_information.values,mode='lines'))
    st.plotly_chart(graph_gen)
elif option == 'Gráfico de dispersão':
    st.text("Gráfico de dispersão que relaciona altura e peso dos pokemon")
    st.text("Utilizando a biblioteca matplotlib")
    st.markdown("<br>", unsafe_allow_html=True)
    grath_dispersal = px.scatter(pokemon_df, x='height', y='weight', title='Relação entre Altura e Peso dos Pokémon',
                 labels={'height': 'Altura', 'weight': 'Peso'})
    st.plotly_chart(grath_dispersal)