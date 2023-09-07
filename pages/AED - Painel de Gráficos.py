import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import tratamento_dados

with open('assets/css/style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def obter_caracteristicas(row):
    caracteristicas = []
    if row['baby_pokemon']:
        caracteristicas.append('baby')
    if row['legendary']:
        caracteristicas.append('legendary')
    if row['mythical']:
        caracteristicas.append('mythical')
    if row['comum']:
        caracteristicas.append('comum')
    return ', '.join(caracteristicas)


color_discrete_map = {
    'baby': '#2AB4FF',
    'legendary': '#942BF9',
    'mythical': '#EF54FC',
    'comum': '#FFE115'
}

with open('assets/css/style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
pokemon_df = pd.read_parquet("data/pokemon.parquet")
# CODIGO QUE RETIRA OS DUPICADAS DE NÚMERO DA POKEDEX
pokemon_df = pokemon_df.drop_duplicates(subset='pokedex_number')

type_information = pokemon_df['typing'].value_counts()
format_information = pokemon_df['shape'].value_counts()
gen_information = pokemon_df['gen_introduced'].value_counts()
evolve_information = pokemon_df['can_evolve'].value_counts()
color_information = pokemon_df['primary_color'].value_counts()

# COPIANDO OS DADOS E GERANDO UMA NOVA TABELA PARA AVALIAR COM PRECISÃO A RARIDADE
raridade_df = pokemon_df[['baby_pokemon', 'legendary', 'mythical']].copy()
raridade_df['comum'] = ~(raridade_df['baby_pokemon'] | raridade_df['legendary'] | raridade_df['mythical'])
raridade_df['raridade'] = raridade_df.apply(obter_caracteristicas, axis=1)
rarity_information = raridade_df['raridade'].value_counts()

# MAIN
st.image("assets\icons\logo2.png")
st.markdown('<h1 class="site-title">Análise de Dados</h1>', unsafe_allow_html=True)
option = st.selectbox(
    'Selecione uma das opções para explorar os dados referentes a ela:',
    ['Explorar', 'Quantidade de espécies por tipo', 'Porcentagem de espécies pelo formato', 
     'Quantidade de espécies pela geração', 'Porcentagem de espécies que evoluem', 'Contagem de espécies pela cor principal', 
     'Porcentagem de espécies por raridade', 'Gráfico de dispersão', 'Clusterização'])

# Exibe os dados correspondentes ao filtro selecionado
if option == 'Quantidade de espécies por tipo':
    graph_type = go.Figure(
        data=go.Bar(y=type_information.index, x=type_information.values, orientation='h', marker=dict(color='#F55555')))
    st.plotly_chart(
        graph_type.update_layout(title='Gráfico dos Tipos', xaxis_title='Quantidade', yaxis_title='Tipos', height=3000))
    # st.plotly_chart(graph_type.update_layout(title='Gráfico dos Tipos', xaxis_title='Quantidade', yaxis_title='Tipos', height=3000,yaxis=dict(categoryorder='total ascending')))

elif option == 'Porcentagem de espécies pelo formato':
    graph_format = go.Figure(data=go.Pie(labels=format_information.index, values=format_information.values))
    st.plotly_chart(graph_format.update_layout(title='Gráfico dos Formatos', height=600))

elif option == 'Quantidade de espécies pela geração':
    gen_information = gen_information.sort_index()
    graph_gen = go.Figure(data=go.Bar(x=gen_information.index, y=gen_information.values, marker=dict(color='#F55555')))
    st.plotly_chart(
        graph_gen.update_layout(title='Gráfico das Gerações', xaxis_title='Gerações', yaxis_title='Quantidade',
                                height=600))

elif option == 'Porcentagem de espécies que evoluem':
    graph_evolve = go.Figure(data=go.Pie(labels=evolve_information.index, values=evolve_information.values,
                                         marker=dict(colors=['#B22222', '#008000'])))
    st.plotly_chart(graph_evolve.update_layout(title='Gráfico de possibilidade evolutiva', height=500))

elif option == 'Contagem de espécies pela cor principal':
    graph_color = go.Figure(data=go.Bar(x=color_information.index, y=color_information.values, marker=dict(
        color=['#20B2AA', '#8B4513', '#32CD32', '#B22222', '#4B0082', 'gray', '#D3D3D3', '#DAA520', '#FF69B4',
               'black'])))
    st.plotly_chart(
        graph_color.update_layout(title='Gráfico de quantidade por cor', xaxis_title='Cores', yaxis_title='Quantidade'))

elif option == 'Porcentagem de espécies por raridade':
    graph_rarity = go.Figure(data=go.Pie(labels=rarity_information.index, values=rarity_information.values,
                                         marker=dict(colors=[color_discrete_map[x] for x in rarity_information.index])))
    st.plotly_chart(graph_rarity.update_layout(title='Gráfico de raridades', height=600))

elif option == 'Gráfico de dispersão':
    grath_dispersal = px.scatter(pokemon_df, x='height', y='weight', title='Relação entre Altura e Peso dos Pokémon',
                                 labels={'height': 'Altura', 'weight': 'Peso'})
    st.plotly_chart(grath_dispersal)

elif option == 'Clusterização':
    pokemon_features, pca1, pca2 = tratamento_dados.clusterizar_df()

    # Criação do gráfico de dispersão com clusters
    graph_clusters = px.scatter(pokemon_features, pca1, pca2, color='cluster_label',
                                title='Gráfico de Clusters após PCA e Método do Cotovelo')
    st.plotly_chart(graph_clusters)
