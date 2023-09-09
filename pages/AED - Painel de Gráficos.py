import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data import tratamento_dados

st.set_page_config(
    page_title="POKECODE",
    page_icon="assets\icons\logo1.png",
    initial_sidebar_state="collapsed",
)

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

pokemon_df = pd.read_parquet("data\pokemon.parquet")
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

st.image("assets\icons\logo2.png")
st.markdown('<h1 class="site-title">Análise de Dados</h1>', unsafe_allow_html=True)

option = st.selectbox(
    'Selecione uma das opções de gráficos para explorar os seus dados referentes:',
    ['Explorar', 
     'Quantidade de espécies por tipo',
     'Quantidade por tipos principais',
     'Relação pokémon água x Cor Azul',
     'Porcentagem de espécies por formato', 
     'Quantidade de espécies por geração', 
     'Porcentagem de espécies que possuem evolução', 
     'Quantidade de espécies por cor', 
     'Porcentagem de espécies por raridade',
     'Gráfico de relação ataques efetivos e tipos',
     'Gráfico de felicidade', 
     'Gráfico de dispersão',      
     'Clusterização'])

# Exibe os dados correspondentes ao filtro selecionado
if option == 'Quantidade de espécies por tipo':
    graph_type = go.Figure(data=go.Bar(y=type_information.index, x=type_information.values, orientation='h', marker=dict(color='#F55555')))
    st.plotly_chart(graph_type.update_layout(title='Gráfico dos Tipos', xaxis_title='Quantidade', yaxis_title='Tipos', height=3000))
    # st.plotly_chart(graph_type.update_layout(title='Gráfico dos Tipos', xaxis_title='Quantidade', yaxis_title='Tipos', height=3000,yaxis=dict(categoryorder='total ascending')))
elif option == 'Quantidade por tipos principais':
    typing_df = pokemon_df.copy()
    
    # Modifique os valores da coluna 'typing' para manter apenas o primeiro tipo antes de '~'.
    typing_df['typing'] = typing_df['typing'].apply(lambda x: x.split('~')[0])
    st.dataframe(typing_df)
    fig = px.bar(typing_df['typing'].value_counts(), x=typing_df['typing'].value_counts().index,
             y=typing_df['typing'].value_counts().values,
             labels={'x': 'Tipo Principal', 'y': 'Quantidade'},
             title='Quantidade de Pokémon por Tipo Principal')

    # Exiba o gráfico.
    st.plotly_chart(fig)
elif option == 'Relação pokémon água x Cor Azul':
    pokemon_water = pokemon_df[pokemon_df['typing'] == 'Water']

    # Conta quantos Pokémon do tipo 'Water' têm 'primary_color' igual a 'Blue'.
    contagem_blue_water = len(pokemon_water[pokemon_water['primary_color'] == 'Blue'])

    # Calcula a porcentagem de Pokémon do tipo 'Water' com 'primary_color' igual a 'Blue'.
    porcentagem_blue_water = (contagem_blue_water / len(pokemon_water)) * 100

    # Calcula a porcentagem de Pokémon do tipo 'Water' com 'primary_color' diferente de 'Blue'.
    porcentagem_outras_cores = 100 - porcentagem_blue_water

    # Cria um DataFrame para o gráfico de pizza.
    dados_grafico = pd.DataFrame({
        'Categoria': ['Blue', 'Outras Cores'],
        'Porcentagem': [porcentagem_blue_water, porcentagem_outras_cores]
    })

    # Define as cores para o gráfico (azul para 'Blue' e preto para 'Outras Cores').
    cores = ['#1f77b4', 'black']

    # Cria um gráfico de pizza com cores personalizadas.
    fig = px.pie(dados_grafico, names='Categoria', values='Porcentagem',
                title='Porcentagem de Pokémon do Tipo "Water" com Cor Primária "Blue"',
                color_discrete_sequence=cores)

    # Exibe o gráfico.
    st.plotly_chart(fig)
    # Filtra os Pokémon do tipo 'Fire'.
    pokemon_fire = pokemon_df[pokemon_df['typing'] == 'Fire']

    # Conta quantos Pokémon do tipo 'Fire' têm 'primary_color' igual a 'Blue'.
    contagem_blue_fire = len(pokemon_fire[pokemon_fire['primary_color'] == 'Blue'])

    # Calcula a porcentagem de Pokémon do tipo 'Fire' com 'primary_color' igual a 'Blue'.
    porcentagem_blue_fire = (contagem_blue_fire / len(pokemon_fire)) * 100

    # Calcula a porcentagem de Pokémon do tipo 'Fire' com 'primary_color' diferente de 'Blue'.
    porcentagem_outras_cores = 100 - porcentagem_blue_fire

    # Cria um DataFrame para o gráfico de pizza.
    dados_grafico = pd.DataFrame({
        'Categoria': ['Blue', 'Outras Cores'],
        'Porcentagem': [porcentagem_blue_fire, porcentagem_outras_cores]
    })

    # Define as cores para o gráfico (azul para 'Blue' e preto para 'Outras Cores').
    cores = ['black','#1f77b4']

    # Cria um gráfico de pizza com cores personalizadas.
    fig = px.pie(dados_grafico, names='Categoria', values='Porcentagem',
                title='Porcentagem de Pokémon do Tipo "Fire" com Cor Primária "Blue"',
                color_discrete_sequence=cores)

    # Exibe o gráfico.
    st.plotly_chart(fig)
elif option == 'Porcentagem de espécies por formato':
    graph_format = go.Figure(data=go.Pie(labels=format_information.index, values=format_information.values))
    st.plotly_chart(graph_format.update_layout(title='Gráfico dos Formatos', height=600))

elif option == 'Quantidade de espécies por geração':
    gen_information = gen_information.sort_index()
    graph_gen =  go.Figure(data=go.Bar(x=gen_information.index, y=gen_information.values, marker=dict(color='#F55555')))
    st.plotly_chart(graph_gen.update_layout(title='Gráfico das Gerações', xaxis_title='Gerações', yaxis_title='Quantidade', height=600))

elif option == 'Porcentagem de espécies que possuem evolução':
    graph_evolve = go.Figure(data=go.Pie(labels=evolve_information.index, values=evolve_information.values, marker=dict(colors=['#B22222', '#008000'])))
    st.plotly_chart(graph_evolve.update_layout(title='Gráfico de possibilidade evolutiva', height=500))

elif option == 'Quantidade de espécies por cor':
    graph_color =  go.Figure(data=go.Bar(x=color_information.index, y=color_information.values, marker=dict(
        color=['#20B2AA', '#8B4513', '#32CD32', '#B22222', '#4B0082','gray', '#D3D3D3', '#DAA520', '#FF69B4', 'black'])))
    st.plotly_chart(graph_color.update_layout(title='Gráfico de quantidade por cor', xaxis_title='Cores', yaxis_title='Quantidade'))

elif option == 'Porcentagem de espécies por raridade':
    graph_rarity = go.Figure(data=go.Pie(labels=rarity_information.index, values=rarity_information.values, marker=dict(colors=[color_discrete_map[x] for x in rarity_information.index])))
    st.plotly_chart(graph_rarity.update_layout(title='Gráfico de raridades', height=600))

elif option == 'Gráfico de dispersão':
    grath_dispersal = px.scatter(pokemon_df, x='height', y='weight', 
                                 title='Relação entre Altura e Peso dos Pokémon',
                                 labels={'height': 'Altura', 'weight': 'Peso'})
    st.plotly_chart(grath_dispersal)
    grath_dispersal2 = px.scatter(pokemon_df, x='speed', y='attack',
                             title='Relação entre Velocidade e Ataque dos Pokémon',
                             labels={'speed': 'Velocidade', 'attack': 'Ataque'})

    # Exiba o gráfico de dispersão.
    st.plotly_chart(grath_dispersal2)







elif option == 'Gráfico de felicidade':

    valores_interessantes = [35, 70, 140, 50, 0]

        # Conta as ocorrências de cada valor.
    contagem_valores = pokemon_df['base_happiness'].value_counts().reindex(valores_interessantes, fill_value=0)

    # Cria um gráfico de pizza para representar as proporções.
    fig = px.pie(values=contagem_valores.values, names=contagem_valores.index,
                title='Proporção de Valores de Base Happiness')

    # Exibe o gráfico.
    st.plotly_chart(fig)
    # Filtra os Pokémon com 'base_happiness' igual a 0 e 'legendary' igual a True.
    pokemon_felicidade_0_lendarios = pokemon_df[(pokemon_df['base_happiness'] == 0) & (pokemon_df['legendary'] == True)]

    # Conta quantos Pokémon atendem a esses critérios.
    contagem_felicidade_0_lendarios = len(pokemon_felicidade_0_lendarios)

    # Filtra os Pokémon com 'legendary' igual a True.
    pokemon_lendarios = pokemon_df[pokemon_df['legendary'] == True]

    # Conta quantos Pokémon são lendários.
    contagem_lendarios = len(pokemon_lendarios)

    # Conta o número total de Pokémon no DataFrame.
    numero_total_pokemon = len(pokemon_df)

    # Cria um DataFrame para o gráfico de barras empilhadas.
    dados_grafico = pd.DataFrame({
        'Categoria': ['Pokémon com Felicidade 0', 'Pokémon Lendários', 'Número Total de Pokémon'],
        'Quantidade': [contagem_felicidade_0_lendarios, contagem_lendarios, numero_total_pokemon]
    })

    # Cria um gráfico de barras empilhadas.
    fig = px.bar(dados_grafico, x='Categoria', y='Quantidade',
                labels={'Categoria': 'Categoria', 'Quantidade': 'Quantidade'},
                title='Relação de Pokémon com Felicidade 0, Pokémon Lendários e Número Total de Pokémon')

    # Exibe o gráfico.
    st.plotly_chart(fig)  
    
elif option == 'Gráfico de relação ataques efetivos e tipos':
    colunas_eficacia_ataque = ['normal_attack_effectiveness', 'fire_attack_effectiveness',
                           'water_attack_effectiveness', 'electric_attack_effectiveness',
                           'grass_attack_effectiveness', 'ice_attack_effectiveness',
                           'fighting_attack_effectiveness', 'poison_attack_effectiveness',
                           'ground_attack_effectiveness', 'fly_attack_effectiveness',
                           'psychic_attack_effectiveness', 'bug_attack_effectiveness',
                           'rock_attack_effectiveness', 'ghost_attack_effectiveness',
                           'dragon_attack_effectiveness', 'dark_attack_effectiveness',
                           'steel_attack_effectiveness', 'fairy_attack_effectiveness']
    

    # Calcula o número de valores únicos em cada coluna.
    fire_type = pokemon_df[pokemon_df['typing'] == 'Fire']
    valores_unicos_por_coluna =  fire_type[colunas_eficacia_ataque].nunique()

    # Cria um gráfico de barras empilhadas interativo usando Plotly e Streamlit.
    fig = px.bar(valores_unicos_por_coluna, x=valores_unicos_por_coluna.index, y=valores_unicos_por_coluna.values)
    fig.update_layout(
        title="Distribuição de Valores Únicos nas Colunas de Eficácia de Ataque",
        xaxis_title="Coluna de Eficácia de Ataque",
        yaxis_title="Número de Valores Únicos",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig)
      
elif option == 'Clusterização':
    pokemon_features, pca1, pca2 = tratamento_dados.clusterizar_df()

    # Criação do gráfico de dispersão com clusters
    graph_clusters = px.scatter(pokemon_features, pca1, pca2, color='cluster_label',
                                title='Gráfico de Clusters após PCA e Método do Cotovelo')
    st.plotly_chart(graph_clusters)
    