import streamlit as st
import pandas as pd
import random

with open('assets\css\style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

pokemon_df = pd.read_parquet("data\pokemon.parquet")
# CODIGO QUE RETIRA OS DUPICADAS DE NÚMERO DA POKEDEX
pokemon_df = pokemon_df.drop_duplicates(subset='pokedex_number')
# adicionando os links das imagens em uma nova coluna no banco
pokemon_df['image'] = ''
pokemon_df['image'] = pokemon_df['pokedex_number'].apply(lambda x: f'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{x}.png')


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


def search_by_type(pokemon_type):
    return pokemon_df.loc[pokemon_df['typing'].str.contains(pokemon_type), ['pokedex_number', 'name', 'genus', 'typing', 'image']]


def search_by_shape(pokemon_shape):
    return pokemon_df.loc[pokemon_df['shape'].str.contains(pokemon_shape), ['pokedex_number', 'name', 'genus', 'shape', 'image']]


def search_by_gen(pokemon_gen):
    return pokemon_df.loc[pokemon_df['gen_introduced'] == pokemon_gen, ['pokedex_number', 'name', 'genus', 'gen_introduced', 'image']]


def search_by_color(pokemon_color):
    return pokemon_df.loc[pokemon_df['primary_color'].str.contains(pokemon_color), ['pokedex_number', 'name', 'genus', 'primary_color', 'image']]


def search_is_baby():
    return pokemon_df.loc[pokemon_df['baby_pokemon'] == True, ['pokedex_number', 'name', 'genus', 'baby_pokemon', 'image']]


def search_is_legendary():
    return pokemon_df.loc[pokemon_df['legendary'] == True, ['pokedex_number', 'name', 'genus', 'legendary', 'image']]


def search_is_mythical():
    return pokemon_df.loc[pokemon_df['mythical'] == True, ['pokedex_number', 'name', 'genus', 'mythical', 'image']]


# MAIN PAGE START --
st.image("assets\icons\logo2.png")
st.markdown('<h1 class="site-title">Visualização do DF com filtros</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="site-subt">Selecione o filtro de busca:</h3>', unsafe_allow_html=True)

with st.expander("Buscar pelo tipo"):
    types = pokemon_df.loc[~pokemon_df['typing'].str.contains('~'), 'typing'].unique()
    pokemon_type = st.selectbox(
        'Pesquise pelo tipo do seu pokemon:',
        (types))
    if pokemon_type:
        result = search_by_type(pokemon_type)
        generate_data_editor(result)

with st.expander("Buscar pelo formato"):
    shapes = pokemon_df['shape'].unique()
    pokemon_shape = st.selectbox(
        'Pesquise pelo formato com que ele se parece:',
        (shapes))
    if pokemon_shape:
        result1 = search_by_shape(pokemon_shape)
        generate_data_editor(result1)

with st.expander("Buscar pela geração"):
    gens = pokemon_df['gen_introduced'].unique()
    pokemon_gen = st.selectbox(
        'Pesquise pela geração em que ele foi lançado:',
        (gens))
    if pokemon_gen:
        result2 = search_by_gen(pokemon_gen)
        generate_data_editor(result2)

with st.expander("Buscar pela cor"):
    colors = pokemon_df['primary_color'].unique()
    pokemon_color = st.selectbox(
        'Pesquise pela sua cor principal:',
        (colors))
    if pokemon_color:
        result3 = search_by_color(pokemon_color)
        generate_data_editor(result3)

with st.expander("Buscar os pokémon por raridade"):
    rarity = st.radio(
        "",
        ["baby_pokemon", "legendary", "mythical"], 
        horizontal=True
    )
    if rarity == 'baby_pokemon':
        result4 = search_is_baby()
        generate_data_editor(result4)

    elif rarity == 'legendary':
        result5 = search_is_legendary()
        generate_data_editor(result5)

    else:
        result6 = search_is_mythical()
        generate_data_editor(result6)


with st.expander("Busca com mais de um parâmetro"):
    options = st.multiselect(
    'Selecione todas as opções que deseja utilizar na busca',
    ['tipo', 'formato', 'geração', 'cor'])

    if 'tipo' in options:
        pokemon_type_radio = st.radio(
            'Informe o tipo',
            (types),
            key='tipo',
            help='Selecione o tipo desejado',
            horizontal=True
        )
    if 'formato' in options:
        pokemon_shape_radio = st.radio(
            'Informe o formato',
            (shapes),
            key='formato',
            help='Selecione o formato desejado',
            horizontal=True
        )
    if 'geração' in options:
        pokemon_generation_radio = st.radio(
            'Informe a geração',
            (gens),
            key='geração',
            help='Selecione a geração desejada',
            horizontal=True
        )
    if 'cor' in options:
        pokemon_color_radio = st.radio(
            'Informe a cor',
            (colors),
            key='cor',
            help='Selecione a cor desejada',
            horizontal=True
        )
    if st.button('Buscar'):
        union_list = []
        if 'tipo' in options and pokemon_type_radio:
            union_list.append(search_by_type(pokemon_type_radio))
        if 'formato' in options and pokemon_shape_radio:
            union_list.append(search_by_shape(pokemon_shape_radio))
        if 'geração' in options and pokemon_generation_radio:
            union_list.append(search_by_gen(pokemon_generation_radio))
        if 'cor' in options and pokemon_color_radio:
            union_list.append(search_by_color(pokemon_color_radio))

        radios_selcted = []
        # Verificando cada elemento da lista union_list
        for element in union_list:
            if element is not None:
                radios_selcted.append(element)

        if len(radios_selcted) > 0:
            count = 0
            result_df = union_list[0]
            for i in range(1, len(union_list)):
                if union_list[i] is not None:
                    count += 1
                    result_df = result_df.merge(union_list[i], on="pokedex_number", how="inner")
                    if count >= 1:
                        result_df = result_df.rename(columns={'image_x': 'image', 'genus_x': 'genus', 'name_x': 'name'})
                        result_df = result_df.filter(regex='^(?!.*(_y)$)')
        generate_data_editor(result_df)

        

