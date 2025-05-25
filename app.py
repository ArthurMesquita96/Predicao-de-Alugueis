import streamlit as st
import pandas as pd
from unidecode import unidecode
from datetime import datetime
import ast
import importlib
import numpy as np
from geopy.geocoders import Nominatim
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler



def make_filters(data, list_columns=None, selector=None, selectors_and_columns=None):

    dict_filters = dict()

    if list_columns:
        for col in list_columns:
            filter = selector(col, data[col].unique().tolist())
            dict_filters[col] = filter
    else:
        for col, selector in selectors_and_columns.items():
            match selector[0].__name__:
                case 'selectbox':
                    filter = selector[0](selector[1], data[col].sort_values().unique().tolist(), index=None)
                    dict_filters[col] = filter
                case 'number_input':
                    filter = selector[0](label=selector[1])
                    dict_filters[col] = filter
                case 'pills':
                    filter = selector[0](label=selector[1],options=data[col].sort_values().unique().tolist(), key=f"filter_{col}")
                    dict_filters[col] = filter
                case 'text_input':
                    filter = selector[0](label=selector[1],key=f"filter_{col}")
                    dict_filters[col] = filter

    return dict_filters

def feature_engineering(data):
    # year
    data['year'] = datetime.today().year

    # month
    data['month'] = datetime.today().month

    # day
    data['day'] = datetime.today().day

    # week of day
    data['week_of_year'] = datetime.today().isocalendar().week

    return data

def get_dados_wikipedia(data):
    dados_curitiba = pd.read_csv('params/data/dados_curitiba.csv')

    ajustes = {
        'ecoville':'mossungue',
        'champagnat':'bigorrilho',
        'bigoriilho':'bigorrilho',
        'alto da rua xv': 'alto da xv',
        'novo mundo ': 'novo mundo',
        'jd. das americas': 'jardim das americas',
        'cic':'cidade industrial'
    }

    # ajustando nome dos bairros
    data['bairro'] = data['bairro'].replace(ajustes)
    data['bairro'] = data['bairro'].apply(lambda x: unidecode(x))

    # cruzando dados da wikipedia 
    data = pd.merge(data, dados_curitiba[['bairro', 'regiao', 'qtd_domicilios_particulares', 'renda_media_responsaveis_domicilio']], on = 'bairro', how ='left')
    data = data.dropna()

    # tratando dados da wikipedia
    data['qtd_domicilios_particulares'] = data['qtd_domicilios_particulares'].astype('int64')
    data['renda_media_responsaveis_domicilio'] = data['renda_media_responsaveis_domicilio'].astype('float64')

    return data


def get_lat_long(lat_long, coordenada):
    if lat_long:
        try: 
            lat_long_list = ast.literal_eval(str(lat_long))
            match coordenada:
                case 'latitude':
                    return lat_long_list[0]
                case 'longitude':
                    return lat_long_list[1]
        except:
            pass

def obter_lat_long(endereco):

    def get_localization(endereco):
        geolocator = Nominatim(user_agent="meu_app")
        localizacao = geolocator.geocode(endereco)

        return localizacao

    def return_lat_long(lat_long):
        return [lat_long.latitude, lat_long.longitude ]
    
    lat_long = get_localization(endereco)

    if lat_long:
        return return_lat_long(lat_long)
    else:
        # print('quebrando endereco')
        partes_do_endereco = endereco.split(', ')
        # print(partes_do_endereco)
        qtd_partes_endereco = len(partes_do_endereco)
        for i in range(1, qtd_partes_endereco+1):
            # print(qtd_partes_endereco)
            # print(f'partes_do_endereco: {partes_do_endereco[0:len(partes_do_endereco) - i]}')
            # print(f'i: {i}')
            try:
                novo_endereco = ' '.join(partes_do_endereco[0:len(partes_do_endereco) - i])
                # print(f'novo_endereco: {novo_endereco}')
                lat_long = get_localization(novo_endereco)
                if lat_long:
                    return return_lat_long(lat_long)
                else:
                    pass
            except:
                pass

def save_picked_file(file, name):
    return pickle.dump(file,open(f'params/{name}.pkl','wb'))

def load_picked_file(name):
    return pickle.load(open(f'params/{name}.pkl','rb'))

def coordenadas_geograficas(data):
    data['latitude'] = data['endereco'].apply(lambda x: get_lat_long(obter_lat_long(x), 'latitude'))
    data['longitude'] = data['endereco'].apply(lambda x: get_lat_long(obter_lat_long(x), 'longitude'))

    return data

def robust_scaler(df, column, train=True):

    data = df.copy()

    if train:
        scaler = RobustScaler()
        data[column] = scaler.fit_transform(data[[column]])

        save_picked_file(scaler, f'preparation/robust_scaler_{column}')

    else:

        scaler = load_picked_file(f'preparation/robust_scaler_{column}')
        data[column] = scaler.transform(data[[column]])

    return data

def standart_scaler(df, column, train=True):

    data = df.copy()

    if train:
        scaler = StandardScaler()
        data[column] = scaler.fit_transform(data[[column]])

        save_picked_file(scaler, f'preparation/standart_scaler{column}')

    else:

        scaler = load_picked_file(f'preparation/standart_scaler{column}')
        data[column] = scaler.transform(data[[column]])

    return data

def target_encode(df, column, train = True):
    
    data = df.copy()

    if train:
        media_por_categoria = data[[column,'valor_total']].groupby(column).agg(total=('valor_total','mean')).reset_index()
        media_por_categoria['total'] = round(media_por_categoria['total'],2)
        media_por_categoria = media_por_categoria.set_index(column)['total'].to_dict()

        data[column] = data[column].map(media_por_categoria)
        save_picked_file(media_por_categoria, f'preparation/target_encode_{column}')

    else:
        media_por_categoria = load_picked_file(f'preparation/target_encode_{column}')
        data[column] = data[column].map(media_por_categoria)
        data.loc[data[column].isna(), column] = data[column].mean()

    return data

def dummie_encode(df, column):
    
    data = df.copy()

    data[column] = data[column].map({'Sim':1, 'Não': 0})

    return data

def ciclycal_encode(df, column):

    data = df.copy()

    match column:
        case 'year':
            max_value = 2
        case 'month':
            max_value = 12
        case 'day':
            max_value = 365        
        case 'week_of_year':
            max_value = 52

    data[f'{column}_sin'] = data[f'{column}'].apply(lambda x: np.sin(x*(2*np.pi/max_value)))
    data[f'{column}_cos'] = data[f'{column}'].apply(lambda x: np.cos(x*(2*np.pi/max_value)))

    return data

def preparacao_dos_dados(df, dict_preparation, is_train=True):

    data = df.copy()

    for column, preparation in dict_preparation.items():
        try:
            match preparation:
                case 'standart_scaler':
                    data = standart_scaler(data, column, train=is_train)
                case 'robust_scaler':
                    data = robust_scaler(data, column, train=is_train)
                case 'target_encode':
                    data = target_encode(data, column, train=is_train)
                case 'dummie_encode':
                    data = dummie_encode(data, column)   
                case 'ciclycal_encode':
                    data = ciclycal_encode(data, column)  
                case np.log1p:
                    data[f'{column}'] = np.log1p(data[f'{column}'])
        except:
            pass

    return data     

st.set_page_config(page_title="Predição de Alugueis", layout='wide')


df = pd.read_pickle('params/data/df_exp.pkl')

container1 = st.container(border=True)


selectors_and_columns1 = {
    'area': [st.number_input, 'Qual a área do imóvel em m²?'],
    'endereco': [st.text_input, 'Qual o endereço?'],
    'bairro': [st.selectbox, 'Qual o bairro?'],
    'quartos': [st.selectbox, 'Qual o número de quartos?']
}
selectors_and_columns2 = {
    'suites': [st.selectbox, 'Qual o número de suítes'],
    'banheiros': [st.selectbox, 'Qual o número de **banheiros**?'],
    'vagas_garagem': [st.selectbox, 'Quantas vagas de garagem?'],
    'sacada': [st.pills, 'Imóvel possui sacada?']
}

selectors_and_columns3 = {
    'hidromassagem': [st.pills, 'O imóvel possúi banheira de hidromassagem?'],
    'mobiliado': [st.pills, 'Imóvel é mobiliado?'],
    'mobilia_planejada': [st.pills, 'A mobilia é planejada?'],
    'imovel_decorado': [st.pills, 'O imóvel é decorado?']
}
selectors_and_columns4 = {
    'totalmente_mobiliado': [st.pills, 'O imóvel é totalmente mobiliado?']
}



selectors_and_columns5 = {
    'piscina': [st.pills, 'Condomínio possui piscina?'],
    'academia': [st.pills, 'Condomínio possui academia?'],
    'churrasqueira': [st.pills, 'Condomínio possui churrasqueira?']
}

selectors_and_columns6 = {
    'salao_de_festas': [st.pills, 'Condomínio possui salao de festas?'],
    'espaco_coworking': [st.pills, 'Condomínio possui espaço coworking?'],
    'playground': [st.pills, 'Condomínio possui playground?']
}

selectors_and_columns7 = {
    'quadra_esportes': [st.pills,'Condomínio possui quadra de esportes?'],
    'salao_de_jogos': [st.pills,'Condomínio possui salão de jogos?']
}


st.header('Dados do Imóvel')
st.write('Preencha os dados referentes ao imóvel')

col1, col2, col3, col4 = st.columns(4, border=True)

with col1:
    filters1 = make_filters(df, selectors_and_columns=selectors_and_columns1)

with col2:
    filters2 = make_filters(df, selectors_and_columns=selectors_and_columns2)

with col3:
    filters3 = make_filters(df, selectors_and_columns=selectors_and_columns3)

with col4:
    filters4 = make_filters(df, selectors_and_columns=selectors_and_columns4)

container2 = st.container(border=True)

st.header('Dados do Condomínio')
st.write('Preencha os dados referentes ao condomínio')

col5, col6, col7, col8 = st.columns(4, border=True)

with col5:
    filters5 = make_filters(df, selectors_and_columns=selectors_and_columns5)

with col6:
    filters6 = make_filters(df, selectors_and_columns=selectors_and_columns6)

with col7:
    filters7 = make_filters(df, selectors_and_columns=selectors_and_columns7)



st.header('Estimação de Valores')
st.write('Clique no botão para estimar o valor do aluguel + condomínio do imóvel')

fazer_predicao = st.button("Estimar valores")

if fazer_predicao:
    
    filters = {}
    filters.update(filters1)
    filters.update(filters2)
    filters.update(filters3)
    filters.update(filters4)
    filters.update(filters5)
    filters.update(filters6)
    filters.update(filters7)
    input = pd.DataFrame([filters])

    if input.isna().sum().sum() > 0:
        st.write('Por favor, preencha todas as características do imóvel acima para realizar a predição')
    else:

        mensagem = 'Processando dados...'
        st.write(mensagem)
        input = get_dados_wikipedia(input)
        input = feature_engineering(input)

        input['latitude'] = input['endereco'].apply(lambda x: get_lat_long(obter_lat_long(x), 'latitude'))
        input['longitude'] = input['endereco'].apply(lambda x: get_lat_long(obter_lat_long(x), 'longitude'))

        mensagem = 'Carregando modelo...'
        st.write(mensagem)
        dict_preparation = load_picked_file(name='preparation/dict_data_preparation')
        features_selected = load_picked_file(name='preparation/features_selected')
        best_model = load_picked_file(name='models/best_model')

        input_transformed = preparacao_dos_dados(df=input, dict_preparation=dict_preparation, is_train=False)

        mensagem = 'Fazendo predição'
        st.write(mensagem)
        y_pred = best_model.predict(input_transformed[features_selected])

        valor_em_reais = round(float(y_pred[0]), 2)

        mensagem = f'Valor do Aluguel + Condomínio estimado em R$: {valor_em_reais}'
        st.write(mensagem)

else:
    pass