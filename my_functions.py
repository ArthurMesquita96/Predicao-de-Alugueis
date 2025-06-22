import numpy as np
import pandas as pd
import requests
import bs4
from unidecode import unidecode
from google.cloud import storage
import re
import ast
import pickle
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn import metrics as mt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans

from geopy.geocoders import Nominatim


def get_data(bucket_name:str, imobiliarias:str = ['apolar', 'cilar'],by:str = ['date','date_diff'], dates:list = [], date_diff:int = 2):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    files_on_bucket = [i.name for i in bucket.list_blobs()]
    files = pd.DataFrame(files_on_bucket, columns=['name'])
    files['date'] = pd.to_datetime(files['name'].apply(lambda f: f.split(' - ')[0]))
    files['imobiliaria'] = files['name'].apply(lambda f: f.split(' - ')[-1].replace('.csv',''))

    match by:
        case 'date':
            files = files.loc[files['date'].isin(dates)]
            files = files.loc[files['imobiliaria'].isin(imobiliarias)]
        case 'date_diff':
            dates = files['date'].sort_values(ascending=False).drop_duplicates().reset_index(drop=True)[:date_diff].tolist()
            files = files.loc[files['date'].isin(dates)]
            files = files.loc[files['imobiliaria'].isin(imobiliarias)]
    
    df_full = pd.DataFrame()

    for file_name in files['name'].tolist():
        try:
            df_aux = pd.read_csv(f'gs://{bucket_name}/{file_name}')
            df_full = pd.concat([df_full, df_aux], axis = 0)
        except:
            pass

    df_full = df_full.reset_index(drop=True)

    return df_full

def get_all_dates(bucket_name):

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    datas = set([i.name.split(' - ')[0] for i in bucket.list_blobs()])

    return datas

def get_infos_curitiba():

    ## Request site
    response = requests.get('https://pt.wikipedia.org/wiki/Lista_de_bairros_de_Curitiba')

    ## beautiful soup object
    soup = bs4.BeautifulSoup(response.content, 'html.parser')

    # tabelas da wikipedia
    infos_tabela = soup.findAll('table', {'class','wikitable'})

    bairros_info_list = []

    ## para cada tabela
    for tabela in infos_tabela:

        # colunas
        columns = [i.text.replace('\n','') for i in tabela.findAll('th')]

        # valores da tabela
        regiao = ' '.join([i.text.replace('\xa0','').replace('\n','') for i in tabela.findAll('td')][0].replace('Bairros oficiais de Curitiba - Regional ','').split(' ')[:-1])
        table_values = [i.text.replace('\xa0','').replace('\n','') for i in tabela.findAll('td')][1::]
        table_values_list = [] 
        for i in range(0,len(table_values),7): 
            table_values_list.append(table_values[i:i+7])

        # preenchenco dicionário
        for b in table_values_list:

            bairros_info_dict = {}
            bairros_info_dict['regiao'] = regiao
            bairros_info_dict['bairro'] = b[0]
            bairros_info_dict['area_bairro'] = b[1].replace(',','.')
            bairros_info_dict['qtd_homens'] = b[2].replace(',','.')
            bairros_info_dict['qtd_mulheres'] = b[3].replace(',','.')
            bairros_info_dict['total'] = b[4].replace(',','.')
            bairros_info_dict['qtd_domicilios_particulares'] = b[5].replace(',','.')
            bairros_info_dict['renda_media_responsaveis_domicilio'] = b[6].replace(',','.')

            bairros_info_list.append(bairros_info_dict)

    data = pd.DataFrame(bairros_info_list)

    ajustes = {
    'ecoville':'mossungue',
    'champagnat':'bigorrilho',
    'alto da rua xv': 'alto da xv',
    'novo mundo ': 'novo mundo',
    'jd. das americas': 'jardim das americas'
    }

    data['bairro'] = data['bairro'].apply(lambda x: unidecode(x.lower()).strip())
    data['bairro'] = data['bairro'].replace(ajustes)
    data = data.drop_duplicates('bairro').reset_index(drop=True)
    
    return data

def feature_engeniering(data, columns_selected):

    df = data.copy()

    df_aux = df.groupby('link').agg(data_min = ('data_coleta','min'), data_max=('data_coleta','max')).reset_index()
    df_aux['tempo_ate_locacao'] = (pd.to_datetime(df_aux['data_max']) - pd.to_datetime(df_aux['data_min'])).dt.days
    df_aux = df_aux.loc[df_aux['tempo_ate_locacao'] > 0]
    
    df = df[columns_selected].sort_values('data_coleta').drop_duplicates('link', keep = 'first')

    df_aux = pd.merge(df_aux, df, on = 'link', how = 'left')

    df_aux = df_aux[columns_selected + ['tempo_ate_locacao']]

    return df_aux

def barplot(group:str, 
            agg:str, 
            agg_name:str, 
            data:pd.DataFrame, 
            agg_func:str, 
            figure= plt.figure, 
            title_font_size:int =10, 
            figsize=(10,5),
            title:str='',
            subplot:plt.subplot = None, 
            grid:list = None, 
            orient:str='h',
            label=True,
            rotation_label:int = 45,
            position_label:str = 'center',
            color_label:str = 'white',
            size_label:str = 'small',
            fmt:str = '%.0f',
            sort: bool = True, 
            hue:str = None,
            stacked:bool = False):
    
    group_list = [group]
    if hue:
        group_list.append(hue)

    # group data
    aux = data[group_list + [agg]].groupby(group_list).agg(agg_func).reset_index().rename(columns={agg:agg_name})

    if sort:
        aux = aux.sort_values(agg_name, ascending=False)
        
    # plot
    if subplot:
        subplot(grid)
    else:
        figure(figsize=figsize)

    # plot configs
    plt.title(title, fontsize=title_font_size)
    plt.xticks(rotation = rotation_label)

    # figure
    if orient == 'h':
        g = sns.barplot(x = group, y = agg_name, hue = hue, dodge = not stacked, data = aux)
    elif orient == 'v':
        g = sns.barplot(y = group, x = agg_name, hue = hue, dodge = not stacked, data = aux)
    else:
        raise("Variável 'orient' informada não é válida")

    if label:
        for i in g.containers:
            g.bar_label(i, color = color_label, label_type=position_label, fontsize = size_label, fmt = fmt)
    else:
        pass

def tratamento_dados_cilar(data):

    df = data.copy()
    def formata_valores(valores):
        return valores.str.replace('.','').apply(lambda x: x if pd.isna(x) else x.split(',')[0]).astype('float64')

    def extrai_valores_string(string,substring):

        # Padronizar a expressão regular para encontrar a área total
        padrao = f'{substring} (\d+)'

        # Encontrar a área total usando regex
        area_total = re.search(padrao, string)

        if area_total:
            # Extrair o valor numérico da área total
            valor_area = area_total.group(1)
            
            # Remover vírgulas e converter para float
            valor_area = int(valor_area.replace(',', '.'))
            
        else:
            valor_area = np.nan
        
        return valor_area

    def verifica_existencia_palavras(palavras, texto):
        return any(palavra in texto for palavra in palavras)

    df['detalhes'] = df['detalhes'].apply(lambda x: x if pd.isna(x) else ast.literal_eval(x))
    df['detalhes'] = df['detalhes'].apply(lambda x: ' '.join(x).replace('Características do imóvel ','').strip() if isinstance(x,list) else x)

    # corrigindo alguns valores de iptu na coluna de condomínio
    df.loc[df['condominio'].str.contains('IPTU', na=False), 'iptu'] = df.loc[df['condominio'].str.contains('IPTU', na=False), 'condominio']
    df.loc[df['condominio'].str.contains('IPTU', na=False), 'condominio'] = np.nan

    df['aluguel'] = formata_valores(df['aluguel'].str.replace('AluguelR$','')).fillna(0)
    df['condominio'] = formata_valores(df['condominio'].str.replace('Condominio  R$','')).fillna(0)
    df['iptu'] = formata_valores(df['iptu'].str.replace('IPTU  R$','')).fillna(0)

    ## Detalhes do imóvel
    df['area'] = df['detalhes'].apply(lambda x: 0 if pd.isna(x) else extrai_valores_string(x,'Área Total')).fillna(0)
    df['quartos'] = df['detalhes'].apply(lambda x: 0 if  pd.isna(x) else extrai_valores_string(x,'Quartos')).fillna(0)
    df['suites'] = df['detalhes'].apply(lambda x: 0 if  pd.isna(x) else extrai_valores_string(x,'Suítes')).fillna(0)
    df['banheiros'] = df['detalhes'].apply(lambda x: 0 if  pd.isna(x) else extrai_valores_string(x,'Banheiros')).fillna(0)
    df['andar'] = df['detalhes'].apply(lambda x: 0 if  pd.isna(x) else extrai_valores_string(x,'Andar')).fillna(0)
    df['vagas_garagem'] = df['mais_detalhes_imovel'].apply(lambda x: 0 if  pd.isna(x) else extrai_valores_string(x,'Vagas de garagem:')).fillna(0)

    # Localidade
    df['bairro'] = df['endereco'].apply(lambda x: x if pd.isna(x) else unidecode(x.split(' - ')[-2].capitalize()))
    df['cidade'] = df['endereco'].apply(lambda x: x if pd.isna(x) else unidecode(x.split(' - ')[-1].capitalize()))

    # Atributos do imóvel e condomínio
    df['mobiliado'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['mobilia', 'mobiliado', 'semi-mobiliado','sofa', 'armario', 'armarios', 'tv', 'cama', 'mesa', 'cadeiras', 'eletrodomesticos'], unidecode(x.lower())) else 'Não')
    df['mobilia_planejada'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['armarios planejados', 'planejados', 'planejado', 'sob medida'], unidecode(x.lower())) else 'Não')
    df['piscina'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if 'piscina' in unidecode(x.lower()) else 'Não')
    df['academia'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['academia', 'fitness', 'espaco fitness', 'sala fitness', 'ginastica'], unidecode(x.lower())) else 'Não')
    df['sacada'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if 'sacada' in unidecode(x.lower()) else 'Não')
    df['churrasqueira'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['churrasqueira', 'espaco churrasco', 'churrasco'], unidecode(x.lower())) else 'Não')
    df['salao_de_festas'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['salao de festa', 'salao de festas'], unidecode(x.lower())) else 'Não')
    df['salao_de_jogos'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['salao de jogos'], unidecode(x.lower())) else 'Não')
    df['espaco_coworking'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['coworking', 'sala coworking'], unidecode(x.lower())) else 'Não')
    df['quadra_esportes'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['quadra de esportes',  'quadra coberta', 'quadra poliesportiva', 'poliesportiva'], unidecode(x.lower())) else 'Não')
    df['playground'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['playground'], unidecode(x.lower())) else 'Não')
    df['lavanderia'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['lavanderia'], unidecode(x.lower())) else 'Não')
    df['espaco_pet'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['espaco pet', 'pet space'], unidecode(x.lower())) else 'Não')
    df['imovel_decorado'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['decorado', 'charmoso', 'charmosa', 'design', 'layout'], unidecode(x.lower())) else 'Não')
    df['totalmente_mobiliado'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['100% mobiliado', 'todo mobiliado',  'studio mobiliado'], unidecode(x.lower())) else 'Não')
    df['hidromassagem'] = df['caracteristicas_imovel'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['hidromassagem'], unidecode(x.lower())) else 'Não')


    return df

def tratamento_dados_apolar(data):

    df = data.copy()
    def busca_substring(substring, string_list):
        result = np.nan
        for s in string_list:
            if substring in s:
                try:
                    result = re.findall(r'\s(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)', s)[0]
                except:
                    result = s
                break
                
        return result

    def separa_valores_imovel(string):

        # Padrao regex para encontrar nome e valor monetário
        padrao = r'(\w+)\sR\$\s(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)'

        # Encontrar todas as correspondências na string
        correspondencias = re.findall(padrao, string)

        # Imprimir os resultados
        list_values = []
        for correspondencia in correspondencias:
            nome, valor = correspondencia
            list_values.append(f'{nome}: {valor}')
        
        return list_values

    def formata_valores(valores):
        return valores.str.strip().str.replace('.','').apply(lambda x: x if pd.isna(x) else x.split(',')[0]).astype('float64')

    def verifica_existencia_palavras(palavras, texto):
        return any(palavra in texto for palavra in palavras)

    df['titulo'] = df['titulo'].apply(lambda x: x if pd.isna(x) else x.replace('\n','').strip())
    df['endereco'] = df['endereco'].apply(lambda x: x if pd.isna(x) else x.replace('\n','').strip())
    df['descricao'] = df['descricao'].apply(lambda x: x if pd.isna(x) else x.replace('\n','').strip())

    df['bairro'] = df['endereco'].str.strip().apply(lambda x: x if pd.isna(x) else unidecode(x.replace('\n','').strip().split(', ')[-1].split(' - ')[0].capitalize()))
    df['cidade'] = df['endereco'].str.strip().apply(lambda x: x if pd.isna(x) else unidecode(x.replace('\n','').strip().split(', ')[-1].split(' - ')[-1].capitalize()))

    # Valores
    df['aluguel'] = df['valores'].apply(lambda x: x if pd.isna(x) else 
                        x.split(', ,')[0].replace('R$ ','').strip() if "Aluguel" not in x else
                        x.split(', ,')[0].replace('R$ ','').replace('Aluguel ','').strip() if "Aluguel" in x else
                        x)
    df['aluguel'] = df['aluguel'].apply(lambda x: x if pd.isna(x) else x.split(' ')[0])
    df['condominio'] = df['valores'].apply(lambda x: busca_substring('Condomínio', separa_valores_imovel(x)) if not pd.isna(x) else x)
    df['iptu'] = df['valores'].apply(lambda x: busca_substring('IPTU', separa_valores_imovel(x)) if not pd.isna(x) else x)
    df['seguro_incendio'] = df['valores'].apply(lambda x: busca_substring('Incêndio', separa_valores_imovel(x)) if not pd.isna(x) else x)

    # formatando valores
    df['aluguel'] = formata_valores(df['aluguel'])
    df['condominio'] = formata_valores(df['condominio']).fillna(0)
    df['iptu'] = formata_valores(df['iptu']).fillna(0)
    df['seguro_incendio'] = formata_valores(df['seguro_incendio']).fillna(0)

    # Atributos
    df['area'] = df['atributos'].apply(lambda x: x if pd.isna(x)  else busca_substring('m²', x.split(', '))).str.replace('m²','')
    df['banheiros'] = df['atributos'].apply(lambda x: x if pd.isna(x)  else busca_substring('banheiro', x.split(', '))).str.replace('banheiro','').str.replace('s','').fillna(0)
    df['quartos'] = df['atributos'].apply(lambda x: x if pd.isna(x)  else busca_substring('quarto', x.split(', '))).str.replace('quarto','').str.replace('s','').fillna(0)
    df['suites'] = df['atributos'].apply(lambda x: x if pd.isna(x)  else busca_substring('suite', x.split(', '))).str.replace('suite','').str.replace('s','').fillna(0)
    df['vagas_garagem'] = df['atributos'].apply(lambda x: x if pd.isna(x) else busca_substring('vaga', x.split(', '))).str.replace('vaga','').str.replace('s','').fillna(0)

    # Detalhes do imóvel/condomínio
    df['mobiliado'] = df['descricao'].apply(lambda x: np.nan if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['mobilia', 'moveis planejados', 'mobiliado', 'semi-mobiliado','sofa', 'armario', 'armarios', 'tv', 'cama', 'mesa', 'cadeiras', 'eletrodomesticos'], unidecode(x.lower())) else 'Não')
    df['mobilia_planejada'] = df['descricao'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['armarios planejados', 'planejados', 'planejado', 'sob medida'], unidecode(x.lower())) else 'Não')
    df['piscina'] = df['descricao'].apply(lambda x: np.nan if pd.isna(x) else 'Sim' if 'piscina' in unidecode(x.lower()) else 'Não')
    df['academia'] = df['descricao'].apply(lambda x: np.nan if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['academia', 'fitness', 'espaco fitness', 'sala fitness', 'ginastica'], unidecode(x.lower())) else 'Não')
    df['sacada'] = df['descricao'].apply(lambda x: np.nan if pd.isna(x) else 'Sim' if 'sacada' in unidecode(x.lower()) else 'Não')
    df['churrasqueira'] = df['descricao'].apply(lambda x: np.nan if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['churrasqueira', 'espaco churrasco', 'churrasco'], unidecode(x.lower())) else 'Não')
    df['salao_de_festas'] = df['descricao'].apply(lambda x: np.nan if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['salao de festa', 'salao de festas'], unidecode(x.lower())) else 'Não')
    df['salao_de_jogos'] = df['descricao'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['salao de jogos'], unidecode(x.lower())) else 'Não')
    df['espaco_coworking'] = df['descricao'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['coworking', 'sala coworking'], unidecode(x.lower())) else 'Não')
    df['quadra_esportes'] = df['descricao'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['quadra de esportes',  'quadra coberta', 'quadra poliesportiva', 'poliesportiva'], unidecode(x.lower())) else 'Não')
    df['playground'] = df['descricao'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['playground'], unidecode(x.lower())) else 'Não')
    df['lavanderia'] = df['descricao'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['lavanderia'], unidecode(x.lower())) else 'Não')
    df['espaco_pet'] = df['descricao'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['espaco pet', 'pet space'], unidecode(x.lower())) else 'Não')
    df['imovel_decorado'] = df['descricao'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['decorado', 'charmoso', 'charmosa', 'design', 'layout'], unidecode(x.lower())) else 'Não')
    df['totalmente_mobiliado'] = df['descricao'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['100% mobiliado', 'todo mobiliado', 'studio mobiliado'], unidecode(x.lower())) else 'Não')
    df['hidromassagem'] = df['descricao'].apply(lambda x: x if pd.isna(x) else 'Sim' if verifica_existencia_palavras(['hidromassagem'], unidecode(x.lower())) else 'Não')

    return df

def tratamento_dados_razao(data):

    df = data.copy()
    # 'titulo'
    df['titulo'] = df['titulo'].str.strip()

    # 'endereco'
    df['endereco'] = df['endereco'].str.strip()
    df['bairro'] = df['endereco'].apply(lambda x: x.split(' - ')[1])
    df['cidade'] = df['endereco'].apply(lambda x: x.split(' - ')[-1])

    # 'condominio'
    df['condominio'] = df['condominio'].apply(lambda x: 0 if pd.isna(x) else np.nan if x == '' else x.replace('Cond. ','').replace('R$ ','').split(',')[0].replace('Sob consulta','0').replace(' ','')).fillna(0).astype('float64')

    # 'iptu'
    df['iptu'] = df['iptu'].apply(lambda x: x if pd.isna(x) else np.nan if x == '' else x.replace('Sobconsulta','0').replace('IPTU  R$ ', '').split(',')[0]).fillna(0).astype('float64')

    # 'aluguel'
    df['aluguel'] = df['aluguel'].apply(lambda x: np.nan if not re.search(r'\d', x) else x.replace('R$ ','').replace('.','')).astype('float64')

    # 'itens_imovel'

    # 'descricao'

    df['atributos'] = df['atributos'].apply(lambda x: x if pd.isna(x) else ast.literal_eval(x))
    # formatando valores
    df['area'] = df['atributos'].apply(lambda x: x[4].split(' ')[0]).replace('(--)',0)
    df['quartos'] = df['atributos'].apply(lambda x: x[0]).replace('(--)',0)
    df['suites'] = df['atributos'].apply(lambda x: x[1]).replace('(--)',0)
    df['banheiros'] = df['atributos'].apply(lambda x: x[2]).replace('(--)',0)
    df['vagas_garagem'] = df['atributos'].apply(lambda x: x[3]).replace('(--)',0)

    df['mobiliado'] = df['descricao'].apply(lambda x: np.nan if isinstance(x,float) else 'Sim' if 'mobiliado' in unidecode(x.lower()) else 'Não')
    df['piscina'] = df['descricao'].apply(lambda x: np.nan if isinstance(x,float) else 'Sim' if 'piscina' in unidecode(x.lower()) else 'Não')
    df['academia'] = df['descricao'].apply(lambda x: np.nan if isinstance(x,float) else 'Sim' if 'academia' in unidecode(x.lower()) else 'Não')
    df['sacada'] = df['descricao'].apply(lambda x: np.nan if isinstance(x,float) else 'Sim' if 'sacada' in unidecode(x.lower()) else 'Não')
    df['churrasqueira'] = df['descricao'].apply(lambda x: np.nan if isinstance(x,float) else 'Sim' if 'churrasqueira' in unidecode(x.lower()) else 'Não')
    df['salao_de_festas'] = df['descricao'].apply(lambda x: np.nan if isinstance(x,float) else 'Sim' if 'salao de festas' in unidecode(x.lower()) else 'Não')
    

    return df

def salvar_grafico(nome_arquivo, pasta_destino, figura=None, formato='png', dpi=300):
    """
    Salva um gráfico matplotlib em uma pasta específica com nome definido.

    Parâmetros:
        nome_arquivo (str): nome do arquivo de imagem (sem extensão).
        pasta_destino (str): caminho da pasta onde o gráfico será salvo.
        figura (matplotlib.figure.Figure, opcional): figura a ser salva. Se None, usa a figura atual.
        formato (str): formato da imagem (ex: 'png', 'jpg'). Padrão: 'png'.
        dpi (int): resolução da imagem. Padrão: 300.
    """
    if figura is None:
        figura = plt.gcf()  # pega a figura atual
    
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
    
    caminho_completo = os.path.join(pasta_destino, f"{nome_arquivo}.{formato}")
    figura.savefig(caminho_completo, format=formato, dpi=dpi, bbox_inches='tight')
    print(f"Gráfico salvo em: {caminho_completo}")
    return caminho_completo

def plot_matrix(data, columns_features, n_rows, n_cols, plot, sort_by= None, plot_kwargs = {}, loop_feature = None, figure = None, figsize = (15,15), label = True, save_image = False, nome_imagem='imagem', formato='png', dpi=700, pasta_destino='images/'):

    grid = gridspec.GridSpec(n_rows, n_cols)

    if figure:
        figure
    else:
        plt.figure(figsize=figsize)

    for r in range(0, n_rows):
        for c in range(0, n_cols ):
            if (c + r*n_cols) >= len(columns_features):
                break
            else:
                feature = columns_features[ (c + r*n_cols) ]

                if sort_by:
                    data = data.sort_values(f'{sort_by}',ascending = False)
                else:
                    data = data.sort_values(f'{feature}',ascending = False)

                if loop_feature:
                    plot_kwargs[loop_feature] = feature
                    
                plt.subplot(grid[r, c])
                plt.title(f'{feature}')
                g = plot(data = data, **plot_kwargs)

                plt.xticks(rotation = 30)

                if label:
                    if plot.__name__ == 'lineplot':
                        if loop_feature == 'y':
                            y_col = feature
                            x_col = plot_kwargs['x']
                        else:
                            y_col = plot_kwargs['y']
                            x_col = feature

                        for i, row in data.iterrows():
                            g.annotate(str(np.round(row[y_col], 1)), (row[x_col], row[y_col]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=10)
                    else:
                        for i in g.containers:
                            g.bar_label(i, color = 'black',label_type='edge')
                else:
                    pass

    plt.tight_layout()

    if save_image:
        salvar_grafico(nome_imagem, pasta_destino, figura=plt.gcf(), formato=formato, dpi=dpi)
            
def ml_error(model_name, y, yhat):
    mae = mt.mean_absolute_error(y,yhat)
    mape = mt.mean_absolute_percentage_error(y,yhat)
    rmse = np.sqrt(mt.mean_squared_error(y,yhat))
    
    return pd.DataFrame({'Model Name': model_name,
                         'MAE': mae,
                         'MAPE':mape,
                         'RMSE':rmse},index=[0])

def ml_error_cv(model_name, y, yhat, k):
    mae = mt.mean_absolute_error(y,yhat)
    mape = mt.mean_absolute_percentage_error(y,yhat)
    rmse = np.sqrt(mt.mean_squared_error(y,yhat))
    
    return pd.DataFrame({'Model Name': model_name,
                         'K Fold': k,
                         'MAE': mae,
                         'MAPE':mape,
                         'RMSE':rmse},index=[0])


## Data Preparation

def save_picked_file(file, name):
    return pickle.dump(file,open(f'params/preparation/{name}.pkl','wb'))

def load_picked_file(name):
    return pickle.load(open(f'params/preparation/{name}.pkl','rb'))

def save_parquet_file(file, name):
    return file.to_parquet(f'params/data/{name}.parquet')

def load_parquet_file(name):
    return pd.read_parquet(f'params/preparation/{name}.parquet')

def robust_scaler(df, column, train=True):

    data = df.copy()

    if train:
        scaler = RobustScaler()
        data[column] = scaler.fit_transform(data[[column]])

        save_picked_file(scaler, f'robust_scaler_{column}')

    else:

        scaler = load_picked_file(f'robust_scaler_{column}')
        data[column] = scaler.transform(data[[column]])

    return data

def standart_scaler(df, column, train=True):

    data = df.copy()

    if train:
        scaler = StandardScaler()
        data[column] = scaler.fit_transform(data[[column]])

        save_picked_file(scaler, f'standart_scaler{column}')

    else:

        scaler = load_picked_file(f'standart_scaler{column}')
        data[column] = scaler.transform(data[[column]])

    return data

def target_encode(df, column, train = True):
    
    data = df.copy()

    if train:
        media_por_categoria = data[[column,'valor_total']].groupby(column).agg(total=('valor_total','mean')).reset_index()
        media_por_categoria['total'] = round(media_por_categoria['total'],2)
        media_por_categoria = media_por_categoria.set_index(column)['total'].to_dict()

        data[column] = data[column].map(media_por_categoria)
        save_picked_file(media_por_categoria, f'target_encode_{column}')

    else:
        media_por_categoria = load_picked_file(f'target_encode_{column}')
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
            max_value = df[column].max()
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

def run_clustering(data):
    features = ['area', 'quartos', 'suites','banheiros', 'vagas_garagem' ]

    dict_preparation = {
    'area': 'standart_scaler',
    'quartos': 'standart_scaler',
    'suites': 'standart_scaler',
    'banheiros': 'standart_scaler',
    'vagas_garagem': 'standart_scaler',
    'mobiliado': 'dummie_encode',
    'piscina': 'dummie_encode',
    'academia': 'dummie_encode',
    'sacada': 'dummie_encode',
    'churrasqueira': 'dummie_encode',
    'salao_de_festas': 'dummie_encode'
    }

    df_transformed = preparacao_dos_dados(df=data, dict_preparation=dict_preparation, is_train=True)

    kmeans = KMeans(n_clusters=8 , random_state=42)
    kmeans.fit(df_transformed[features])

    labels = kmeans.labels_

    return labels

## Machine Learning

def prepare_fit_and_predict(model, data_train, data_validation, dict_preparation, features_selected, log_on = False):

    data_train_transf = preparacao_dos_dados(data_train, dict_preparation, is_train=True)
    data_validation_transf = preparacao_dos_dados(data_validation, dict_preparation, is_train=False)

    X_train = data_train_transf[features_selected]
    y_train = data_train_transf['valor_total']

    X_validation = data_validation_transf[features_selected]
    y_validation = data_validation_transf['valor_total']

    # fit model
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_validation)

    if log_on:
        return np.expm1(y_validation), np.expm1(y_pred), model
    else:
        return y_validation, y_pred, model

def cross_validation(data, dict_models, k_fold, len_train, len_validation, metodo_split, dict_preparation, features_selected, verbose=False):
    
    df_results = pd.DataFrame()
    data = data.sort_values('data_coleta')

    for model_name, model in dict_models.items():

        if verbose:
            print(f'\nModel: {model_name}')

        for k in range(1, k_fold+1):

            data_train_cv, data_validation_cv = split_dataset(data, k_fold, k, len_train, len_validation, metodo_split, verbose)

            y_validation_cv, y_pred, model  = prepare_fit_and_predict(model, data_train_cv, data_validation_cv, dict_preparation, features_selected)

            # get results
            df_results_model = ml_error_cv(model_name, y_validation_cv, y_pred, k)
            # df_results_model = ml_error_cv(model_name, y_validation_cv, y_pred, k)
            
            if verbose:
                MAE = round(df_results_model['MAE'].tolist()[0], 2)
                MAPE = round(df_results_model['MAPE'].tolist()[0], 2)
                RMSE = round(df_results_model['RMSE'].tolist()[0], 2)
                print(f'MAE: {MAE}, MAPE: {MAPE}, RMSE: {RMSE}')
                print('-----------')

            # save results
            df_results = pd.concat([df_results, df_results_model], axis=0)

    # df_results = df_results.groupby('Model Name').agg( MAE = ('MAE','mean'), MAE_std = ('MAE','std'), MAPE = ('MAPE','mean'), MAPE_std = ('MAPE','std'), RMSE = ('RMSE','mean'), RMSE_std = ('RMSE','std'))

    return df_results

def split_dataset(data, k_fold, k, len_train, len_validation, metodo_split, verbose = False):
    '''
    len VALIDATION = 3

    |---------------------------- TRAIN -------------------------|
    |------------------ TRAIN ----------------|----VALIDATION----|
    | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | ---------------- |
    | ---- TRAIN -----|--- VALIDATION --|
    | ------- TRAIN --------|--- VALIDATION --|
    | ---------- TRAIN -----------|--- VALIDATION ---|
    | ------------- TRAIN ---------------|--- VALIDATION ---|
    | ---------------- TRAIN -----------------|--- VALIDATION ---|

    '''

    # len_fold = int((data.shape[0] - 2*len_validation)/(k_fold - 1))

    # cálculo sem tamanho do treino
    
    # ## train start end
    # train_start = 0
    # train_end = len_validation + (k-1)*len_fold

    # ## validation start/end
    # validation_start = len_validation + (k-1)*len_fold
    # validation_end = 2*len_validation + (k-1)*len_fold

    # cálculo com tamanho do treino

    len_data = data.shape[0]
    if k_fold == 1: 
        len_fold = 0 
    else:
        len_fold = int(np.floor((len_data - (len_train + len_validation)) / (k_fold - 1)))
    
    match metodo_split:
        case 'tamanho_treino_fixo':    
            train_start      = (k-1)*len_fold
            train_end        = (k-1)*len_fold + len_train 
            validation_start = (k-1)*len_fold + len_train
            validation_end   = (k-1)*len_fold + len_train + len_validation
        case 'tamanho_treino_cheio':
            train_start      = 0
            train_end        = (k-1)*len_fold + len_train 
            validation_start = (k-1)*len_fold + len_train
            validation_end   = (k-1)*len_fold + len_train + len_validation
        case 'dados_validacao_fixa':
            train_start      = len_data - len_validation - len_train - (k-1)*len_fold
            train_end        = len_data - len_validation
            validation_start = len_data - len_validation
            validation_end   = len_data

    if verbose:
        
        print(f'K Fold {k} --------')
        print(f'Start Train     : {train_start}')
        print(f'End Train       : {train_end}')
        print(f'Start Validation: {validation_start}')
        print(f'End Validation  : {validation_end}')

    ## select train and validation fold
    data_validation_cv = data.iloc[validation_start : validation_end]
    data_train_cv = data.iloc[train_start:train_end]

    return data_train_cv, data_validation_cv