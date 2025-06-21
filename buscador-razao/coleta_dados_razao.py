import requests
import bs4
import numpy as np
import pandas as pd
from unidecode import unidecode
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from google.cloud import storage


def coalesce(value):
    try:
        value
    except:
        return np.nan
    return value

def try_get_value(dict_, key):
    try:
        return dict_[key]
    except:
        return np.nan

def scroll_page_down(driver,wait):

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(wait)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def wait_load_button(driver):
    print('Esperando o botão carregar...')
    wait = WebDriverWait(driver, 50)
    wait.until(EC.presence_of_element_located((By.XPATH,'//*[@id="resultados"]/section/div/div[2]/div/ul/li[1]/a')))  

    return None

def get_number_page(soup):

    print('Coletando o número da página atual')
    n_pagina_atual = int(soup.find_all('span',{'class':'btn-padr active'})[0].getText())
    print(f'Pagina Atual: {n_pagina_atual}')

    return n_pagina_atual

def pass_next_page(driver, pagina_atual, n_pagina_atual, n_ultima_pagina):

    print('Selecionando e clicando na próxima página\n')
    if pagina_atual <= 3:
        page_pass = driver.find_element(By.XPATH,f'//*[@id="resultados"]/section/div/div[2]/div/ul/li[{2+pagina_atual}]/a')
        page_pass.click()
    elif n_pagina_atual > n_ultima_pagina-2:
        page_pass = driver.find_element(By.XPATH,f'//*[@id="resultados"]/section/div/div[2]/div/ul/li[{int(6-(n_ultima_pagina - n_pagina_atual))}]/a')
        page_pass.click()
    else:
        page_pass = driver.find_element(By.XPATH,'//*[@id="resultados"]/section/div/div[2]/div/ul/li[5]/a')
        page_pass.click()
    
    return driver

def get_last_page(LINK,chrome_options):

    print('Abrindo site')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(LINK)
    driver.maximize_window()

    print('Indo para o fim da página primeira página')
    scroll_page_down(driver,wait=0.5)

    print('Esperando o botão ser carregado')
    wait = WebDriverWait(driver, 50)
    wait.until(EC.presence_of_element_located((By.XPATH,'//*[@id="resultados"]/section/div/div[2]/div/ul/li[5]/a')))

    time.sleep(3)

    print('Encontrando o botão da última página e clicando')
    ultima_pagina = driver.find_element(By.XPATH,'//*[@id="resultados"]/section/div/div[2]/div/ul/li[5]/a')
    ultima_pagina.click()

    print("Esperando o botão ser carregado")
    wait = WebDriverWait(driver, 50)
    wait.until(EC.presence_of_element_located((By.XPATH,'//*[@id="resultados"]/section/div/div[2]/div/ul/li[1]/a')))  

    print('Coletando o número total de páginas')
    soup = bs4.BeautifulSoup(driver.page_source, 'html.parser')
    n_ultima_pagina = int(soup.find_all('span',{'class':'btn-padr active'})[0].getText())

    driver.quit()

    return n_ultima_pagina

def get_link_anuncios(LINK, n_ultima_pagina,chrome_options):

    lista_de_links = []

    print('Abrindo a primeira página novamente')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(LINK)
    driver.maximize_window()

    for pagina_atual in range(1, n_ultima_pagina):

        scroll_page_down(driver,wait=0.5)

        wait_load_button(driver)

        soup = bs4.BeautifulSoup(driver.page_source, 'html.parser')

        n_pagina_atual = get_number_page(soup)

        print('Coletando links dos anúncios')
        lista_de_links = lista_de_links + [ anuncio['onclick'].split("'")[1] for anuncio in soup.find_all('button',{'class':'btn btn-padr btn-detalhes detalhes'})]

        wait_load_button(driver)

        driver = pass_next_page(driver, pagina_atual, n_pagina_atual, n_ultima_pagina)
    
    return lista_de_links

def get_info_anuncios(lista_de_links):

    anuncios_list = []
    base_url = 'https://imobiliariarazao.com.br/'

    for anuncio in lista_de_links:

        link = base_url + anuncio
        print(link)
        res = requests.get(link)
        soup = bs4.BeautifulSoup(res.content,'html.parser')

        anuncios_aux = {}

        # site
        anuncios_aux['site'] = 'razao'

        # link
        anuncios_aux['data_coleta'] = datetime.today().strftime("%Y-%m-%d")

        # link
        anuncios_aux['link'] = link

        # titulo
        anuncios_aux['titulo'] = unidecode(coalesce(soup.find_all('h1',{'class':'titleFicha'})[0].getText()))

        # endereco
        anuncios_aux['endereco'] = unidecode(coalesce(soup.find_all('div',{'class':'enderecoImovel enderecoFicha'})[0].find('p').getText().strip()))

        # condominio
        anuncios_aux['condominio'] = coalesce(soup.find_all('p',{'class':'valorCond'})[0].getText())

        # iptu
        anuncios_aux['iptu'] = coalesce(soup.find_all('p',{'class':'valorIptu'})[0].getText())

        # aluguel
        anuncios_aux['aluguel'] = coalesce(soup.find_all('p',{'class':'valorPrincipalImovel'})[0].getText())

        # itens_do_imovel
        anuncios_aux['itens_imovel'] = unidecode(coalesce(', '.join([i.getText() for i in soup.find_all('div',{'class':'col-sm-6 itensDescricao'})])))

        # descrição
        anuncios_aux['descricao'] = unidecode(coalesce(soup.find_all('div',{'class','observaFicha'})[0].find('p').getText()))

        # atributos
        try:
            anuncios_aux['atributos'] = [item.find('p',{'class':'quantItem'}).getText() for item in soup.find_all('div',{'class':'itensImovel'})]
        except:
            anuncios_aux['atributos'] = np.nan

        anuncios_list.append(anuncios_aux)

    return pd.DataFrame(anuncios_list)

def save_on_bucket(BUCKET_NAME, imobiliaria, data):
    FILE_NAME = f'{datetime.today().strftime("%Y-%m-%d")} - apartamentos - {imobiliaria}.csv'
    TEMP_FILE = 'local.csv'  
    RAW_PATH = '/tmp/{}'.format(TEMP_FILE)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)

    data.to_csv(RAW_PATH, index=False)
    blob = bucket.blob(FILE_NAME)
    blob.upload_from_filename(RAW_PATH)


if __name__ == '__main__':
    LINK = 'https://imobiliariarazao.com.br/busca.php?termoPesquisa=Curitiba&codCity=3314&tipoNegocio=2&isLancamento=0&tipo_Imovel%5B%5D=3&referencia=&endereco='

    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-dev-shm-usage')

    n_ultima_pagina = get_last_page(LINK,chrome_options)

    lista_de_links = get_link_anuncios(LINK, n_ultima_pagina,chrome_options)

    print("Coletando dados os anuncios individualmente:")
    df_anuncios = get_info_anuncios(lista_de_links)

    print('Salvando dados no Bucket')
    save_on_bucket(BUCKET_NAME= 'busca-apartamentos-bucket', imobiliaria = 'razao', data=df_anuncios)
