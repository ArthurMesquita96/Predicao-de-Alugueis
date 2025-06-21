# 🏠 Modelo de Precificação de Aluguéis com Machine Learning

Este projeto tem como objetivo desenvolver um **modelo preditivo para estimar o valor de aluguéis** residenciais na cidade de Curitiba, utilizando dados públicos coletados de sites de imobiliárias. 

A proposta é substituir métodos tradicionais de precificação de aluguais, mais caros e subjetivos, por uma abordagem baseada em dados e aprendizado de máquina, promovendo maior precisão, escalabilidade e transparência no processo de precificação.

## Objetivos

- Criar um modelo de Machine Learning capaz de estimar o valor total de locação (aluguel + condomínio + iptu) de imóveis residenciais.
- Reduzir o viés humano na precificação
- Automatizar o processo e permitir precificações dinâmicos com base na realidade do mercado.
- Fornecer maior explicabilidade sobre os fatores que mais influenciam o preço de locação.

## Metodologia
O projeto segue as principais etapas de um pipeline de Data Science:

- **Coleta de Dados**: Web scraping em sites de imobiliárias para reunir anúncios de imóveis.

- **Limpeza e Tratamento**: Remoção de valores nulos, padronização de categorias e exclusão de outliers.

- **Feature Engineering**: Criação e transformação de variáveis descritivas dos imóveis, condomínios e regiões.

- **Análise Exploratória**: Visualização e entendimento do comportamento das variáveis.

- **Seleção de Features**: Escolha das variáveis mais relevantes para o modelo.

- **Modelagem**: Testes com diferentes algoritmos de regressão e avaliação com métricas como MAE, MAPE e RMSE.

- **Avaliação e Tunagem**: Cross-validation, tuning de hiperparâmetros e comparação com baseline.

- **Validação Final**: Avaliação da performance do modelo com dados nunca vistos.

## Sobre os Dados

Os dados foram coletados via web scraping (seleium) em três grandes portais de imóveis. 
As variáveis utilizadas incluem:

- **Características do imóvel**: área, número de quartos, banheiros, suítes, vagas de garagem, presença de mobília, etc.

- **Características do condomínio**: presença de piscina, academia, salão de festas, entre outros.

- **Características geográficas e socioeconômicas**: bairro, região, renda média e quantidade de domicílios por bairro.

## Resultados
O modelo final, baseado em Random Forest, obteve os seguintes resultados em comparação com um baseline simples:

| Modelo	      | MAE	    | MAPE |	RMSE  |
| --------------| --------| -----| ------- |
| Random Forest	| 253,68	| 10%	 | 411,37  |
| Baseline (m²)	| 713,34	| 26%	 | 964,66 |

Comparado ao modelo baseline (valor médio por m² por bairro), o modelo final apresenta uma redução significativa no erro absoluto médio, indicando maior capacidade preditiva com uso de ML.

## Limitações

- O valor analisado é o do anúncio, não necessariamente o valor negociado final.
- O tempo de vacância do imóvel (tempo em que o imóvel permanece disponível para locação) não foi considerado.
- A qualidade dos dados depende da precisão e consistência das informações publicadas nos anúncios 

🔍 Próximos Passos

- Configurar o retreino automático do modelo com novos imóveis
- Inclusão de novas features para o modelo visando a redução do erro médio
- Explorar a influencia de cada atributo dos imóveis sobre o valor final da locação

## Produto Final

Para fazer uma estimativa do valor final de locação com o modelo final, [acesse aqui](https://predicao-de-alugueis-5b442gzpt8xlwacs4n397w.streamlit.app/)

