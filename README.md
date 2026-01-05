# Análise de Preço de Diamantes

Projeto de **Análise Exploratória de Dados (EDA)** com Python para compreender
os principais fatores que influenciam o preço de diamantes, utilizando dados reais.

O projeto inclui uma etapa de modelagem preditiva como complemento à análise.

## Objetivo
Explorar, analisar e extrair insights de um dataset de diamantes, avaliando
como características físicas e categóricas impactam o preço final.

## Tecnologias utilizadas
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- Scikit-learn
- TensorFlow / Keras (etapa complementar)

## Estrutura do projeto
- data/diamonds.csv — dataset utilizado
- notebooks/exploracao.ipynb — análise exploratória e visualização dos dados
- src/ — scripts de limpeza, pré-processamento e modelagem
- models/ — modelo treinado (etapa opcional)

## Análises realizadas
- Limpeza e tratamento de dados
- Análise estatística descritiva
- Distribuição de preços
- Relação entre preço, carat, corte, cor e clareza
- Identificação de outliers
- Visualizações para suporte à tomada de decisão

## Principais insights
- O preço apresenta forte correlação com o peso (carat)
- Características como corte e clareza influenciam significativamente o valor
- A presença de outliers impacta a média de preços
- O volume do diamante é um fator relevante para previsão de valor

## Modelagem (etapa complementar)
Foi implementado um modelo de regressão para estimar preços de diamantes,
utilizado como extensão da análise exploratória.

Métricas avaliadas:
- MAE
- RMSE

## Como executar
1. Criar ambiente virtual
2. Instalar dependências
3. Executar o notebook de exploração:
   ```bash
   jupyter lab notebooks/exploracao.ipynb
