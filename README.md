# deep-diamonds

Projeto de exemplo para regressão de preço de diamantes usando redes neurais (TensorFlow / Keras).  
Objetivo: código simples, legível

## Estrutura
- data/diamonds.csv        — dataset (esperado)
- notebooks/exploracao.ipynb — análises exploratórias
- src/
  - data.py                — carregamento, limpeza e pré-processamento
  - model.py               — definição e compilação do modelo neural
  - train.py               — pipeline de treino (gera `models/diamond_model.keras`)
  - predict.py             — previsão pontual usando um modelo treinado
- models/                  — onde o modelo final é salvo
- requirements.txt         — dependências

## Requisitos
- Python 3.8+
- Instalar dependências:
  pip install -r requirements.txt

## Uso rápido

1. Criar ambiente (Windows):
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

2. Treinar:
   python src\train.py
   - Salva o modelo em `models/diamond_model.keras`.
   - Ao final, mostra métricas claras (MAE em dólar) e um exemplo: preço real vs preço previsto.

3. Fazer uma previsão
   python src\predict.py
   O script faz uma previsão com os valores de um exemplo fixo

4. Notebook (exploração):
   jupyter lab notebooks/exploracao.ipynb

## Notas importantes
- O pipeline espera o arquivo `data/diamonds.csv`. Ajuste o caminho se necessário.
- Pré-processamento (src/data.py):
  - Remove registros com dimensões inválidas.
  - Mapeia categorias (`cut`, `color`, `clarity`) para inteiros.
  - Cria `volume = x*y*z` e remove `x,y,z`.
  - Escala features com MinMaxScaler.
- Modelo salvo no formato Keras nativo (`.keras`).
- Para evitar logs verbosos do TensorFlow, os scripts setam:
  - `TF_CPP_MIN_LOG_LEVEL=3`
  - `TF_ENABLE_ONEDNN_OPTS=0`
  (esses são definidos nos próprios scripts antes de importar TF)

## Boas práticas usadas no projeto
- limpeza de outliers
- divisão treino/validação via validation_split
- early stopping com base no val_loss
- métricas: MAE e RMSE
- relatório final do treinamento com:
  - erro médio em dólares
  - um exemplo real comparado com previsão

## Licença
Código para fins educacionais. Use e adapte livremente.
