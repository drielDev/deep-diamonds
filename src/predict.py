import os

# reduzir verbosidade do TensorFlow antes de importar o pacote
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0=all,1=info,2=warning,3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # desativa mensagens oneDNN

import warnings
warnings.filterwarnings('ignore')  # silencia warnings não críticos para saída limpa

import pandas as pd
from keras.models import load_model
from data import preprocess
from sklearn.preprocessing import MinMaxScaler


def load_scaler():
    """
    Carrega os dados originais e ajusta um MinMaxScaler usando as mesmas features
    usadas no treino. Retorna o scaler ajustado.
    Observação: o DataFrame de entrada para transform deve ter as mesmas colunas e ordem.
    """
    
    df = pd.read_csv('./data/diamonds.csv')
    X, _ = preprocess(df)            # aplica o mesmo pré-processamento do pipeline de treino

    scaler = MinMaxScaler()
    scaler.fit(X)                    # aprende min/max de cada feature
    return scaler

def prepare_input(feature_dict):
    """
    Converte um dicionário de features (uma amostra) em DataFrame pronto para transformar.
    - Espera que as chaves em feature_dict correspondam às colunas usadas no treino.
    - Se necessário, reindexar para garantir ordem/colunas antes de chamar scaler.transform.
    """
    
    df = pd.DataFrame([feature_dict])
    # garantir tipo numérico; scaler do sklearn espera floats
    df = df.astype(float)
    return df

def main():
    """
    Pipeline principal:
    1) carrega modelo salvo
    2) carrega e ajusta scaler baseado no dataset original
    3) prepara um exemplo manual, aplica scaler e prevê
    4) imprime resultado
    """
    
    # carregar o modelo (formato .keras salvo pelo pipeline de treino)
    model = load_model('./models/diamond_model.keras')

    # carregar e ajustar o scaler (baseado nas mesmas features do treino)
    scaler = load_scaler()

    # exemplo de entrada — as chaves devem corresponder às colunas do X usado no treino
    example = {
        "carat": 0.9,
        "cut": 3,        # Very Good
        "color": 2,      # F
        "clarity": 3,    # VS2
        "depth": 62.0,
        "table": 57.0,
        "volume": 55.0
    }

    print("\nPrevendo com os seguintes dados:")
    print(example)

    # preparar DataFrame e garantir ordem/colunas
    df_input = prepare_input(example)

    # aplicar o mesmo escalonamento usado no treino
    X_scaled = scaler.transform(df_input)

    # prever (verbose=0 para saída limpa)
    pred = model.predict(X_scaled, verbose=0)[0][0]

    print(f"\nPreço previsto: ${pred:.2f} dólares")

if __name__ == "__main__":
    main()