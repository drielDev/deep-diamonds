import os

# reduzir verbosidade do TensorFlow antes de importar o pacote
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0=all,1=info,2=warning,3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # desativa mensagens oneDNN

import warnings
warnings.filterwarnings('ignore')  # silencia warnings não críticos para saída limpa

import numpy as np

# funções do projeto: carregar/pré-processar dados e construir o modelo
from data import load_and_prepare
from model import build_model

def main():
    """
    Pipeline principal:
    1) carregar e preparar dados
    2) construir modelo (com dimensões corretas de entrada)
    3) treinar, avaliar e salvar o modelo
    """

    print("Carregando e preparando os dados...")
    # retorna arrays prontos para treino/avaliação
    X_train, X_test, y_train, y_test = load_and_prepare()

    print("Construindo o modelo...")
    # informa ao build_model quantas features de entrada temos
    model = build_model(input_dim=X_train.shape[1])

    print("Iniciando treinamento...")
    # treino:
    # - batch_size: quantas amostras por atualização de gradiente
    # - epochs: quantas passagens sobre todo o conjunto de treino
    # - validation_split: fração do treino usada como validação durante o fit
    # - shuffle=True para embaralhar os dados a cada época
    history = model.fit(
        X_train,
        y_train,
        batch_size=100,
        epochs=50,
        validation_split=0.2,
        shuffle=True
    )

    print("Avaliando no conjunto de teste...")
    # avaliar sem imprimir barra de progresso (verbose=0)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    # Mostrar resultados resumidos
    print(f"\nResultados finais:")
    print(f" - MSE (loss): {loss:.4f}")
    print(f" - MAE: {mae:.4f}")
    print(f"\nEm média, o modelo erra o preço do diamante por aproximadamente ${mae:.2f} dólares.")

    print("\nExemplo de previsão em um dado real:")
    # escolher um exemplo aleatório do conjunto de teste
    idx = np.random.randint(0, len(X_test))

    # garantir que foi passado um batch 2D para model.predict (shape: (1, n_features))
    X_sample = np.array([X_test[idx]])
    # y_test é um pandas Series — usamos iloc para indexar pela posição
    y_true = y_test.iloc[idx]

    # prever sem output extra
    y_pred = model.predict(X_sample, verbose=0)[0][0]

    # imprimir valor real, previsto e erro absoluto formatados
    print(f"Preço real:     ${y_true:.2f}")
    print(f"Preço previsto: ${y_pred:.2f}")
    print(f"Erro:           ${abs(y_true - y_pred):.2f}")

    # garantir diretório para salvar o modelo
    os.makedirs("models", exist_ok=True)

    print("\nSalvando o modelo em 'models/diamond_model.keras'...")
    # salvar no formato nativo Keras (.keras)
    model.save("models/diamond_model.keras")

    print("\nTreinamento concluído com sucesso!")


if __name__ == "__main__":
    main()