import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from data import load_and_prepare
from model import build_model

def main():
    
    print("Carregando e preparando os dados...")
    X_train, X_test, y_train, y_test = load_and_prepare()
    
    print("Construindo o modelo...")
    model = build_model(input_dim=X_train.shape[1])
    
    print("Iniciando treinamento...")
    history = model.fit(X_train, y_train, batch_size=100, epochs=50, validation_split=0.2, shuffle=True)
    
    print("Avaliando no conjunto de teste...")
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nResultados finais:")
    print(f" - MSE (loss): {loss:.4f}")
    print(f" - MAE: {mae:.4f}")
    print(f"\nEm média, o modelo erra o preço do diamante por aproximadamente ${mae:.2f} dólares.")
    
    print("\nExemplo de previsão em um dado real:")

    # Escolher um índice aleatório
    idx = np.random.randint(0, len(X_test))

    X_sample = np.array([X_test[idx]])
    y_true = y_test.iloc[idx]

    y_pred = model.predict(X_sample, verbose=0)[0][0]

    print(f"Preço real:     ${y_true:.2f}")
    print(f"Preço previsto: ${y_pred:.2f}")
    print(f"Erro:           ${abs(y_true - y_pred):.2f}")

    # Criar pasta models caso não exista
    os.makedirs("models", exist_ok=True)

    print("\nSalvando o modelo em 'models/diamond_model.keras'...")
    model.save("models/diamond_model.keras")

    print("\nTreinamento concluído com sucesso!")


if __name__ == "__main__":
    main()