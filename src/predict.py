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

    df = pd.read_csv('./data/diamonds.csv')
    X, _ = preprocess(df)
    
    scaler = MinMaxScaler()
    scaler.fit(X)
    
    return scaler 

def prepare_input(feature_dict):
    df = pd.DataFrame([feature_dict])
    df = df.astype(float)

    return df

def main():
    
    model = load_model('./models/diamond_model.keras')
    scaler = load_scaler()
    
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

    df_input = prepare_input(example)
    X_scaled = scaler.transform(df_input)

    pred = model.predict(X_scaled, verbose=0)[0][0]

    print(f"\nPreço previsto: ${pred:.2f} dólares")


if __name__ == "__main__":
    main()