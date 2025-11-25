import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Carrega o CSV e retorna um DataFrame pandas.
def load_data(path='./data/diamonds.csv'):
    """Lê o arquivo CSV e retorna um DataFrame."""
    return pd.read_csv(path)

def preprocess(df):
    """
    Pré-processamento básico:
    - Remove linhas com dimensões inválidas (x,y,z <= 0).
    - Converte colunas categóricas para códigos inteiros (map).
    - Cria uma feature 'volume' a partir de x*y*z.
    - Remove colunas x,y,z (usamos 'volume' no lugar).
    - Separa X (features) e y (target price).
    Retorna: X (DataFrame) e y (Series).
    """
    # remover registros com medidas físicas inválidas
    df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]

    # mapear categorias para inteiros (mantém ordem/semântica definida aqui)
    df['cut'] = df['cut'].map({'Fair': 0, 'Ideal': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4})
    df['color'] = df['color'].map({'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6 })
    df['clarity'] = df['clarity'].map({
        'SI2': 0, 'SI1': 1, 'VS1': 2, 'VS2': 3,
        'VVS2': 4, 'VVS1': 5, 'I1': 6, 'IF': 7,
    })

    # nova feature numérica útil para o preço
    df['volume'] = df['x'] * df['y'] * df['z']

    # remover dimensões originais
    df = df.drop(columns=['x', 'y', 'z'])
    
    # separar em features (X) e target (y)
    X, y = df.drop('price', axis=1), df['price']
    
    return X, y

def scale_features(X_train, X_test):
    """
    Escalonamento Min-Max:
    - Ajusta o scaler no conjunto de treino e transforma treino/teste.
    - Retorna arrays numpy escalados (prontos para o modelo).
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

def load_and_prepare(path='./data/diamonds.csv'):
    """
    Pipeline de carregamento:
    - Lê dados, pré-processa, divide treino/teste e escala features.
    - Retorna X_train, X_test, y_train, y_test prontos para treino.
    """
    df = load_data(path)
    X, y = preprocess(df)
    
    # divisão treino/teste; shuffle para embaralhar os exemplos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # escalonar features numéricas
    X_train, X_test = scale_features(X_train, X_test)
    
    return X_train, X_test, y_train, y_test