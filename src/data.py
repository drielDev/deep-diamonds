import pandas as pd 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(path='./data/diamonds.csv'):
    return pd.read_csv(path)

def preprocess(df):
    df['cut'] = df['cut'].map({'Fair': 0, 'Ideal': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4})
    df['color'] = df['color'].map({'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6 })
    df['clarity'] = df['clarity'].map({'SI2': 0, 'SI1': 1, 'VS1': 2, 'VS2': 3, 'VVS2': 4, 'VVS1': 5, 'I1': 6, 'IF': 7,})

    X, y = df.drop('price', axis=1), df['price']
    
    return X, y

def scale_features(X_train, X_test):
   
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

def load_and_prepare(path='./data/diamonds.csv'):
    
    df = load_data(path)
    X, y = preprocess(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train, X_test = scale_features(X_train, X_test)
    
    return X_train, X_test, y_train, y_test