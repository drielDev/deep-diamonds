import tensorflow as tf

def build_model(input_dim):
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim)),
        tf.keras.layer.Dense(32, activation='relu'),
        tf.keras.layer.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model