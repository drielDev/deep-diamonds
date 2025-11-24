import tensorflow as tf

def build_model(input_dim):
    
    he = tf.keras.initializers.HeNormal()
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=he, input_shape=[input_dim,]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=he),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer=he),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model