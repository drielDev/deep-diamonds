import tensorflow as tf

# build_model: constrói e compila a rede neural usada para regressão do preço.
# - input_dim: número de features de entrada (colunas do X).
# Arquitetura simples e adequada para portfólio:
# - Inicializador HeNormal para camadas com ReLU (boa prática para redes com ReLU).
# - Camadas Dense com BatchNormalization + Dropout para estabilizar treino e reduzir overfitting.
# - Saída única sem ativação (regressão).
def build_model(input_dim):
    
    # Inicializador recomendado para camadas com ReLU
    he = tf.keras.initializers.HeNormal()
    
    model = tf.keras.Sequential([
        # primeira camada densa; input_shape define a forma de entrada
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer=he, input_shape=[input_dim,]),
        # normaliza as ativações do batch para acelerar a convergência
        tf.keras.layers.BatchNormalization(),
        # desliga 20% dos neurônios durante o treino para reduzir overfitting
        tf.keras.layers.Dropout(0.2),
        
        # camada escondida intermediária
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=he),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # camada escondida menor (funil)
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer=he),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # camada de saída para regressão (valor contínuo)
        tf.keras.layers.Dense(1)
    ])
    
    # Compilação: Adam + MSE como loss e MAE como métrica interpretável
    model.compile(
        optimizer='adam',
        loss='mse',     # loss usada para otimizar (penaliza grandes erros)
        metrics=['mae'] # métrica mais intuitiva: erro médio absoluto
    )
    
    return model