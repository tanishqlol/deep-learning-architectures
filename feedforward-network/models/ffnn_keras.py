import keras
from keras import layers, callbacks

model_simple = keras.Sequential([

    # hidden ReLU layer
    layers.Dense(units=128, activation='relu', input_shape=(784,)),
    # output layer with 10 outputs with sigmoid activation
    layers.Dense(units=10, activation='sigmoid')
])

model_wider = keras.Sequential([

    # hidden ReLU layer
    layers.Dense(units=256, activation='relu', input_shape=(784,)),
    layers.Dropout(rate=0.3),
    # output layer with 10 outputs with sigmoid activation
    layers.Dense(units=10, activation='sigmoid')
])

model_deeper = keras.Sequential([

    # hidden ReLU layers
    layers.Dense(units=40, activation='relu', input_shape=(784,)),
    layers.Dropout(rate=0.3),
    layers.Dense(units=20, activation='relu'),
    layers.Dropout(rate=0.2),
    # output layer with 10 outputs with softmax activation
    layers.Dense(units=10, activation='softmax')
])

model_high_capacity = keras.Sequential([

    # hidden ReLU layers
    layers.Dense(units=256, activation='relu', input_shape=(784,)),
    layers.Dropout(rate=0.4),
    layers.Dense(units=128, activation='relu'),
    layers.Dropout(rate=0.2),
    layers.Dense(units=128, activation='relu'),
    layers.Dropout(rate=0.2),
    # output layer with 10 outputs with softmax activation
    layers.Dense(units=10, activation='softmax')
])

