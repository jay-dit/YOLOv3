import tensorflow as tf
from source import model as yl
from sklearn.model_selection import train_test_split
import numpy as np

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

model = yl.build_model(128, 128, 1, 1)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

x = np.load('data/X.npy')
y = np.load('data/Y.npy')

X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8, 
                                                    random_state=42)

model.fit(x = X_train,
          y = y_train,
          epochs = 10,
          validation_split=0.05)

model.save('my_model.h5')