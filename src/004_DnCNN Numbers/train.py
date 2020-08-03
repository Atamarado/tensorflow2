# In this program we get data generated with generate_data.py and create a neural network
# model and export it

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 28

# Load datasets
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

print(x_train.shape)

# Model definition
model = models.Sequential()
model.add(layers.Conv2D(1, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(layers.Conv2D(2, (3,3), activation='relu'))
model.add(layers.Conv2DTranspose(2, (3,3), activation='relu'))
model.add(layers.Conv2DTranspose(1, (3,3), activation='relu'))

#model.summary()

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1), loss=tf.keras.losses.MeanSquaredError())

# Train model
model.fit(x=x_train, y=y_train, epochs = 5,
                    validation_data=(x_test, y_test))

# Save model
model.save('model.h5')


