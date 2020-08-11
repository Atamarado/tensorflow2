import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt

# 0 means women, and 1 means men

IMG_SIZE = 100

# Load data
x_train = np.load('trainData.npy')
y_train = np.load('trainLabels.npy')
x_test = np.load('valData.npy')
y_test = np.load('valLabels.npy')

# Shuffle data
random = np.arange(len(x_train))
np.random.shuffle(random)
x_train = x_train[random]
y_train = y_train[random]

random = np.arange(len(x_test))
np.random.shuffle(random)
x_test = x_test[random]
y_test = y_test[random]

# Layers
model = models.Sequential()

model.add(layers.Conv2D(IMG_SIZE, (3, 3), activation='relu', padding='same', input_shape=(100, 100, 1)))
model.add(layers.Conv2D(IMG_SIZE*2, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(IMG_SIZE*2, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(IMG_SIZE*2, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(IMG_SIZE*2, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(IMG_SIZE*2, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# Now, we add layers to flat and compress info until we arrive to 1 neuron
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))

model.summary()

# Train
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=4,
                    validation_data=(x_test, y_test))

# Model save
model.save('model.h5')

