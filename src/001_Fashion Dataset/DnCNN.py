# Code adapted from https://www.youtube.com/watch?v=LFqigdDW4Uk

import os
import numpy as np
import zipfile
from urllib import request
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow as tf
import cv2
import sklearn
#       CONVOLUCIÓN!!!!
# Tensorboard!!!

# Paramétros
learning_rate = 0.1
epochs = 30
batch_size = 100  # En cada iteración del algoritmo solo se cogerán 100 imágenes para analizar

########### Begin data
img_database = np.loadtxt('fashion-mnist_train.csv', delimiter=',', skiprows=1)

# Aleatorizar el orden de las imágenes
np.random.shuffle(img_database)

# Label is in the first element, not the last one, so remove first label (not useful here)
imagenes = img_database[0:200]
imagenes = np.delete(imagenes, 0, axis=1)

print(imagenes.shape)

imagenes = np.reshape(imagenes, (200, 28, 28))

# Cogemos solo una parte de las imágenes (50000)
x_train = imagenes[:20]


for i in range(5):
    x_train = np.concatenate((x_train, x_train))

# Añadimos un ruido normalizado a la imágenes
x_train_noisy = x_train + 10 * np.random.normal(0, 1, size=x_train.shape)

'''
# Mostramos una imagen con y sin ruido
plt.imshow(x_train[0].reshape(28, 28), cmap='gray')
plt.show()

plt.imshow(x_train_noisy[0].reshape(28, 28), cmap='gray')
plt.show()'''
########### End data

print(x_train[0].shape)

# Creamos las capas

# Define the model of DnCNN
model_cnn = tf.keras.Sequential()

#PROBAR COIN LA FUNCION DE ACTIVACION tf.nn.sigmoid
model_cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)))
#model_cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
model_cnn.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), activation=tf.nn.relu))

# Compilamos el modelo
model_cnn.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate))

# Mostramos los detalles de la red
model_cnn.summary()


# This is tf.2.0. There are no sessions now
metrics_names = model_cnn.metrics_names

for epoch in range(epochs):
    # Reset the metric accumulators
    model_cnn.reset_metrics()

    x_train, x_train_noisy = sklearn.utils.shuffle(x_train, x_train_noisy, random_state=0)


    for i in range(int(x_train.size/batch_size)):
        x_epoch = x_train[i * batch_size: (i + 1) * batch_size]
        x_noise_epoch = x_train_noisy[i * batch_size: (i + 1) * batch_size]
        result = model_cnn.train_on_batch(x_epoch, x_noise_epoch)
        print('step ' + str(i))
        print("train: ",
              "{}: {:.3f}".format(metrics_names[0], result))


# Test one image
x_actual = x_train[10:11, :]
noisy_image = x_train_noisy[10:11, :]

denoised_image = model_cnn.predict_on_batch(noisy_image) #.numpy()

plt.imshow(x_actual.reshape(28, 28), cmap='gray')
plt.show()

plt.imshow(noisy_image.reshape(28, 28), cmap='gray')
plt.show()

plt.imshow(denoised_image.numpy().reshape(28, 28), cmap='gray')
plt.show()



