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
# Tamaño de las capas
n_input_size = 784  # (28*28)
hidden_layer_1_size = 256
hidden_layer_2_size = 32
hidden_layer_3_size = 32
hidden_layer_4_size = 256
output_layer_size = 784  # El mismo que la entrada, ya que el resultado debe ser una imagen

pool_1_size = n_input_size/hidden_layer_1_size
pool_2_size = hidden_layer_1_size/hidden_layer_2_size
pool_3_size = 1/pool_2_size
pool_4_size = 1/pool_1_size

# Paramétros
learning_rate = 0.1
epochs = 30
batch_size = 100  # En cada iteración del algoritmo solo se cogerán 100 imágenes para analizar

########### Begin data
img_database = np.loadtxt('fashion-mnist_train.csv', delimiter=',', skiprows=1)

total_num_images = (img_database.shape[0])
# Aleatorizar el orden de las imágenes
np.random.shuffle(img_database)

imagenes_original = img_database

# Label is in the first element, not the last one, so remove first label (not useful here)
imagenes = np.delete(imagenes_original, 0, axis=1)

# Cogemos solo una parte de las imágenes (50000)
x_train = imagenes[:20]

for i in range(10):
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

# Creamos las capas
#p1 = tf.keras.layers.MaxPool2D(pool_size=(pool_1_size, pool_1_size), padding='valid', data_format=None)
z1 = tf.keras.layers.Dense(hidden_layer_1_size, activation=tf.nn.sigmoid)
#p2 = tf.keras.layers.MaxPool2D(pool_size=(pool_2_size, pool_2_size), padding='valid', data_format=None)
z2 = tf.keras.layers.Dense(hidden_layer_2_size, activation=tf.nn.sigmoid)
z3 = tf.keras.layers.Dense(hidden_layer_3_size, activation=tf.nn.sigmoid)
#p3 = tf.keras.layers.MaxPool2D(pool_size=(pool_3_size, pool_3_size), padding='valid', data_format=None)
z4 = tf.keras.layers.Dense(hidden_layer_4_size, activation=tf.nn.sigmoid)
#p4 = tf.keras.layers.MaxPool2D(pool_size=(pool_4_size, pool_4_size), padding='valid', data_format=None)
output = tf.keras.layers.Dense(output_layer_size)

# New: you only have created the layers, now you have to create the stack of them
# This is one way to do it, but the are many others

model_cnn = tf.keras.Sequential([z1, z2, z3, z4, output])

# Also, you need to compile it. This is one of the main steps you were missing!!

model_cnn.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate))


# This is tf.2.0. There are no sessions now
metrics_names = model_cnn.metrics_names

for epoch in range(epochs):
    # Reset the metric accumulators
    model_cnn.reset_metrics()
    # Maybe you should add a shuffle here, to avoid visiting images in same order

    x_train, x_train_noisy = sklearn.utils.shuffle(x_train, x_train_noisy, random_state=0)


    for i in range(int(total_num_images / batch_size)):
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



