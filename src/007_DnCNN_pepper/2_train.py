# github.com/Atamarado
# In this program we get data generated with 1_generate_data.py and create a neural network
# model and export it

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import time
import matplotlib.pyplot as plt
from skimage.util import random_noise

IMG_SIZE = 50
BATCH = 200
TRAIN_FILES = 65
EPOCHS = 3

# Model definition
model = models.Sequential()
model.add(layers.Conv2D(3, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3), padding='same'))
model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D((2,2), padding='same'))
model.add(layers.Conv2D(3, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())

# model.add(layers.Conv2DTranspose(8, (3,3), activation='relu', padding='same'))
# model.add(layers.BatchNormalization())
# #model.add(layers.UpSampling2D((2,2)))
# model.add(layers.Conv2DTranspose(3, (3,3), activation='sigmoid', padding='same'))
# #model.add(layers.BatchNormalization())


model.summary()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1), loss=tf.keras.losses.MeanSquaredError())

order = np.array([i for i in range(TRAIN_FILES)])

# We'll fit in our DnCNN every image from every train file epoch times
# For every epoch


# Randomize order to improve accuracy
np.random.shuffle(order)
tiempo = time.time()
for b in range(TRAIN_FILES):
    tiempo = tiempo*(int(TRAIN_FILES-b))

    print("Progress: " + str(b/TRAIN_FILES))
    print("Time remaining: " + str(tiempo/60) + " min.")
    #print("Time remaining: " + str(time.hour) + "h, " + str(time.minute) + "m.")

    start = time.time()
    # read_files
    y_images = np.load(os.path.join('data', 'y_train'+str(order[b])+'.npy'))

    # Randomize every batch
    random = np.arange(BATCH)
    np.random.shuffle(random)
    y_images = y_images[random]

    # Generate noisy images
    x_images = random_noise(y_images, mode='s&p', amount=0.15)
    x_images = (x_images + y_images)

    # Train batch
    model.fit(x=x_images, y=y_images, epochs=EPOCHS, verbose=2)

    tiempo = time.time() - start

# Save model
model.save('model.h5')


