# github.com/Atamarado
# In this program we get data generated with 1_generate_data.py and create a neural network
# model and export it

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

IMG_SIZE = 50
BATCH = 100
TRAIN_FILES = 131
EPOCHS = 3

# Model definition
model = models.Sequential()
model.add(layers.Conv2D(3, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3), padding='same'))
model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D((2,2), padding='same'))
model.add(layers.Conv2D(8, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
# model.add(layers.BatchNormalization())

# model.add(layers.Conv2DTranspose(256, (3,3), activation='relu', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2DTranspose(128, (3,3), activation='relu', padding='same'))
# model.add(layers.BatchNormalization())
model.add(layers.Conv2DTranspose(32, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2DTranspose(8, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
#model.add(layers.UpSampling2D((2,2)))
model.add(layers.Conv2DTranspose(3, (3,3), activation='sigmoid', padding='same'))
#model.add(layers.BatchNormalization())


model.summary()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1), loss=tf.keras.losses.MeanSquaredError())

order = np.array([i for i in range(TRAIN_FILES)])

# We'll fit in our DnCNN every image from every train file epoch times
# For every epoch
for a in range(EPOCHS):
    # Randomize order to improve performance
    np.random.shuffle(order)
    for b in range(TRAIN_FILES):
        print("Progress: " + str(a/EPOCHS+b/TRAIN_FILES/EPOCHS))

        # read_files
        x_images = np.load(os.path.join('data', 'x_train'+str(order[b])+'.npy'))
        y_images = np.load(os.path.join('data', 'y_train'+str(order[b])+'.npy'))

        # Randomize every batch
        random = np.arange(BATCH)
        np.random.shuffle(random)
        x_images = x_images[random]
        y_images = y_images[random]

        # Train batch
        model.fit(x=x_images, y=y_images, epochs=1, verbose=2)

# Save model
model.save('model.h5')


