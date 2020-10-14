# github.com/Atamarado
# In this program we get data generated with 1_generate_data.py and create a neural network
# model and export it

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import time
import matplotlib.pyplot as plt

IMG_SIZE = 150
BATCH = 100
TRAIN_FILES = 241
EPOCHS = 10

N_TIME_VALUES = 10

# Model definition
model = models.Sequential()
model.add(layers.Conv2D(3, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3), padding='same'))
model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D((2,2), padding='same'))
model.add(layers.Conv2D(3, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(3, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(3, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(3, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(3, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(3, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(3, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())


model.summary()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1), loss=tf.keras.losses.MeanSquaredError())

order = np.arange(TRAIN_FILES)

# We'll fit in our DnCNN every image from every train file epoch times
# For every epoch

TOTAL_FITS = EPOCHS*TRAIN_FILES
fitsDone = 0

# Execution time
startTime = time.time()

for a in range(EPOCHS):
    np.random.shuffle(order)
    for b in range(TRAIN_FILES):

        # read_files
        fileName = 'y_train'+str(order[b])+'.npy'
        print("Opening " + fileName)
        y_images = np.load(os.path.join('data', fileName))

        # Randomize every batch
        #-------------------------------------------------------------------------------
        np.random.shuffle(y_images)

        # Generate noisy images
        noise = 20 / 256 * np.random.normal(0, 1, size=y_images.shape)
        x_images = y_images + noise

        # Train batch
        model.fit(x=x_images, y=y_images, epochs=1, verbose=1)
        fitsDone = fitsDone + 1

        progress = fitsDone / TOTAL_FITS
        tiempo = time.time()
        timeRemaining = (tiempo-startTime)*(TOTAL_FITS-fitsDone)/fitsDone

        print("Progress: " + str(progress))
        print("Time remaining: " + str(timeRemaining/60) + " min.")
        #print("Time remaining: " + str(time.hour) + "h, " + str(time.minute) + "m.")
        print("")

# Save model
model.save('model.h5')

# Execution time
endTime = time.time()

print("Total time: " + str(endTime-startTime))


