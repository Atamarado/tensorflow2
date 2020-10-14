# We generate 4 files with this code, prepared to be fit in the neural network
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
import random

IMG_SIZE = 150
BATCH = 100

def prepare_data(inputFile, outputFile_y, number):
    # Read fileNames and y_labels
    fileNames = np.loadtxt(inputFile, delimiter=' ', dtype=np.str)

    # Shuffle data
    np.random.shuffle(fileNames)

    # nImages should be multiple of BATCH
    nImages = int(np.shape(fileNames)[0])

    # Path of the images
    path = Path().absolute()
    # Go up a level
    path = path.parent
    pathFolder = path.as_posix() + '/datasets/faces/'

    # For everyImage:
    for a in range(int(nImages/BATCH)):
        # Separate groups of images
        y_file = np.empty((BATCH, int(IMG_SIZE), int(IMG_SIZE), 3))
        for b in range(BATCH):
            # Image reading
            impath = pathFolder + fileNames[a*BATCH+b][0]
            originalimg = Image.open(impath)

            # Reducing image size to IMG_SIZE * IMG_SIZE
            reducedimg = originalimg.resize((IMG_SIZE, IMG_SIZE))

            # Convert to numpy array
            npimg = np.array(reducedimg)

            # Normalize values
            y_file[b] = npimg/255

        y_filename = outputFile_y + str(number) + '.npy'
        number = number + 1

        # Save data
        np.save(os.path.join('data', y_filename), y_file)

# Number of the file generated (if more than one file is generated)
numberTrain = 0
numberTest = 0

LANDSCAPE_TRAIN = 24200
LANDSCAPE_TEST = 24300

trainName = "y_train"
testName = "y_test"

# Read faces dataset
prepare_data("train.txt", trainName, numberTrain)
prepare_data("val.txt", testName, numberTest)

# ------------------------------------------------------------------
# Read landscape dataset

# Read path
path = Path().absolute()
# Go up a level
path = path.parent
pathFolder = path.as_posix() + '/datasets/landscapes/'

landscapeNumber = 1

# Prepare read files
y_file = np.empty((BATCH, int(IMG_SIZE), int(IMG_SIZE), 3))

while landscapeNumber <= LANDSCAPE_TRAIN:
    # For every batch image
    for a in range(BATCH):
        # Open image
        impath = pathFolder + str(landscapeNumber) + '.jpg'
        landscapeNumber = landscapeNumber + 1
        image = Image.open(impath)

        # Resizing
        image = image.resize((IMG_SIZE, IMG_SIZE))
        # Change to numpy arrays
        image = np.array(image)
        y_file[a] = image/255

    # Save image batch
    y_filename = trainName + str(numberTrain) + '.npy'
    numberTrain = numberTrain + 1

    # Save data
    np.save(os.path.join('data', y_filename), y_file)

# Test images
for a in range(BATCH):
    impath = pathFolder + str(landscapeNumber) + '.jpg'
    landscapeNumber = landscapeNumber + 1
    image = Image.open(impath)

    # Change to numpy arrays
    image = np.array(image)
    y_file[a] = image / 255

# Save image batch
y_filename = testName + str(numberTest) + '.npy'
numberTrain = numberTest + 1

# Save data
np.save(os.path.join('data', y_filename), y_file)
