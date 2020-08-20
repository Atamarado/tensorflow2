# We generate 4 files with this code, prepared to be fit in the neural network
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
import random

IMGSIZE = 50
BATCH = 100

def prepare_data(inputFile, outputFile_x, outputFile_y):
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
        y_file = np.empty((BATCH, int(IMGSIZE), int(IMGSIZE), 3))
        x_file = np.empty((BATCH, int(IMGSIZE), int(IMGSIZE), 3))
        for b in range(BATCH):
            # Image reading
            impath = pathFolder + fileNames[a*BATCH+b][0]
            originalimg = Image.open(impath)

            # Reducing image size to 200*200
            reducedimg = originalimg.resize((IMGSIZE, IMGSIZE))

            # Convert to numpy array
            npimg = np.array(reducedimg)

            # Normalize values
            npimg = npimg/255

            # Assign image to main array
            y_file[b] = npimg
            # create and save noise in the image
            noise = 40 / 256 * np.random.normal(0, 1, size=npimg.shape)
            x_file[b] = npimg + noise

        x_filename = outputFile_x + str(a) + '.npy'
        y_filename = outputFile_y + str(a) + '.npy'

        # Save data
        np.save(os.path.join('data', x_filename), x_file)
        np.save(os.path.join('data', y_filename), y_file)

prepare_data("train.txt", "x_train", "y_train")
prepare_data("val.txt", "x_test", "y_test")
