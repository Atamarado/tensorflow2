# We generate 4 files with this code, prepared to be fit in the neural network
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
import random

IMGSIZE = 200
BATCH = 200

def prepare_data(inputFile, outputFile_y):
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
        for b in range(BATCH):
            # Image reading
            impath = pathFolder + fileNames[a*BATCH+b][0]
            originalimg = Image.open(impath)

            # Reducing image size to 200*200
            reducedimg = originalimg.resize((IMGSIZE, IMGSIZE))

            # Convert to numpy array
            npimg = np.array(reducedimg)

            # Normalize values
            y_file[b] = npimg/255

        y_filename = outputFile_y + str(a) + '.npy'

        # Save data
        np.save(os.path.join('data', y_filename), y_file)

prepare_data("train.txt", "y_train")
prepare_data("val.txt", "y_test")
