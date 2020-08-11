# We generate 4 files with this code, prepared to be fit in the neural network
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMGSIZE = 100

def prepare_data(inputFile, outputFileImg, outFileLabels):
    # Read fileNames and y_labels
    fileNames = np.loadtxt(inputFile, delimiter=' ', dtype=np.str)

    nImages = int(np.shape(fileNames)[0])

    # Array of images
    array = np.empty((nImages, int(IMGSIZE), int(IMGSIZE)))

    # For everyImage:
    for a in range(nImages):
        # Image reading
        impath = 'images/' + fileNames[a][0]
        originalimg = Image.open(impath)

        # Reducing image size to 100*100
        reducedimg = originalimg.resize((IMGSIZE, IMGSIZE))

        # Changing image to grayscale
        grayscaleimg = reducedimg.convert('L')

        # Convert to numpy array
        npimg = np.array(grayscaleimg)

        # Normalize values
        npimg = npimg/255

        # Assign image to main array
        array[a] = npimg

    # Prepare data to [100, 100, 1]
    array = np.reshape(array, (nImages, IMGSIZE, IMGSIZE, 1))

    # Array of gender labels
    labels = fileNames[:, 1].astype(float)

    # Save data
    np.save(outputFileImg, array)
    np.save(outFileLabels, labels)


prepare_data("train.txt", "trainData.npy", "trainLabels.npy")
prepare_data("val.txt", "valData.npy", "valLabels.npy")
