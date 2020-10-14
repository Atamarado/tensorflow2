# github.com/Atamarado
# In this program we test an example of the neural network saved in model.h5

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from PIL import Image
import os

IMG_SIZE = 150
SAMPLE = 1

# Try a simple batch for example
y_test = np.load(os.path.join('data', 'y_test0.npy'))

# Generate noisy images
noise = 20 / 256 * np.random.normal(0, 1, size=y_test.shape)
x_test = y_test + noise

model = models.load_model('model.h5')

prediction = model.predict(x_test)

# # Denormalize
# prediction = prediction*255
# prediction = prediction.astype(int)
# 
# print(np.amax(prediction))
# print(np.amin(prediction))

#plt.imshow(prediction[7])
#plt.show()
#plt.imshow(x_test[7])
#plt.show()
#plt.imshow(y_test[7])
#plt.show()

rows = 3
cols = 3

fig = plt.figure()

print(prediction[0])

for i in range(cols):
    fig.add_subplot(rows, cols, i+1)
    plt.imshow(x_test[i+SAMPLE])
    plt.axis('off')
    fig.add_subplot(rows, cols, cols+i+1)
    plt.imshow(prediction[i+SAMPLE])
    plt.axis('off')
    fig.add_subplot(rows, cols, cols*2+i+1)
    plt.imshow(y_test[i+SAMPLE])
    plt.axis('off')

plt.suptitle("With noise vs. Denoised vs. Original", ha='center', fontsize=16)

plt.savefig('results.png')

# Now we will try to predict an image with noise that isn't a number

# Reading
noiseimg = Image.open('example.jpg')
# Resizing
noiseimg = noiseimg.resize((IMG_SIZE, IMG_SIZE))
# Convert to numpy array
noisenp = np.array(noiseimg)
# Normalizing
noisenp = noisenp/256

# Reshaping to 100*100*1
noisenp = np.reshape(noisenp, (1, IMG_SIZE, IMG_SIZE, 3))

denoised = model.predict(noisenp)

# Prepare plot output
fig = plt.figure()

fig.add_subplot(1, 2, 1)
plt.imshow(noisenp[0])
plt.axis('off')

fig.add_subplot(1, 2, 2)
plt.imshow(denoised[0])
plt.axis('off')

plt.suptitle("Input with noise vs. DnCNN Output", ha='center', fontsize=16)

plt.savefig('external_results.png')