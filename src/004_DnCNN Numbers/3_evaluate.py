# github.com/Atamarado
# In this program we test an example of the neural network saved in model.h5

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from PIL import Image

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

model = models.load_model('model.h5')

prediction = model.predict(x_test)
#plt.imshow(prediction[35])
#plt.show()
#plt.imshow(x_test[35])
#plt.show()
#plt.imshow(y_test[35])
#plt.show()

rows = 3
cols = 10

fig = plt.figure()

for i in range(cols):
    fig.add_subplot(rows, cols, i+1)
    plt.imshow(x_test[i])
    plt.axis('off')
    fig.add_subplot(rows, cols, cols+i+1)
    plt.imshow(prediction[i])
    plt.axis('off')
    fig.add_subplot(rows, cols, cols*2+i+1)
    plt.imshow(y_test[i])
    plt.axis('off')

plt.suptitle("With noise vs. Denoised vs. Original", ha='center', fontsize=16)

plt.savefig('results.png')

# Now we will try to predict an image with noise that isn't a number

# Reading
noiseimg = Image.open('example.jpg')
# Putting into grayscale
noiseimg = noiseimg.convert('L')
# Resizing
noiseimg = noiseimg.resize((28, 28))
# Convert to numpy array
noisenp = np.array(noiseimg)
# Normalizing
noisenp = noisenp/256

# Reshaping to 100*100*1
noisenp = np.reshape(noisenp, (1, 28, 28, 1))

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