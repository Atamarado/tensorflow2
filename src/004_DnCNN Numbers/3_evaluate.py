# github.com/Atamarado
# In this program we test an example of the neural network saved in model.h5

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

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