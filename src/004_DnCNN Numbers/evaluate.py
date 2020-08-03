# In this program we test an example of the neural network saved in model.h5

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

model = models.load_model('model.h5')

prediction = model.predict(x_test)
plt.imshow(prediction[35])
plt.show()
plt.imshow(x_test[35])
plt.show()
plt.imshow(y_test[35])
plt.show()