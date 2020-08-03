# In this program I read, adapt and generate more images from a database to train and
# evaluate a DnCNN. The database is mnist numbers

from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
IMG_SIZE = 28

# Leemos los datos
(y_train, yyy), (y_test, yyz) = mnist.load_data()

# Data normalization
y_train = y_train/256
y_test = y_test/256

# We get over a million samples
#for i in range(3):
   #y_train = np.concatenate((y_train, y_train))

# Shuffle order
#np.random.shuffle(y_train)

# Add noise
x_train = y_train + 20/256 * np.random.normal(0, 1, size=y_train.shape)
x_test = y_test + 20/256 * np.random.normal(0, 1, size=y_test.shape)

# plt.imshow(x_test[6])
# plt.show()
# plt.imshow(y_test[6])
# plt.show()

# Resize
x_train = np.reshape(x_train, (x_train.shape[0], IMG_SIZE, IMG_SIZE, 1))
y_train = np.reshape(y_train, (y_train.shape[0], IMG_SIZE, IMG_SIZE, 1))
x_test = np.reshape(x_test, (x_test.shape[0], IMG_SIZE, IMG_SIZE, 1))
y_test = np.reshape(y_test, (y_test.shape[0], IMG_SIZE, IMG_SIZE, 1))

# Save database to txt
np.save('x_train', x_train)
np.save('y_train', y_train)
np.save('x_test', x_test)
np.save('y_test', y_test)





