import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

#Adaptamos los datos a valores normalizados
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Definimos un modelo con 3 capas de neuronas:
model= tf.keras.Sequential([
    # input_shape define el tipo de datos que tendremos (en este caso, imágenes de 28x28 px) (se declara un array)
    tf.keras.layers.Flatten(input_shape=(28,28)),

    # Hidden layer, con 128 neuronas, que trabajan de manera parecida a una función,
    # donde w0x0+w1x1+...+w783x783=n, donde n es el tipo de solución que se dará
    tf.keras.layers.Dense(128,activation=tf.nn.relu),

    # Tenemos una capa con 10 neuronas, ya que tenemos que clasificar 10 tipos de ropa diferente
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

# Optimizamos y compilamos el modelo, en este caso 5 veces (epochs=5)
model.compile(optimizer = tf.compat.v1.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=9)

# Ponemos a prueba nuestra red neuronal, con imágenes que no estan en el dataset introducido antes
model.evaluate(test_images, test_labels)