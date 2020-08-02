import tensorflow as tf

mnist=tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images/255.0

test_images=test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

# Definimos un modelo con 3 capas de neuronas:
model= tf.keras.Sequential([
    # Capas convolucionales

    # Realizamos 64 convoluciones, típicamente es un número cualquier, pero del orden de x*32
    # El tamaño de la convolución será de 3*3
    # La función de activation 'relu' devuelve x si es número es >0, sino devuelve 0
    # Recibirá un input de 28*28, con solo 1 byte para definir el color
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),

    # Las capas de convolución devuelven tantas imágenes como filtros haya (64)

    # Capa de pooling
    # Es un MaxPooling porque cogeremos el valor máximo
    # De cada 2*2 píxeles, nos quedaremos con el que tenga el valor más alto
    # De cada 4 píxeles que había, cogemos solo 1. La resolución es ahora 1/4 de la original
    tf.keras.layers.MaxPooling2D(2,2),

    # Otra convolución, con su respectivo pooling.
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),


    #
    #
    # Capas iguales a la Week2

    # input_shape define el tipo de datos que tendremos (en este caso, imágenes de 28x28 px) (se declara un array)
    tf.keras.layers.Flatten(input_shape=(28,28)),

    # Hidden layer, con 128 neuronas, que trabajan de manera parecida a una función,
    # donde w0x0+w1x1+...+w783x783=n, donde n es el tipo de solución que se dará
    tf.keras.layers.Dense(128,activation=tf.nn.relu),

    # Tenemos una capa con 10 neuronas, ya que tenemos que clasificar 10 tipos de ropa diferente
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Muestra por pantalla de las características de cada capa
model.summary()

model.fit(training_images,training_labels,epochs=5)

# Comprobamos la eficiencia del test mediante imágenes de fuera de la database
test_loss=model.evaluate(test_images,test_labels)

#Observaciones:
# Por qué las convoluciones entran dentro de la definición del modelo secuencial
# ¿Las convoluciones son iguales o no en cada epoch? ¿Se ejecutan en cada epoch?
# He intentado poner 4 capas de convolución+pooling, pero se muere :\
# Como ajusto el input para poder hacer una operación sin convoluciones?


# RESULTADOS CAMBIANDO PARÁMETROS

# 2 capas de 64 convoluciones:      0.9300 database      0.9100 externo     20s/epoch (aprox)
# 2 capas de 32 convoluciones:      0.9158 database      0.8997 externo
# 2 capas de 16 convoluciones:      0.9045 database      0.8946 externo

# 1 capa de 64 convoluciones:       0.9456 database      0.9126 externo     33s/epoch (aprox)

# 3 capas de 64 convoluciones:      0.9001 database      0.8836 externo     40s/epoch (aprox)