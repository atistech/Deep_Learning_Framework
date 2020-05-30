import tensorflow as tf
from tensorflow import keras
import cv2

fashion_mnist = keras.datasets.fashion_mnist

#load fashion_mnist dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#cv2.imwrite('/home/atom/Desktop/Deep_Learning_Framework/example4.png', train_images[1])
train_images = train_images / 255.0

test_images = test_images / 255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

model.save("fashion.h5")