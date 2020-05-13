from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

#load mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#network architecture
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

#compile step
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#preparing train and test
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#preparing labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#training network
model.fit(train_images, train_labels, epochs=5, batch_size=128)

#save model
model.save("mnist.model")