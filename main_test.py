import numpy as np
import layers
import network

input = np.random.rand(1, 784)

n1 = network.Network(input)
n1.add(layers.Dense(512, 'relu'))
n1.add(layers.Dense(10, 'softmax'))
n1.summary()
#print(n1.feedforward())