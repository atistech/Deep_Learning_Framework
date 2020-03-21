import numpy as np
from layers import Dense
import network

input = np.random.rand(1, 784)

n1 = network.Network(input, input)
n1.add(Dense.Dense(512, 'relu'))
n1.add(Dense.Dense(10, 'softmax'))
#n1.summary()
n1.backprop()
print(n1.feedforward())