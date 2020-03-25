import numpy as np
from layers import Dense
import network

data = np.load('/home/atom/Documents/mnist.npz')
train_data = data['x_train'][0].reshape((1, 28*28))
train_data = train_data.astype('float32') / 255
output = np.array([8])

n1 = network.Network()
n1.add(Dense.Dense(512, 'logsigmoid'))
n1.add(Dense.Dense(10, 'logsigmoid'))
n1.compile(train_data, output)