import numpy as np
import network
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
'''
img = mpimg.imread('deneme.png')
plt.imshow(img)'''

inputs = np.random.rand(1, 784)
output = np.random.rand(10, 1)


n1 = network.Network(inputs, output)
n1.Dense(512, "sigmoid")
n1.Dense(10, "sigmoid")
n1.compile()

'''
data = np.load('/home/atom/Documents/mnist.npz')
train_data = data['x_train'][0].reshape((1, 28*28))
train_data = train_data.astype('float32') / 255
output = np.array([8])
'''