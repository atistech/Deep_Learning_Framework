import utils.random as r
import layer
import network

input = r.randomFillArray(784)

n1 = network.Network(input)
n1.addLayer(512, 'logsigmoid')
n1.addLayer(10, 'relu')
print(n1.getResult())