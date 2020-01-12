import random
import layer
import network

input = []
for i in range(784):
    input.append(random.randint(0, 1))


n1 = network.Network(input)
n1.addLayer(512, 'logsigmoid')
n1.addLayer(10, 'relu')
print(n1.getResult())