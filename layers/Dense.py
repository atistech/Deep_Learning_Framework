import numpy as np
from activations import activations as a
from activations import derivatived_activations as d


class Dense:
    def __init__(self, output_size, activationFunc):
        self.output_size = output_size
        self.activationFunc = getattr(a, activationFunc)
        self.input = None
        self.weight = None
        self.output = None

    def feedforward(self, input):
        self.input = input
        self.weight = np.random.rand(input.shape[0]*input.shape[1],self.output_size)
        self.output = self.activationFunc(np.dot(input, self.weight))
        return self.output, self.weight

    def backprop(self, actual_output):
        self.temp_weight = np.dot(self.input.T,(2*(actual_output-self.output)*getattr(d, self.activationFunc.__name__)))
        np.add(self.weight, self.temp_weight)

    def summary(self):
        print("Dense            "+str(self.output_size))
        print("________________________________________")
