import numpy as np
import activations as a

class Layer:
    def __init__(self, output_size, activationFunc):
        self.output_size = output_size
        self.activationFunc = getattr(a, activationFunc)
    
    def getOutput(self, input):
        self.weights = np.random.rand(input.shape[0]*input.shape[1],self.output_size)
        self.output = self.activationFunc(np.dot(input, self.weights))
        return self.output


class Dense(Layer):
    def summary(self):
        print("Dense            "+str(self.output_size))
        print("________________________________________")
