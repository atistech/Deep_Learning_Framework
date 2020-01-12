import random
import activations as a

class Layer:
    def __init__(self, input, output_size, activationFunc):
        self.input = input
        self.output_size = output_size
        self.activationFunc = getattr(a, activationFunc)
        self.weights = []
        for i in range(len(input)):
            self.weights.append(random.randint(0, 1))
    
    def getOutput(self):
        outputs = []
        for a in range(self.output_size):
            sum = 0
            for i in range(len(self.input)):
                sum += self.weights[i]*self.input[i]
            outputs.append(self.activationFunc(sum))
        return outputs