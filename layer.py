import utils.random as r
import activations as a

class Layer:
    def __init__(self, input, output_size, activationFunc):
        self.input = input
        self.output_size = output_size
        self.activationFunc = getattr(a, activationFunc)
        self.weights = r.randomFillArray(len(input))
    
    def getOutput(self):
        outputs = []
        for a in range(self.output_size):
            sum = 0
            for i in range(len(self.input)):
                sum += self.weights[i]*self.input[i]
            outputs.append(self.activationFunc(sum))
        return outputs