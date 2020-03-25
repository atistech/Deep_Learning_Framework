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

        #initialize weight with input shape and output size
        self.weight = np.random.rand(input.shape[0]*input.shape[1],self.output_size)
        
        #
        self.output = self.activationFunc(np.dot(input, self.weight))
        return self.output, self.weight

    '''
        Backpropogation Function
        params: actual_output or network
    '''
    def backprop(self, actual_output):
        #transpose of layer input
        trans_input = self.input.transpose()

        #difference of between actual output and layer output
        diff = np.subtract(actual_output.reshape(self.output.shape[0]),self.output)

        #define activation function
        print(getattr(d, self.activationFunc.__name__))
        act_func = getattr(d, self.activationFunc.__name__)

        #dot product of trans_input and multiply by 2, diff and act_func
        self.temp_weight = np.dot(trans_input,2*diff*act_func(self.output))
        
        #update weights with temprorary weights
        np.add(self.weight, self.temp_weight)

    def summary(self):
        print("Dense            "+str(self.output_size))
        print("________________________________________")
