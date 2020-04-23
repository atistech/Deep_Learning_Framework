import numpy as np
import activations.activations as a
import activations.derivatived_activations as d

class Dense:

    def __init__(self, activation, output_size):
        self.activation = getattr(a, activation)
        self.output_size = output_size
        self.bias = np.zeros((1, self.output_size))

    def feedforward(self, inputs):
        self.inputs = inputs
        self.weights = np.random.rand(self.inputs.shape[1],self.output_size)

        self.output = self.activation(np.dot(self.inputs, self.weights)+self.bias)
        return self.output

    def backprop(self, actual_output):
        #transpose of layer input
        transpose_input = self.inputs.transpose()

        #difference of between actual output and layer output
        print(self.output.shape)
        np.reshape(actual_output, self.output.shape)
        print(actual_output.shape)
        difference = np.subtract(actual_output, self.output)

        #define activation function
        derivate_func = getattr(d, self.activation.__name__)
        result = derivate_func(self.output)

        #dot product of trans_input and multiply by 2, diff and act_func
        self.temp_weights = np.dot(transpose_input,2*difference*result)
        
        #update weights with temprorary weights
        np.add(self.weights, self.temp_weights)

    def summary(self):
        print("Dense            "+str(self.output_size))
        print("________________________________________")
