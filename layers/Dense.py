import numpy as np
import activations.activations as a
import activations.derivatived_activations as d

class Dense:

    def __init__(self, activation, output_size):
        self.weights = np.random.rand(self.inputs.shape[0]*self.inputs.shape[1],self.output_size)
        self.activation = activation
        self.output_size = output_size

    def feedforward(self, inputs):
        self.inputs = inputs
        
        self.func = a.Activation(np.dot(self.inputs, self.weights), self.activation)
        self.output = self.func.callFunc()
        return self.output

    def backprop(self, actual_output):
        #transpose of layer input
        trans_input = self.inputs.transpose()

        #difference of between actual output and layer output
        diff = np.subtract(actual_output.reshape(self.output.shape[0]),self.output)

        #define activation function
        result = a.Activation(self.output, self.activation)

        #dot product of trans_input and multiply by 2, diff and act_func
        self.temp_weight = np.dot(trans_input,2*diff*result.callFunc())
        
        #update weights with temprorary weights
        np.add(self.weight, self.temp_weight)

    def summary(self):
        print("Dense            "+str(self.output_size))
        print("________________________________________")
