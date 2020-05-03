import numpy as np
import activations.activations as a
import activations.derivatived_activations as d
import losses.losses as losses

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


        loss = losses.error(self.output, actual_output)

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
