import numpy as np

class Network:
    all_layers = []
    weights = []

    def __init__(self, input, actual_output):
        self.input = input
        self.actual_output = actual_output
        self.last_output = input

    def add(self, Layer):
        self.all_layers.append(Layer)

    def feedforward(self):
        for l in self.all_layers:
            self.last_output, temp_weights = l.feedforward(self.last_output)
            self.weights.append(temp_weights)
        return self.last_output

    def backprop(self):
        for l in self.all_layers:
            l.backprop(l.input)
            
    def summary(self):
        print("Layer (type)     OutputShape     Param #")
        print("========================================")
        for l in self.all_layers:
            l.summary()