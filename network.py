import numpy as np
import layers

class Network:
    all_layers = []

    def __init__(self, input):
        self.input = input
        self.last_output = input

    def add(self, Layer):
        self.all_layers.append(Layer)

    def feedforward(self):
        for l in self.all_layers:
            #print(self.last_output.shape)
            self.last_output = l.getOutput(self.last_output)
        return self.last_output
            
    def summary(self):
        print("Layer (type)     OutputShape     Param #")
        print("========================================")
        for l in self.all_layers:
            l.summary()