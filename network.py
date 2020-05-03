import numpy as np
import layers.Dense as dense

class Network:
    all_layers = []

    def __init__(self, inputs, actual_output):
        self.actual_output = actual_output
        self.last_output = inputs

    def add(self, Layer):
        self.all_layers.append(Layer)

    def compile(self):
        #feedforward
        for l in self.all_layers:
            self.last_output = l.feedforward(self.last_output)
        #backpropagation
        for l in self.all_layers.reverse():
            l.backprop(self.actual_output)
        self.all_layers.reverse()
        #self.result()

    def result(self):
        print(self.last_output)
            
    def Dense(self, size, act):
        d = dense.Dense(act, size)
        self.all_layers.append(d)

    def summary(self):
        print("Layer (type)     OutputShape     Param #")
        print("========================================")
        for l in self.all_layers:
            l.summary()