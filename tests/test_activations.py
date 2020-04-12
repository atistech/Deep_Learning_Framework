import sys
sys.path.append('activations/')

import activations as a
import numpy as np

def test_logsigmoid():
    input = np.array([0])
    fun = a.Activation(input, "logsigmoid")
    assert fun.callFunc() == np.array([0.5])

def test_linear():
    input = np.array([1])
    fun = a.Activation(input, "linear")
    assert fun.callFunc() == np.array([1])