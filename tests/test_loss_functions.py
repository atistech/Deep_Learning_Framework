import sys
sys.path.append('losses/')

import losses as l
import numpy as np

def test_mean_squared_error():
    input = np.array([1, 2, 3, 4, 5])
    output = np.array([10])
    assert l.mean_squared_error(input, output) == 51