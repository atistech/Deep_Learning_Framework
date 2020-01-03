import random

def randomFillArray(size):
    array = []
    for i in range(size):
        array.append(random.randint(0, 1))
    return array