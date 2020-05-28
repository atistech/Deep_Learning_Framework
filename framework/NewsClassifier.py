#remove tensoflow console logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras.datasets import reuters
import numpy as np


class NewsClassifier:

    def __init__(self):
        self.model = load_model("/home/atom/Desktop/Deep_Learning_Framework/framework/sources/reuters/reuters.model")

    def getExample(self):
        (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
        return self.vectorize_sequences(train_data[0])
    
    def getResult(self, new):
        return self.model.predict_classes(new)[0]

    def vectorize_sequences(self, sequences,dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results