#remove tensoflow console logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras.datasets import imdb
import numpy as np

class SentimentClassifier:
    
    def __init__(self):
        self.model = load_model("/home/atom/Desktop/Deep_Learning_Framework/framework/sources/imdb/imdb.model")

    def getExample(self):
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)
        return self.vectorize_sequences(train_data[0])

    def getResult(self, sentence):
        result = self.model.predict_classes(sentence)[0]
        if(result==1):
            return "Positive"
        elif(result==0):
            return "Negative"

    def vectorize_sequences(self, sequences,dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results