#remove tensoflow console logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model

class SentimentAnalyzer:
    
    def __init__(self):
        self.model = load_model("/home/atom/Desktop/Deep_Learning_Framework/framework/sources/imdb/imdb.model")

    #def getResult(self):