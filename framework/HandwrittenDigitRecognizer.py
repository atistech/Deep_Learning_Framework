#remove tensoflow console logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras.preprocessing import image
import cv2


class HandwrittenDigitRecognizer:
    
    def __init__(self):
        self.model = load_model("/home/atom/Desktop/Deep_Learning_Framework/framework/sources/mnist/mnist.model")
    
    def getResult(self, img_path):
        image = cv2.cv2.imread(img_path,cv2.cv2.IMREAD_GRAYSCALE)
        image = image.reshape((1, 28 * 28))
        image = image.astype('float32') / 255
        return self.model.predict_classes(image)[0]