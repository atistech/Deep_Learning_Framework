#remove tensoflow console logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ClothingClassifier:

    def __init__(self):
        self.model = tf.keras.models.load_model("/home/atom/Desktop/Deep_Learning_Framework/framework/sources/fashion/fashion.h5")

    def getResult(self, cloth_img):
        image = cv2.cv2.imread(cloth_img,cv2.cv2.IMREAD_GRAYSCALE)
        image = (np.expand_dims(image,0))
        
        probability_model = tf.keras.Sequential([self.model, 
                                                tf.keras.layers.Softmax()])
        self.predictions = probability_model.predict(image)[0]
        return self.getClothName(self.predictions)

    def getClothName(self, predictions):
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        index = 0
        for i in predictions:
            if(i!=1):
                index+1
        return class_names[index]

    def showResultFigure(self, cloth_img):
        img = mpimg.imread(cloth_img)
        plt.figure(figsize=(6,3))
        plt.imshow(img)
        plt.title("Prediction is "+str(self.getResult(cloth_img)))
        plt.show()