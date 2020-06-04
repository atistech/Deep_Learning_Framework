#remove tensoflow console logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ImageClassifier:

    def __init__(self):
        self.model = tf.keras.models.load_model("/home/atom/Desktop/Deep_Learning_Framework/framework/sources/cifar10/cifar10.h5")

    def getResult(self, img_path):
        image = cv2.cv2.imread(img_path,cv2.cv2.IMREAD_COLOR)
        image = (np.expand_dims(image,0))

        self.predictions = self.model.predict(image)[0]
        
        return self.getClassName(self.predictions)

    def getClassName(self, predictions):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        max = predictions[0] 
        for x in predictions: 
            if x > max : 
                max = x
        index = 0
        for i in predictions:
            if(i!=max):
                index=index+1
            else:
                break
        return class_names[index]

    def showResultFigure(self, img_path):
        img = mpimg.imread(img_path)
        plt.figure(figsize=(6,3))
        plt.imshow(img)
        plt.title("Prediction is "+str(self.getResult(img_path)))
        plt.show()
  