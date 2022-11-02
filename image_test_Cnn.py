import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from keras.datasets import cifar10




def takePhoto(img):  
    
    
    
    img_matrix=cv.imread(img)
    
    new_array=cv.resize(img_matrix, (32,32))
    X = np.array(new_array).reshape(-1,32,32,3)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print(X.shape)
    return X









#confusion_m=tf.math.confusion_matrix(labels=y_true,predictions=predictions).np()

def main():
    model = tf.keras.models.load_model('C:\\Users\\IWC42\\Desktop\\IMAGESCNN\\cifar_test.model')
    test_image= takePhoto("C:\\Users\\IWC42\Desktop\\IMAGESCNN\\CNNIMAGES\\BerberisPinnata\\medium1.jpg")
    print(test_image.shape)
    imshow('image',test_image)
    test_image=test_image/255.0
    predictions=model.predict(test_image,verbose=2)
    

main()