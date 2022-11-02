from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import cifar10
from matplotlib import pyplot
from pandas import Index
from keras.optimizers import Adam
import numpy as np




def loadData():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainY_encoded =to_categorical(trainY)
    testY_encoded = to_categorical(testY)
    trainX,testX = trainX/255.0, testX/255.0

    return trainX,trainY_encoded,testX,testY_encoded


def displayImages(data,data_encoded):

    categories = ['Airplane','Automobile','Bird','cat','deer','dog','frog','horse','ship','truck']
    f= pyplot.figure()
    
    for i in range(15):
        f.add_subplot(5,3,i + 1)
        pyplot.axis('off')
        x = (np.where(data_encoded[i]==1))
        pyplot.title(categories[x[0][0]])
        pyplot.imshow(data[i])

   
    
    pyplot.show(block=True)

def model_(trainX):

    model = Sequential()

    model.add(Conv2D(64,(3,3), input_shape=trainX.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(256,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dense(128))
    model.add(Activation("relu"))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(10,activation='softmax'))

    
    model.compile(optimizer='adam',loss='MeanSquaredError',metrics=['accuracy'])
    return model





def main():
    trainX,trainY_encoded,testX,testY_encoded = loadData()

    print(trainX.shape[1:])
    print(f'Training Images: {trainX.shape[0]} \nTesting Images: {testX.shape[0]}')


    displayImages(trainX,trainY_encoded)
    
    model_new=model_(trainX)

    model_new.fit(trainX,trainY_encoded,epochs=20,batch_size=64,validation_data=(testX,testY_encoded),verbose=1)

    loss,accuracy=model_new.evaluate(testX,testY_encoded,verbose=1)

    print(f"loss: {loss}")
    print(f"accuracy: {accuracy}")
    model_new.save('C:\\Users\\IWC42\\Desktop\\IMAGESCNN\\cifar_test.model')
    del model_new 

main()