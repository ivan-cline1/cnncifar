from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import cifar10
from matplotlib import pyplot
from pandas import Index
from keras.optimizers import Adam
from itertools import permutations
import numpy as np
import os



def summarry(history,fileName,denseLayerPerms): #code found online for displaying a graph of my accuracy
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.title(f'model accuracy for{denseLayerPerms}')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'val'], loc='upper left')
    pyplot.savefig(fileName+"\\"+f'{denseLayerPerms[0]}_{denseLayerPerms[1]}_{denseLayerPerms[2]}.png')
    pyplot.close()
    

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

def model_(trainX,learningRate,dropoutRate,listOfDenseLayerSizes):
    
    listOfDenseLayerSizes[0]
    listOfDenseLayerSizes[1]
    listOfDenseLayerSizes[2]


    model = Sequential()




    model.add(Conv2D(64,(3,3),input_shape=trainX.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Flatten())

    model.add(Dense(listOfDenseLayerSizes[0]))
    model.add(Activation("relu"))
    model.add(Dropout(dropoutRate))

    model.add(Dense(listOfDenseLayerSizes[1]))
    model.add(Activation("relu"))
    model.add(Dropout(dropoutRate))

    model.add(Dense(listOfDenseLayerSizes[2]))
    model.add(Activation("relu"))
    model.add(Dropout(dropoutRate))

    model.add(Dense(10,activation='softmax'))

    opt=Adam(learning_rate=learningRate)
    model.compile(optimizer=opt,loss='MeanSquaredError',metrics=['accuracy'])
    return model





def main():


    epochs,batchSize,learningRate,momentum,dropoutRate,modelName,= int(input('Amt of Epochs: ')),int(input('Batch Size:')), float(input("learningRate")),float(input("momentum")),float(input("Dropout Rate (0 if you dont want any):")),input("Model Name:")
    print(f'{epochs}\n{batchSize}\n{learningRate}\n{momentum}\n{dropoutRate}\n{modelName}\n')
    trainX,trainY_encoded,testX,testY_encoded = loadData()

    #print(trainX.shape[1:])
    #print(f'Training Images: {trainX.shape[0]} \nTesting Images: {testX.shape[0]}')
    #displayImages(trainX,trainY_encoded)


    listOfDenseLayerSizes = [64,128,256,512]
    listOfDenseLayerSizes=list(permutations(listOfDenseLayerSizes,3))
    for i in listOfDenseLayerSizes:
        #set up file directory for models
        directory=f'{i[0]}_{i[1]}_{i[2]}'
        parentdir=f'C:/Users/IWC42/Desktop/CNN_Cifar10/(Personal Models)'
        fileDir = os.path.join(parentdir, directory)
        os.mkdir(fileDir)
       

        model_new=model_(trainX,learningRate,dropoutRate,i)
        hist = model_new.fit(trainX,trainY_encoded,epochs=epochs,batch_size=batchSize,validation_data=(testX,testY_encoded),verbose=1)
        loss,accuracy=model_new.evaluate(testX,testY_encoded,verbose=1)


        print(f"loss: {loss}")
        print(f"accuracy: {accuracy}")
        summarry(hist,fileDir,i)
    

        with open(fileDir+"\\"+directory+".txt",'w') as f:
            f.write(f'Layer 1: 64 conv2d, reLu activation, 3x3 filter size,maxPooling size(2x2)\n')
            f.write(f'Layer 2: 128 conv2d, reLu activation, 3x3 filter size,maxPooling size(2x2)\n')
            f.write(f'Layer 2: 64 conv2d, reLu activation, 3x3 filter size,maxPooling size(2x2)\n')
            f.write('DENSE LAYERS\n')
            f.write(f'Order of dense Layers:{directory}\n')
            f.write(f'Dropout Rate for dense layers:{dropoutRate}\nOptimization:Adam\nLearningRate:{learningRate}\n')
            f.write(f'Epochs:{epochs}\nBatch Size:{batchSize}\nLoss:{loss}\nAccuracy:{accuracy}')
            f.close()
        
        model_new.save(f'{fileDir}\\{i[0]}_{i[1]}_{i[2]}.model')
        del model_new 

main()
