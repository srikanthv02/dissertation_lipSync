#.15 is the final, use it bruh :D

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import glob
import pandas as pd
encoder = OneHotEncoder(sparse=False)   
import csv
import matplotlib.pyplot as pyplot

from keras import backend as K
from keras import losses

from keras.optimizers import SGD
from keras import optimizers

from sklearn.preprocessing import StandardScaler
from keras.utils import plot_model

path = 'maleAudio/*.txt'
files = glob.glob(path)

pathForPoints = 'maleVideo/*.csv'
fileForPoints = glob.glob(pathForPoints)
windowSizeForNetwork = 11
concatenatedData = np.empty((0,(38*windowSizeForNetwork)))
listOfAllPoints = np.empty((0,72))

#for normalizing the inputs
sc = StandardScaler()

def window_stack(a, stepsize, width):
    #n = a.shape[0]
    return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width))

'''
def window_stack(a, stepsize, width):
    half_window_size = (width-1)/2
    print(half_window_size)
    #n = a.shape[0]
    #print(np.hstack( a[i-half_window_size:1+n+i+half_window_size:stepsize] for i in range(0,width)))
    return np.hstack( a[int(i-half_window_size)+int(i+half_window_size-width) or None:stepsize] for i in range(0,width) )
    [:-5] + [1:-4] + [2:-3]
'''

#extracting video features from list of csv files(Flat 72 points that will be the output of this network)
for file in fileForPoints:
    listOfPoints = pd.read_csv(file,header=None)

    listOfPoints = listOfPoints[listOfPoints.columns[1:3]]
    numpyList = np.asanyarray(listOfPoints)
    numpyList = numpyList.flatten()
    numpyList = numpyList[2:]
    numpyList = np.reshape(numpyList, (-1, 72))
    listOfAllPoints = np.concatenate((listOfAllPoints, numpyList), axis=0)
    
    #np.savetxt("testVideo.csv", listOfAllPoints, fmt='%s')    

#
for file in files:
    try:
        fileList = pd.read_csv(file, sep=',', skiprows=1)
        numpyList = np.asarray(fileList)
        
        #inverted = argmax(numpyList[30])
        #print(inverted)
        #print(file,numpyList)
        #print(numpyList.shape)
    except ValueError:
        print("errorFile", file)
    numpyList = window_stack(numpyList,1,11)
    #print(numpyList)
    concatenatedData = np.concatenate((concatenatedData,numpyList),axis=0)    
    #print(concatenatedData)
    #np.savetxt("textSri.csv", concatenatedData) 


## for testing

pathForTestingAudio = 'volunteerTestingAudio/*.txt'
filesForTesting = glob.glob(pathForTestingAudio)

pathForPointsTesting = 'volunteerTestingVideo/*.csv'
fileForPointsTestingVideos = glob.glob(pathForPointsTesting)
X_test = np.empty((0,(38*windowSizeForNetwork)))
Y_test = np.empty((0,72))



#extracting video features from list of csv files(Flat 72 points that will be the output of this network)
for file in fileForPointsTestingVideos:
    listOfPoints = pd.read_csv(file,header=None)

    listOfPoints = listOfPoints[listOfPoints.columns[1:3]]
    numpyList = np.asanyarray(listOfPoints)
    numpyList = numpyList.flatten()
    numpyList = numpyList[2:]
    numpyList = np.reshape(numpyList, (-1, 72))
    Y_test = np.concatenate((Y_test, numpyList), axis=0)
    #Y_test = sc.fit_transform(Y_test)
    np.savetxt("testVideoMaleOnly.csv", Y_test, fmt='%s')  

#reading data from the input file(audio files)
for file in filesForTesting:
    try:
        fileList = pd.read_csv(file, sep=',', skiprows=1)
        numpyListTest = np.asarray(fileList)
        
    except ValueError:
        print("errorFile", file)
    numpyListTest = window_stack(numpyListTest,1,11)
    X_test = np.concatenate((X_test,numpyListTest),axis=0)    
    #np.savetxt('xTest.csv',X_test,fmt='%s')


#for 9 #listOfAllPoints = listOfAllPoints[0:-13281]#trying to adjust the output and input layer

#for 11
#listOfAllPoints = listOfAllPoints[0:-14673]#trying to adjust the output and input layer

#for 7
#listOfAllPoints = listOfAllPoints[0:-13273]#trying to adjust the output and input layer

#for 13
#listOfAllPoints = listOfAllPoints[0:-15373]#trying to adjust the output and input layer

#for male
listOfAllPoints = listOfAllPoints[0:-2844]#trying to adjust the output and input layer

#for9#Y_test = Y_test[0:-1756]#trying to adjust the output and input layer

#for 11
#Y_test = Y_test[0:-1118]#trying to adjust the output and input layer

#for 7
#Y_test = Y_test[0:-1010]#trying to adjust the output and input layer

#for 13
#Y_test = Y_test[0:-1109]#trying to adjust the output and input layer

#for single video
#Y_test = Y_test[0:-9]#trying to adjust the output and input layer

#Y_test = Y_test[0:-3188]#trying for male speakers as testing

Y_test = Y_test[0:-344]#trying for male speakers as testing

print(concatenatedData.shape)
print(listOfAllPoints.shape)

print(X_test.shape)
print(Y_test.shape)

model = Sequential()
model.add(Dense(3000, activation='relu', input_dim=(38*windowSizeForNetwork)))
model.summary()
model.add(Dense(3000, activation='relu'))
model.summary()
model.add(Dense(3000, activation='relu'))
model.summary()
model.add(Dropout(0.5))
model.add(Dense(72,activation='linear'))#final output layer 
model.summary()
plot_model(model, to_file='model.png')
#0.00001 made it move

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

model.compile(optimizer='sgd', loss=root_mean_squared_error, metrics=['mape',root_mean_squared_error])
history = model.fit(concatenatedData, listOfAllPoints, epochs=20, batch_size=128)



#print(concatenatedData.shape)
score = model.evaluate(X_test, Y_test, batch_size=32)
print(score)


prediction = model.predict(X_test)

np.savetxt("testPredictionMale.csv", prediction,fmt='%4d') 


pyplot.plot(history.history['mean_absolute_percentage_error'])
#pyplot.plot(history.history['root_mean_squared_error'])
pyplot.show()
#pyplot.plot(history.history['cosine_proximity'])

'''
print(concatenatedData[0,:].shape)
testData = concatenatedData[0,:].reshape(1,418)
test = model.predict(testData)
print(test.shape)

print(test)
pyplot.plot(history.history['mean_absolute_percentage_error'])
pyplot.plot(history.history['cosine_proximity'])

pyplot.plot(test)
pyplot.show()
#print("after",test.shape)

#plt.show()
'''
