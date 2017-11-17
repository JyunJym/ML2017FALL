## Input data manipulations 
import numpy as np
import pandas as pd
from numpy import argmax
from matplotlib import pyplot as plt
import sklearn 
import argparse
import os, sys

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.utils import np_utils # this will make the label to one-hot encoding 
from keras.preprocessing.image import ImageDataGenerator


data_set = pd.read_csv(sys.argv[1], encoding = "big5")
#test_set = pd.read_csv('test.csv', encoding = "big5")
train_temp = data_set.iloc[:,1].values
test_temp = test_set.iloc[:,1].values 
SIZE = train_temp.shape[0] 
TEST_SIZE = test_temp.shape[0]
label = np.ndarray(shape = (SIZE,1))
label[:,0] = data_set.iloc[:,0].values
train = np.ndarray(shape=(SIZE,48*48))
test = np.ndarray(shape=(TEST_SIZE,48*48))

for i in range(SIZE):
    train[i] = np.fromstring(train_temp[i], dtype=int, sep=' ')
for i in range(TEST_SIZE):
    test[i] = np.fromstring(test_temp[i], dtype=int, sep=' ')

train = np.reshape(train, (SIZE,48,48,1))
test = np.reshape(test, (TEST_SIZE,48,48,1))
#plt.imshow(train[10])
#plt.show()   #These are only for showing the images 

# Speed up the processing speed
train = train.astype('float32')
test = test.astype('float32')
train /= 255
test /= 255

Label = np_utils.to_categorical(label, 7)


train_80 = train[0:14000,:]
train_20 = train[14000:28709,:]

Label_80 = Label[0:14000,:]
Label_20 = Label[14000:28709,:]


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

print(1)

## Building a CNN model 


Classifier = Sequential() # initialize the model

Classifier.add(Conv2D(32,3,3, input_shape = (48,48,1), activation = 'relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))

Classifier.add(Conv2D(32,3,3, activation = 'relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))

#Classifier.add(Convolution2D(32,5,5, activation='relu'))
#Classifier.add(MaxPooling2D(pool_size=(2,2)))
#Classifier.add(MaxPooling2D(pool_size=(3,3)))
#Classifier.add(Dropout(0.3))

Classifier.add(Flatten())
Classifier.add(Dense(input_dim = 128, units = 256, activation = 'relu'))

for i in range(8):
    Classifier.add(Dense(units = 256, activation = 'relu'))
    Classifier.add(Dropout(0.2))

Classifier.add(Dense(units = 7, activation = 'softmax'))

Classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Classifier.fit(train_20, Label_20, batch_size=100, epochs=40)

Classifier.save("./good.h5")

result = Classifier.evaluate(train,Label)
print(result[0])
print(result[1])

result1 = Classifier.evaluate(train_80,Label_80)
print(result1[0])
print(result1[1])

result2 = Classifier.evaluate(train_20,Label_20)
print(result2[0])
print(result2[1])


sol_temp = Classifier.predict(test)
sol = np.ndarray((TEST_SIZE,1))
sol = np.argmax(sol_temp, axis=1)
sol = sol.astype(int)

'''
import csv
prediction=[]
prediction.append(list(sol))

prediction_write = []
temp=[]
temp.append('id')
temp.append('label')
prediction_write.extend([temp])


for i in range(TEST_SIZE): #test data size
  temp=[('%s' %(i),sol[i])] 
  #temp.append('id_%s' %(i))
  prediction_write.extend(temp)#temp.append(sol[i])
  #prediction_write.extend(sol[i])



f = open('Ultra.csv',"w")
w = csv.writer(f)
for row in prediction_write:
	w.writerow(row)
f.close()
'''

