## Input data manipulations 
import numpy as np
import pandas as pd
from numpy import argmax
from matplotlib import pyplot as plt
import sklearn 
import argparse
import os, sys

import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.utils import np_utils # this will make the label to one-hot encoding 
from keras.preprocessing.image import ImageDataGenerator


#data_set = pd.read_csv(sys.argv[1], encoding = "big5")
test_set = pd.read_csv(sys.argv[1], encoding = "big5")
test_temp = test_set.iloc[:,1].values 
TEST_SIZE = test_temp.shape[0]
#label = np.ndarray(shape = (SIZE,1))
#label[:,0] = data_set.iloc[:,0].values
test = np.ndarray(shape=(TEST_SIZE,48*48))

for i in range(TEST_SIZE):
    test[i] = np.fromstring(test_temp[i], dtype=int, sep=' ')

test = np.reshape(test, (TEST_SIZE,48,48,1))
#plt.imshow(train[10])
#plt.show()   #These are only for showing the images 

# Speed up the processing speed

test = test.astype('float32')
test /= 255

#Label = np_utils.to_categorical(label, 7)


model = load_model('good.h5')
sol_temp = model.predict(test)

sol = np.ndarray((TEST_SIZE,1))
sol = np.argmax(sol_temp, axis=1)
sol = sol.astype(int)



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



f = open(sys.argv[2],"w")
w = csv.writer(f)
for row in prediction_write:
	w.writerow(row)
f.close()





















