import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

data = []
for i in range(18):
    data.append([])

r = 0
text = open('data/train.csv', 'r') #, encoding='big5')
for row in csv.reader(text):
    if r != 0:
        data_i = (r-1)%18
        for i in range(3,27):
            if row[i] != "NR":
                data[data_i].append(float(row[i]))
            else:
                data[data_i].append(float(0))
    r = r+1
text.close()

x = []
y = []

for i in range(12):
    # 471 data per month
    for j in range(471):
        x.append([])
        y.append(data[9][480*i+j+9])
        for s in range(9):
            x[471*i+j].append(data[9][480*i+j+s] )
x = np.array(x)
y = np.array(y)

# add square term
x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

### init weight & other hyperparams
w = np.zeros(len(x[0]))
l_rate = 15
repeat = 50000

### start training
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

# save model
np.save('model.npy',w)
# read model
#w = np.load('model.npy')
