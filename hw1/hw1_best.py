import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

# read model
w = np.load('model.npy')

### read testing data
test_x = []
n_row = 0
x_i = -1
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 9:
        test_x.append([])
        x_i = x_i + 1
        for i in range(2,11):
            test_x[x_i].append(float(r[i]) )
    n_row = n_row+1

text.close()
test_x = np.array(test_x)

# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

### get ans.csv
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

# "result/predict.csv"
filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
