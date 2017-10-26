import os, sys
import pandas as pd
import numpy as np
from random import shuffle
import argparse
from math import log, floor

X_train = pd.read_csv(sys.argv[3], sep=',', header=0)
X_train = np.array(X_train.values)
Y_train = pd.read_csv(sys.argv[4], sep=',', header=0)
Y_train = np.array(Y_train.values)
X_test = pd.read_csv(sys.argv[5], sep=',', header=0)
X_test = np.array(X_test.values)

### Normalize
# Feature normalization with train and test X
X_all = X_train
X_train_test = np.concatenate((X_all, X_test))
mu = (sum(X_train_test) / X_train_test.shape[0])
sigma = np.std(X_train_test, axis=0)
mu = np.tile(mu, (X_train_test.shape[0], 1))
sigma = np.tile(sigma, (X_train_test.shape[0], 1))
X_train_test_normed = (X_train_test - mu) / sigma

percentage = 0.1
Y_all = Y_train
all_data_size = len(X_all)
valid_data_size = int(floor(all_data_size * percentage))
#X_all, Y_all = _shuffle(X_all, Y_all)
randomize = np.arange(len(X_all))
np.random.shuffle(randomize)
X_all = X_all[randomize]
Y_all = Y_all[randomize]

X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
# Gaussian distribution params
train_data_size = X_train.shape[0]
cnt1 = 0
cnt2 = 0

mu1 = np.zeros((106,))
mu2 = np.zeros((106,))
for i in range(train_data_size):
	if Y_train[i] == 1:
		mu1 += X_train[i]
		cnt1 += 1
	else:
		mu2 += X_train[i]
		cnt2 += 1
mu1 /= cnt1
mu2 /= cnt2

sigma1 = np.zeros((106,106))
sigma2 = np.zeros((106,106))
for i in range(train_data_size):
	if Y_train[i] == 1:
		sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
	else:
		sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
sigma1 /= cnt1
sigma2 /= cnt2
shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
N1 = cnt1
N2 = cnt2

print('=====Saving Param=====')
save_dir = "model"
if not os.path.exists(save_dir):
	os.mkdir(save_dir)
param_dict = {'mu1':mu1, 'mu2':mu2, 'shared_sigma':shared_sigma, 'N1':[N1], 'N2':[N2]}
for key in sorted(param_dict):
	print('Saving %s' % key)
	np.savetxt(os.path.join(save_dir, ('%s' % key)), param_dict[key])
    
print('=====Validating=====')
#valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)
sigma_inverse = np.linalg.inv(shared_sigma)
w = np.dot( (mu1-mu2), sigma_inverse)
x = X_valid.T
#x = np.transpose(X_valid)
b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
a = np.dot(w, x) + b
res = 1 / (1.0 + np.exp(-a))
y = np.clip(res, 1e-8, 1-(1e-8))
y_ = np.around(y)
result = (np.squeeze(Y_valid) == y_)
print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))

print('=====Loading Param from %s=====' % save_dir)
mu1 = np.loadtxt(os.path.join(save_dir, 'mu1'))
mu2 = np.loadtxt(os.path.join(save_dir, 'mu2'))
shared_sigma = np.loadtxt(os.path.join(save_dir, 'shared_sigma'))
N1 = np.loadtxt(os.path.join(save_dir, 'N1'))
N2 = np.loadtxt(os.path.join(save_dir, 'N2'))

# Predict
sigma_inverse = np.linalg.inv(shared_sigma)
w = np.dot( (mu1-mu2), sigma_inverse)
x = X_test.T
b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
a = np.dot(w, x) + b
res = 1 / (1.0 + np.exp(-a))
y = np.clip(res, 1e-8, 1-(1e-8))
y_ = np.around(y)

output_dir = "result"
print('=====Write output to %s =====' % output_dir)
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
output_path = os.path.join(output_dir, 'prediction.csv')
with open(output_path, 'w') as f:
	f.write('id,label\n')
	for i, v in  enumerate(y_):
		f.write('%d,%d\n' %(i+1, v))