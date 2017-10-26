import numpy as np
from numpy.linalg import inv
from random import shuffle
from math import log, floor
import os, sys
import pandas as pd
import argparse

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

# Split to train, test again
X_all = X_train_test_normed[0:X_all.shape[0]]
X_test = X_train_test_normed[X_all.shape[0]:]

### train
# Split a 10%-validation set from the training set
percentage = 0.1
#X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
Y_all = Y_train
all_data_size = len(X_all)
valid_data_size = int(floor(all_data_size * percentage))

#X_all, Y_all = _shuffle(X_all, Y_all)
randomize = np.arange(len(X_all))
np.random.shuffle(randomize)
X_all = X_all[randomize]
Y_all = Y_all[randomize]

X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

#print(X_valid[0])

# Initiallize parameter, hyperparameter
w = np.zeros((106,))
b = np.zeros((1,))
l_rate = 0.1
batch_size = 32
train_data_size = len(X_train)
step_num = int(floor(train_data_size / batch_size))
epoch_num = 1000
save_param_iter = 50

# Start training
total_loss = 0.0
for epoch in range(1, epoch_num):
	# Do validation and parameter saving
	if (epoch) % save_param_iter == 0:
		print('=====Saving Param at epoch %d=====' % epoch)
		save_dir = "model"
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		np.savetxt(os.path.join(save_dir, 'w'), w)
		np.savetxt(os.path.join(save_dir, 'b'), [b,])
		print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
		total_loss = 0.0
#		valid(w, b, X_valid, Y_valid)
		valid_data_size = len(X_valid)

		z = (np.dot(X_valid, np.transpose(w)) + b)
		res = 1 / (1.0 + np.exp(-z))
		y = np.clip(res, 1e-8, 1-(1e-8))
		#y = sigmoid(z)
		y_ = np.around(y)
		result = (np.squeeze(Y_valid) == y_)
		print('Validation acc = %f' % (float(result.sum()) / valid_data_size))

    # Random shuffle
	#X_train, Y_train = _shuffle(X_train, Y_train)
	randomize = np.arange(len(X_train))
	np.random.shuffle(randomize)
	X_train = X_train[randomize]
	Y_train = Y_train[randomize]

    # Train with batch
	for idx in range(step_num):
		X = X_train[idx*batch_size:(idx+1)*batch_size]
		Y = Y_train[idx*batch_size:(idx+1)*batch_size]

		z = np.dot(X, np.transpose(w)) + b
		res = 1 / (1.0 + np.exp(-z))
		y = np.clip(res, 1e-8, 1-(1e-8))
		#y = sigmoid(z)

		cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
		total_loss += cross_entropy

		w_grad = np.sum(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)
		b_grad = np.sum(-1 * (np.squeeze(Y) - y))

        # SGD updating parameters
		w = w - l_rate * w_grad
		b = b - l_rate * b_grad

### infer
test_data_size = len(X_test)

# Load parameters
print('=====Loading Param from %s=====' % save_dir)
w = np.loadtxt(os.path.join(save_dir, 'w'))
b = np.loadtxt(os.path.join(save_dir, 'b'))

# predict
z = (np.dot(X_test, np.transpose(w)) + b)
res = 1 / (1.0 + np.exp(-z))
y = np.clip(res, 1e-8, 1-(1e-8))
#y = sigmoid(z)
y_ = np.around(y)

output_dir = "result"
print('=====Write output to %s =====' % output_dir)
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
output_path = os.path.join(output_dir, 'log_prediction.csv')
with open(output_path, 'w') as f:
	f.write('id,label\n')
	for i, v in  enumerate(y_):
		f.write('%d,%d\n' %(i+1, v))
