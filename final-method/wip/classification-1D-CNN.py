from numpy import mean
from numpy import std
from numpy import dstack
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
import scipy.io as sio
import os
from os import listdir
from os.path import dirname, join as pjoin
import pathlib
import matplotlib.pyplot as plt
from numpy import newaxis

def getData(data_dir):
	AllfileNames = np.array((os.listdir(data_dir)))
	fileNames = np.array(list(filter(lambda x : x.startswith('detected'), AllfileNames)))
	print('----- Carregando arquivos .mat...')
	mat = [sio.loadmat(pjoin(data_dir, name)) for name in fileNames]
	print('----- Arquivos .mat carregados!')
	mat = np.array(mat)
	voltages = np.array([voltage['V'][:256,:] for voltage in mat])
	print(voltages.shape)
	nPoints = 256
	
	voltagesA = np.array([voltage[:nPoints, 0] for voltage in voltages[:]])
	voltagesB = np.array([voltage[:nPoints, 1] for voltage in voltages[:]])
	voltagesC = np.array([voltage[:nPoints, 2] for voltage in voltages[:]])

	voltagesZero = voltagesA + voltagesB + voltagesC
	
	for i in range(voltages.shape[0]):
		for j in range(voltages.shape[2]):
			max_ = np.max(voltages[i,:,j]) 
			min_ = np.min(voltages[i,:,j]) 
			voltages[i,:,j] = (voltages[i,:,j] - min_)/(max_ - min_)


	voltagesZero = voltagesZero[:, :, newaxis]
	for i in range(voltagesZero.shape[0]):
		for j in range(voltagesZero.shape[2]):
			max_ = np.max(voltagesZero[i,:,j]) 
			min_ = np.min(voltagesZero[i,:,j]) 
			voltagesZero[i,:,j] = (voltagesZero[i,:,j] - min_)/(max_ - min_)
	print(voltagesZero.shape)
	X = np.zeros((voltages.shape[0], voltages.shape[1], 4))
	X[:,:,:3] = voltages
	X[:,:,3] = voltagesZero[:,:,0]
	
	plt.plot(X[10,:,0])
	plt.plot(X[10,:,1])
	plt.plot(X[10,:,2])
	plt.plot(X[10,:,3])
	plt.show()
	targets = np.array([data['faultType'][0] for data in mat])
	
	from sklearn.preprocessing import LabelEncoder
	label_encoder = LabelEncoder()
	label_encoder.fit(targets)
	integer_encoded = label_encoder.fit_transform(targets)
	
	from sklearn.preprocessing import OneHotEncoder
	onehot_encoder = OneHotEncoder(sparse=False)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded.reshape(len(integer_encoded), 1))
	print(onehot_encoded.shape)
	
	print(X.shape)
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, onehot_encoded, test_size = 0.1)
	
	return X_train, y_train, X_test, y_test
 
# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 1, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(100, activation='tanh'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
	return accuracy
 
# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=10):
	# load data
	data_dir = pjoin(pathlib.Path().parent / 'detected-signals')
	trainX, trainy, testX, testy = 	getData(data_dir)
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
 
# run the experiment
run_experiment()