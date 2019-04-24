import numpy as np
import pickle
import time
import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from util.prepare_data import f

rng = np.random.RandomState(42)

def get_hyp(arr):
	hyperparams = {}
	hyperparams['C'] = {'value': float(arr[0]), 'description': 'None', 'type': 'continuous', 'range':[0.1, 5]}
	hyperparams['intercept_scaling'] = {'value': float(arr[1]), 'description': 'None', 'type': 'continuous', 'range':[0.01, 2]}
	hyperparams['metainfo'] = ['C', 'intercept_scaling']
	return hyperparams

def prepare_D_surrogate(num_points):
	C_vals = np.linspace(0.1, 5, num_points)
	int_scl_vals = np.linspace(0.5, 2, num_points)
	zeta_set = []
	eta_set = []

	for i in tqdm.trange(len(C_vals)):
		C = C_vals[i]
		int_scl = int_scl_vals[i]
		zeta_set.append([C, int_scl])
		hyperparams = get_hyp([C, int_scl])
		task = Task(hyperparams)
		eta_set.append(task.metric())

	zeta_set = np.reshape(np.array(zeta_set), (num_points, 2))
	eta_set = np.array(eta_set)
	return zeta_set, eta_set

def load_data(filepath):
	with open(filepath, 'rb') as f:
		a = pickle.load(f)
	return a

class Task(object):
	"""docstring for Task"""
	def __init__(self, hyperparams):
		super(Task, self).__init__()
		self.hyperparams = hyperparams
		self.C = self.hyperparams['C']['value']
		self.intercept_scaling = self.hyperparams['intercept_scaling']['value']

		(self.X_train, self.y_train), (self.X_cval, self.y_cval), (self.X_test, self.y_test) = load_data("../data/mnist_preprocessed.pkl")

		self.metainfo = self.hyperparams['metainfo']
		print("Hyperparameters to be optimized: ", " ,".join(self.metainfo))
		
		self.classifier = LogisticRegression(solver='liblinear', intercept_scaling=self.intercept_scaling, C=self.C, multi_class='ovr')

	def train(self):
		print("Training LogisticRegression model...")
		start = time.time()
		self.classifier.fit(self.X_train, self.y_train)
		end = time.time()
		print("Total training time: ", end-start)
		return

	def metric(self):
		self.train()

		y_pred = self.classifier.predict(self.X_cval)
		y_true = self.y_cval

		return accuracy_score(y_true, y_pred)

	def test(self):
		y_pred = self.classifier.predict(self.X_test)
		y_true = self.y_test

		return accuracy_score(y_true, y_pred)
	
