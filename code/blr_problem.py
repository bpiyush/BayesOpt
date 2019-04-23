import numpy as np
import matplotlib.pyplot as plt

def get_hyp(arr):
	hyperparams = {}
	hyperparams['K'] = {'value': int(arr[0]), 'description': 'Kernel Dimension', 'type': 'discrete', 'range':[-4, 4]}
	hyperparams['beta'] = {'value': float(arr[1]), 'description': 'BLR Hyperparameter', 'type': 'continuous', 'range':[-4, 4]}
	hyperparams['lambda'] = {'value': float(arr[2]), 'description': 'BLR Hyperparameter', 'type': 'continuous', 'range':[-1, 1]}
	hyperparams['metainfo'] = ['K', 'beta', 'lambda']
	return hyperparams

def load_data():
	X = np.array([-2.23, -1.30, -0.42, 0.30, 0.33, 0.52, 0.87, 1.80, 2.74, 3.62])
	Y = np.array([1.01, 0.69, -0.66, -1.34, -1.75, -0.98, 0.25, 1.57, 1.65, 1.51])

	X_train, Y_train= np.array([-2.23, -0.42, 0.30, 0.87, 1.80, 3.62]),np.array([1.01, -0.66, -1.34, 0.25, 1.57, 1.51])
	X_cval, Y_cval = np.array([-1.30, 2.74]), np.array([0.69, 1.65])
	X_test, Y_test = np.array([0.33, 0.52]), np.array([-1.75, -0.98])

	return (X_train, Y_train), (X_cval, Y_cval), (X_test, Y_test)

def prepare_D_surrogate(num_examples=None):
	(X_train, y_train), (X_cval, y_cval), (X_test, y_test) = load_data()

	zeta_train = [[2, 1.0, 1.0], [4, 1.0, 1.0], [2, 4.0, 1.0], [3, 4.0, 1.0], 
				  [4, 4.0, 1.0], [5, 4.0, 1.0], [4, 2.0, 3.0], [5, 10.0, 2.0]]
	eta_train = []

	for x in zeta_train:
		task = Task(hyperparams=get_hyp(x))
		eta_train.append(task.metric(X_cval, y_cval, X_train, y_train))

	zeta_train = np.array(zeta_train)
	zeta_train = zeta_train.reshape((len(zeta_train), 3))
	eta_train = np.array(eta_train)

	return zeta_train, eta_train


class Task(object):
	"""docstring for Task"""
	def __init__(self, hyperparams):
		super(Task, self).__init__()
		self.hyperparams = hyperparams
		
		self.metainfo = self.hyperparams['metainfo']
		print("Hyperparameters to be optimized: ", " ,".join(self.metainfo))
		
		self.K_info = self.hyperparams['K']
		self.K = self.K_info['value']
		self.beta_info = self.hyperparams['beta']
		self.beta = self.beta_info['value']
		self.lambd_info = self.hyperparams['lambda']
		self.lambd = self.lambd_info['value']
	
	def transform_input(self, X):
		k = self.K
		# X = {x_{1}, .., x_{N}}
		X_new = np.array([np.power(X, i) for i in range(k+1)])
		X_new = X_new.T
		return X_new
		
	def posterior(self, phi_X, y):
		K, beta, lambda_ =self.K, self.beta, self.lambd
		# X = {x_{1}, .., x_{N}}
		D = phi_X.shape[1]
		N = phi_X.shape[0]
		mu = np.dot(np.dot(np.linalg.inv(np.dot(phi_X.T,phi_X) + (lambda_/beta)*np.eye(D)),(phi_X.T)),y)
		Sigma = np.linalg.inv(beta*np.dot(phi_X.T,phi_X) + lambda_*np.eye(D))

		return mu, Sigma
	
	def marginal_likelihood(self, X, y):
		K, beta, lambda_ =self.K, self.beta, self.lambd
		N = X.shape[0]
		D = X.shape[1]
		v = np.power(1/(2*(np.pi)), N/2)
		mat = (1/beta)*np.eye(N) + (1/lambda_)*np.dot(X, X.T)
		det = np.linalg.det(mat)
		det = np.power(det, -(1.0/2))
		val = (-1.0/2)*(np.dot(y.T, np.dot(np.linalg.inv(mat), y)))
		q = np.exp()

		return np.log(v*det) + val
	
	def posterior_predictive(self, X_cval, X, y):
		k, beta, lambda_ =self.K, self.beta, self.lambd
		
		phi_X = self.transform_input(X)
		mu, Sigma = self.posterior(phi_X, y)
		
		phi_X_cval = self.transform_input(X_cval)
		posterior_mean = np.dot(phi_X_cval, mu)
		posterior_var = (1/beta) + np.array([np.dot(phi_X_cval[i,:].T, np.dot(Sigma, phi_X_cval[i,:])) for i in range(phi_X_cval.shape[0])])

		return posterior_mean, posterior_var
		
	def metric(self, X_cval, y_cval, X, y):
		post_mu, post_var = self.posterior_predictive(X_cval, X, y)
		return np.linalg.norm(y_cval - post_mu) + 1.0/np.average(post_var)
		