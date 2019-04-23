import time
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

from util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from util.plotting import loss_plot


class NeuralNetwork(nn.Module):
	"""
	Description of neural network architechture used 
	as a surrogate model for Bayesian Optimization.
	"""
	def __init__(self, n_inputs, h_units=[50, 50, 50]):
		super(NeuralNetwork, self).__init__()
		self.fc1 = nn.Linear(n_inputs, h_units[0])
		self.fc2 = nn.Linear(h_units[0], h_units[1])
		self.fc3 = nn.Linear(h_units[1], h_units[2])
		self.out = nn.Linear(h_units[2], 1)

	def forward(self, x):
		x = torch.tanh(self.fc1(x))
		x = torch.tanh(self.fc2(x))
		x = torch.tanh(self.fc3(x))
		return self.out(x)

	def basis_functions(self, x):
		x = torch.tanh(self.fc1(x))
		x = torch.tanh(self.fc2(x))
		x = torch.tanh(self.fc3(x))
		return x

class DNGO_v1():
	"""docstring for DNGO_v1: Will fill in later"""
	def __init__(self, batch_size=10, num_epochs=100,
				 learning_rate=0.01, adapt_epoch=5000, h_units_1=50, h_units_2=50, h_units_3=50,
				 alpha=1.0, beta=1000, normalize_input=True, normalize_output=True, rand_num_gen=None):
		super(DNGO_v1, self).__init__()

		if rand_num_gen is None:
			self.rand_num_gen = np.random.RandomState(np.random.randint(0, 10000))
		else:
			self.rand_num_gen = rand_num_gen

		# General settings
		self.X = None
		self.y = None
		self.alpha = alpha
		self.beta = beta
		self.normalize_input = normalize_input
		self.normalize_output = normalize_output

		# Network hyper parameters
		self.network = None
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.init_learning_rate = learning_rate

		self.h_units_1 = h_units_1
		self.h_units_2 = h_units_2
		self.h_units_3 = h_units_3
		self.adapt_epoch = adapt_epoch
		self.models = []
		self.hypers = None

	def check_train_sanity(self, X, y):
		assert X.shape[0] == len(y), "Error: Number of training inputs do not match number of training labels."
		assert len(X.shape) == 2, "Error: Incorrect input shape."
		assert len(y.shape) == 1, "Error: Training labels must be one-dimensional array."

	def sample_minibatch(self, X, y, batch_size):
		indices = np.arange(X.shape[0])
		self.rand_num_gen.shuffle(indices)

		batch_idx = indices[:batch_size]
		return X[batch_idx, :], y[batch_idx]

	def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
		if shuffle:
			indices = np.arange(inputs.shape[0])
			self.rand_num_gen.shuffle(indices)
		for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
			if shuffle:
				excerpt = indices[start_idx:start_idx + batchsize]
			else:
				excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


	def train(self, X, y, do_optimize=True):
		self.check_train_sanity(X, y)

		start_time = time.time()

		# Normalize inputs
		if self.normalize_input:
			self.X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
		else:
			self.X = X

		# Normalize ouputs
		if self.normalize_output:
			self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)
		else:
			self.y = y

		self.y = self.y[:, None]

		# Check if we have enough points to create a minibatch otherwise use all data points
		if self.X.shape[0] <= self.batch_size:
			batch_size = self.X.shape[0]
		else:
			batch_size = self.batch_size


		num_features = X.shape[1]
		self.network = NeuralNetwork(n_inputs=num_features, h_units=[self.h_units_1, self.h_units_2, self.h_units_3])

		optimizer = optim.Adam(self.network.parameters(), lr=self.init_learning_rate)


		# Start training
		lc = np.zeros([self.num_epochs])
		for epoch in tqdm.trange(self.num_epochs):

			epoch_start_time = time.time()

			train_err = 0
			train_batches = 0

			for batch in self.iterate_minibatches(self.X, self.y, batch_size, shuffle=True):
				inputs = torch.Tensor(batch[0])
				targets = torch.Tensor(batch[1])

				optimizer.zero_grad()
				output = self.network(inputs)
				loss = torch.nn.functional.mse_loss(output, targets)
				loss.backward()
				optimizer.step()

				train_err += loss
				train_batches += 1

			lc[epoch] = train_err / train_batches
			curtime = time.time()
			epoch_time = curtime - epoch_start_time
		total_time = curtime - start_time

		loss_arr = []
		for i in range(self.num_epochs):
			if i % 10 == 0:
				loss_arr.append(lc[i])

		# Plotting training loss
		loss_plot(len(loss_arr), loss_arr)

		
		# Design matrix
		self.Theta = self.network.basis_functions(torch.Tensor(self.X)).data.numpy()

		self.Theta = np.reshape(self.Theta, (X.shape[0], self.h_units_3))

		return self.Theta, self.network

	def K_train(self, Theta, a, b):
		return b*np.dot(Theta.T, Theta) + a*np.eye(Theta.shape[1])

	def m_train(self, Theta, y, K, a, b):
		return b*np.dot(np.dot(np.linalg.inv(K),Theta.T), y)

	def prior(self, x_test, Gamma, c, lambda_):
		return lambda_ + np.dot((x_test-c).T, np.dot(Gamma, x_test-c))

	def predict(self, x_test, nn_model, X_train, y_train, Theta, alpha, beta, Gamma, c, lambda_):
		self.eta_x_train = [self.prior(x_train, Gamma, c, lambda_) for x_train in X_train]
		self.eta_x_test = self.prior(x_test, Gamma, c, lambda_)
		self.K = beta*np.dot(Theta.T , Theta) + alpha*np.eye(Theta.shape[1])
		self.m = beta*np.dot(np.dot(np.linalg.inv(self.K), Theta.T), y_train - self.eta_x_train)
		# self.m = beta*np.dot(np.dot(np.linalg.inv(self.K), Theta.T), y_train )
		# self.m = beta*np.dot(np.dot(np.linalg.inv(self.K), Theta.T), y_train -self.eta_x_test)



		self.phi_x_test = nn_model.basis_functions(torch.Tensor(x_test)).data.numpy()

		self.mu = np.dot(self.m.T, self.phi_x_test) + self.eta_x_test
		# self.mu = np.dot(self.m.T, self.phi_x_test) 

		self.var = np.dot(self.phi_x_test.T, np.dot(np.linalg.inv(self.K), self.phi_x_test)) + np.divide(1.0, beta)

		return self.mu, self.var
		

	def marginal_log_likelihood(self, Theta, X_train, y_train, alpha, beta, Gamma, c, lambda_):
		self.D = Theta.shape[0]
		self.N = X_train.shape[0]
		self.eta_x_train = [self.prior(x_train, Gamma, c, lambda_) for x_train in X_train]
		self.y_train_tilda = y_train - self.eta_x_train
		self.K = beta*np.dot(Theta.T , Theta) + alpha*np.eye(Theta.shape[1])
		self.m = beta*np.dot(np.dot(np.linalg.inv(self.K), Theta.T), y_train - self.eta_x_train)

		self.val = ((self.D)/2.0)*np.log(alpha) + ((self.N)/2.0)*np.log(beta) 
		self.val -= (beta/2.0)*np.linalg.norm(self.y_train_tilda - np.dot(Theta, self.m), ord=2)**2
		self.val -= (alpha/2.0)*np.linalg.norm(self.m, ord=2)
		self.val -= 0.5*np.log(np.linalg.det(self.K))

		return self.val


	def objective_alpha(self, alpha, *other_params):
		self.Theta, self.X_train, self.y_train, self.beta, self.Gamma, self.c, self.lambda_ = other_params
		mll_alpha = self.marginal_log_likelihood(self.Theta, self.X_train, self.y_train, alpha, 
			self.beta, self.Gamma, self.c, self.lambda_)

		return -1*mll_alpha

	def objective_beta(self, beta, *other_params):
		self.Theta, self.X_train, self.y_train, self.alpha, self.Gamma, self.c, self.lambda_ = other_params
		mll_beta = self.marginal_log_likelihood(self.Theta, self.X_train, self.y_train, self.alpha, 
			beta, self.Gamma, self.c, self.lambda_)

		return -1*mll_beta

	def objective_gamma(self, gamma, *other_params):
		self.Gamma = np.diag(gamma)
		self.Theta, self.X_train, self.y_train, self.alpha, self.beta, self.c, self.lambda_ = other_params
		mll_gamma = self.marginal_log_likelihood(self.Theta, self.X_train, self.y_train, self.alpha, 
			self.beta, self.Gamma, self.c, self.lambda_)

		return -1*mll_gamma

	def objective_c(self, c, *other_params):
		self.Theta, self.X_train, self.y_train, self.alpha, self.beta, self.Gamma,  self.lambda_ = other_params
		mll_c = self.marginal_log_likelihood(self.Theta, self.X_train, self.y_train, self.alpha, 
			self.beta, self.Gamma, c, self.lambda_)

		return -1*mll_c

	def objective_lambda_(self, lambda_, *other_params):
		self.Theta, self.X_train, self.y_train, self.alpha, self.beta, self.Gamma, self.c = other_params
		mll_lambda_ = self.marginal_log_likelihood(self.Theta, self.X_train, self.y_train, self.alpha, 
			self.beta, self.Gamma, self.c, lambda_)

		return -1*mll_lambda_

	def alternating_optimization(self, init_params):
		self.Theta, self.X_train, self.y_train, self.alpha, self.beta, self.Gamma, self.c, self.lambda_ = init_params
		self.gamma = np.diagonal(self.Gamma)
		self.num_itns = 2

		for i in tqdm.trange(self.num_itns):
			# Optimize alpha keeping others constant
			self.other_params = self.Theta, self.X_train, self.y_train, self.beta, self.Gamma, self.c, self.lambda_
			result = minimize(self.objective_alpha, self.alpha, args=self.other_params)
			self.alpha = result.x

			# Optimize beta keeping others constant
			self.other_params = self.Theta, self.X_train, self.y_train, self.alpha, self.Gamma, self.c, self.lambda_
			result = minimize(self.objective_beta, self.beta, args=self.other_params)
			self.beta = result.x

			# Optimize gamma keeping others constant
			self.other_params = self.Theta, self.X_train, self.y_train, self.alpha, self.beta, self.c, self.lambda_
			result = minimize(self.objective_gamma, self.gamma, args=self.other_params)
			self.gamma = result.x
			self.Gamma = np.diag(self.gamma)

			# Optimize c keeping others constant
			self.other_params = self.Theta, self.X_train, self.y_train, self.alpha, self.beta, self.Gamma, self.lambda_
			result = minimize(self.objective_c, self.c, args=self.other_params)
			self.c = result.x

			# Optimize lambda_ keeping others constant
			self.other_params = self.Theta, self.X_train, self.y_train, self.alpha, self.beta, self.Gamma, self.c
			result = minimize(self.objective_lambda_, self.lambda_, args=self.other_params)
			self.lambda_ = result.x

		self.optimal_params = self.alpha, self.beta, self.Gamma, self.c, self.lambda_
		return self.optimal_params


		









		


