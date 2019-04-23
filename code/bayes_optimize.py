import numpy as np
import scipy.stats as ss
import os
import tqdm

from util.plotting import visualize_predictions, visualize_EI

rng = np.random.RandomState(42)

class BayesOptimize(object):
	"""docstring for BayesOptimize"""
	def __init__(self, D_task, D_surrogate, task_oracle_file, dngo_model, num_epochs, zeta_test):
		super(BayesOptimize, self).__init__()
		
		# (self.X_train, self.y_train), (self.X_cval, self.y_cval), (self.X_test, self.y_test) = D_task
		self.D_surrogate = D_surrogate
		self.zeta_train, self.eta_train = self.D_surrogate
		self.task_oracle_file = task_oracle_file
		self.dngo_model = dngo_model
		self.num_epochs = num_epochs
		self.zeta_test = zeta_test
		self.num_init_points = D_surrogate[0].shape[0]
		self.plot_directory = "../plots/"+str(self.num_init_points)+"-plots"
	
	def posterior_prediction(self, zeta_test):
		mean, var = self.dngo_model.predict(zeta_test)
		return mean, var
	
	def expected_improvement(self, zeta_test):
		zeta_train, eta_train = self.D_surrogate
		eta_curr_best = np.max(eta_train)
		
		mean, var = self.posterior_prediction(zeta_test)
		gamma = (eta_curr_best - mean)/var
		
		std_gaussian = ss.norm(0.0, 1.0)
		
		a = [std_gaussian.cdf(x) for x in gamma]
		b = [std_gaussian.pdf(x) for x in gamma]
		
		return np.multiply(var, np.multiply(gamma, a) + b)
	
	def find_zeta_new(self, zeta_test):
		ei = self.expected_improvement(zeta_test)
		idx = np.argmax(ei)
		
		return zeta_test[idx]

	def optimal_hyperparameters(self, save_plots=False, D_task=None, task_name=''):
		zeta_test = self.zeta_test
		eta_news = []

		if D_task != None:
			(X_train, y_train), (X_cval, y_cval), (X_test, y_test) = D_task


		for i in tqdm.trange(1, self.num_epochs+1):
			self.dngo_model.train(self.zeta_train, self.eta_train, do_optimize=True)

			if save_plots:
				mean, var = self.dngo_model.predict(zeta_test)
				EI = self.expected_improvement(zeta_test)
				name = str(i)+".png"
				if not os.path.exists(self.plot_directory):
					os.system("mkdir "+ self.plot_directory)
				else:
					path = os.path.join(self.plot_directory, name)
					a = visualize_EI(self.zeta_train, self.eta_train, zeta_test, mean, var, np.array(EI), save=True, path=path)


			zeta_new = self.find_zeta_new(zeta_test)
			hyp_new = self.task_oracle_file.get_hyp(list(zeta_new))

			task_1 = self.task_oracle_file.Task(hyperparams=hyp_new)
			if task_name == 'BLR':
				eta_new = task_1.metric(X_cval, y_cval, X_train, y_train)
				eta_news.append(eta_new)
			else:
				eta_new = task_1.metric()

			self.zeta_train = np.concatenate((self.zeta_train, [zeta_new]), axis=0)
			self.eta_train = np.concatenate((self.eta_train, [eta_new]), axis=0)

			continue

		idx = np.argmax(self.eta_train)
		return self.zeta_train[idx]  