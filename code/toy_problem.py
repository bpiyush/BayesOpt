import numpy as np
import matplotlib.pyplot as plt

from util.prepare_data import f

rng = np.random.RandomState(42)

def get_hyp(arr):
	hyperparams = {}
	hyperparams['x'] = {'value': float(arr[0]), 'description': 'None', 'type': 'continuous', 'range':[-4, 4]}
	hyperparams['metainfo'] = ['x']
	return hyperparams

def prepare_D_surrogate(num_points):
	x_arr = rng.rand(num_points)
	# x_arr = np.linspace(0, 1, num_points)
	y = []
	for x in x_arr:
		hyperparams = {}
		hyperparams['x'] = {'value': x, 'description': 'None', 'type': 'continuous', 'range':[-4, 4]}
		hyperparams['metainfo'] = ['x']
		task = Task(hyperparams)
		y.append(task.metric())
	return x_arr, y

class Task(object):
	"""docstring for Task"""
	def __init__(self, hyperparams):
		super(Task, self).__init__()
		self.hyperparams = hyperparams
		
		self.metainfo = self.hyperparams['metainfo']
		print("Hyperparameters to be optimized: ", " ,".join(self.metainfo))
		
		self.x_info = self.hyperparams['x']
		self.x = self.x_info['value']

	def metric(self):
		return f(self.x)
	
