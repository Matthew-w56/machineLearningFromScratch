# Author: Matthew Williams

import math
import numpy as np
from util import optimizers
import matplotlib.pyplot as plt


class LinearRegression:
	"""Machine Learning model that takes in predicting variables, and predicts values for data points"""
	
	def __init__(self, iterations=1000, headers=None,
				 default_prints=True, track_metrics=False, optimizer=optimizers.Adagrad(),
				 stochastic=False, batch_size=16, data_connects=False):
		"""Builds a Linear Regression model with multiple input variables


		Parameters

		-min_gradient: float.  Defines the minimum magnitude of gradient that will still be
		applied before stopping the training process

		-init_values: 1-dimensional, numerical array.  Initial value of the bias, and weights (bias = [0])

		-headers: array of len = 2, String.  Preferred names of the bias and features (bias = [0])
		default is ['b', 'x1', 'x2', .. 'xn']

		-default_prints: boolean.  Dictates whether the methods train() and evaluate() print their
		results by default.  This can be overridden in the method call directly

		-track_metrics: boolean.  Determines if the model calculates and stores things like an error
		history and gradient history, for later analysis.  Default is False, and setting to True
		increases run-time considerably (50% - 100%)
		
		-optimizer: object instance.  The object of the class of the desired optimizer for the model.
		See models.optimizers.py
		
		(Any other key-word arguments get passed on to the optimizer, who will ignore irrelevant arguments)
		"""

		# Initialize state variables ("None"s require later instantiation)
		self.weight_count = None
		self.var_count = None
		self.error = None
		self.default_prints = default_prints
		self.track_metrics = track_metrics
		self.headers = headers
		self.w = None
		self.optimizer = optimizer
		
		self.is1D = False
		
		if stochastic:
			self.update_weights = self.__update_weights_stochastic
		else:
			self.update_weights = self.__update_weights
		self.batch_size = batch_size

		# Metric state variables
		if track_metrics:
			self.gradient_hist = []		# History of model's past gradients, after S is applied
			self.error_hist = []		# History of model's error value each time it is calculated
			self.distances = []			# Distance that each guess is from dataset value (populated in evaluate())

		# Store Hyperparameters
		self.iterations = iterations
		
	def train(self, x, y, prints=None, realtime_chart=False):
		"""Trains the model to the given data by iterating gradient descent
		
		
		Parameters
		
		-x: 2-dimensional, numerical array of inputs of dimension [(data point count) x (feature count)]
		
		-y: 1-dimensional, numerical array of true values for inputs X of dimension [data point count]
		
		-prints: Boolean.  Dictates whether this model will print the results at the end of the
		training process.  If left as None, the model.default_prints is used
		"""
		
		# Verify the inputs -----------------------
		if len(x) == 0:
			raise Exception('Cannot have an empty dataset X!')
		if hasattr(x[0], '__iter__'):
			if len(x[0]) == 0:
				raise Exception('Data points cannot be empty in X!')
		else:
			self.is1D = True
		if len(x) != len(y):
			raise Exception(f'Data sets X and Y must be of equal length! ({len(x)} != {len(y)})')
		# Done verifying inputs ----------------------------
		
		# Initialize the feature-dependant variables for the model (Removes need for user to specify feature count)
		if self.weight_count is None:
			self.__initialize_state(1 if self.is1D else len(x[0]))
		
		# Go through gradient descent for the given iterations
		for _ in range(self.iterations):
			# Update the weights to be a little closer (gradient descent)
			self.update_weights(x, y)
			
		
		# Calculate the error for the model
		y_hat = self.predict_all(x)
		self.__calculate_error(y_hat, y)
		
		# Print results
		if (prints is not None and prints) or (prints is None and self.default_prints):
			print('\nTraining done!')
			print('Error:', self.error)
			print('Training set size:', len(y))
			if self.is1D:
				print(f'Final {self.headers[0]}: {self.w}')
			else:
				for i in range(self.var_count):
					print(f'Final {self.headers[i]}: {self.w[i]}')
	
	def evaluate(self, x, y, prints=None):
		"""Evaluates the performance of the model by predicting points in a dataset, and calculating error


		Parameters

		-x: 2-dimensional, numerical array of inputs of dimension [(data point count) x (feature count)]

		-y: 1-dimensional, numerical array of true values for inputs X of dimension [data point count]

		-prints: Boolean.  Dictates whether this model will print the results rather than just store error.
		If left as None, the model.default_prints is used
		"""

		# Verify inputs --------------------------------
		if len(x) == 0:
			print('Dataset X cannot be empty!')
			return
		if not self.is1D and len(x[0]) != self.weight_count:
			print('Given X set for evaluation contains a different number of features than training set!')
			return
		if len(x) != len(y):
			print(f'Datasets X and Y of different length! ({len(x)} != {len(y)})')
			return
		# Done verifying inputs -------------------------------

		# Build prediction list from inputs given
		y_hat = self.predict_all(x)

		# Calculate error
		self.__calculate_error(y_hat, y)

		# Populate distances metric list
		if self.track_metrics:
			self.distances = np.absolute(y_hat - y)

		# Print results
		if (prints is not None and prints) or (prints is None and self.default_prints):
			print('\nEvaluation Results')
			print('Error:', math.floor(self.error))
			print('Training set size:', len(y))
	
	def predict_all(self, x):
		"""Returns a list that contains one prediction for each data point (row) given

		Parameters

		-x: 2-dimensional, numerical array of inputs of dimension [(data point count) x (feature count)]
		"""

		# Build a list of predictions by applying the model weights to the input, and adding that to the bias
		# (This if statement is the only conditional check needed to turn this model into one that can also
		#   do simple linear regression if passed in only one explanatory variable)
		if self.weight_count == 1:
			return self.w[0] + np.dot(x, self.w[1])
		else:
			return self.w[0] + np.dot(x, self.w[1:])
	
	def __update_weights(self, x, y):
		"""Iterate over gradient descent once, including calculating the gradient and applying it"""
		
		# Reference variables for later
		features = 1 if not hasattr(x[0], '__iter__') else len(x[0])  # Number of features in dataset
		n = len(x)  # Number of data points in dataset
		
		# Predict all inputs with new weights
		y_hat = self.predict_all(x)
		
		# Create a list of the differences between y_hat and y
		diff = y_hat - y
		
		# Split the X matrix into feature columns
		cols = np.hsplit(x, features)
		# This turns [[1], [2], [3]] into [1, 2, 3]
		cols = np.array(cols).reshape((features, -1))
		
		# Calculate the gradient  (features is incremented to account for bias)
		gradient = np.zeros(features + 1)
		gradient[0] = np.sum(diff) / n
		for i in range(features):
			# Notice the i+1. This is parallel to the features+1 (accounts for bias)
			gradient[i + 1] = np.dot(cols[i], diff.T) / n
		
		# Apply the gradient in the style of the chosen optimizer (return only for magnitude calculation)
		self.optimizer.update(self.w, gradient)
		
		# Calculate the magnitude of the gradient
		g_magnitude = abs(np.sum(gradient * gradient))
		
		# If this model is tracking metrics, calculate error and record error and gradient
		if self.track_metrics:
			self.__calculate_error(y_hat, y)
			self.error_hist.append(self.error)
			self.gradient_hist.append(g_magnitude)

		# Return the gradient magnitude so that the train method can know when to stop going
		return g_magnitude
	
	def __update_weights_stochastic(self, x, y):
		"""Iterate over gradient descent once, including calculating the gradient and applying it"""
		
		# Reference variables for later
		features = 1 if not hasattr(x[0], '__iter__') else len(x[0])  # Number of features in dataset
		n = len(x)  # Number of data points in dataset
		
		# Initialize all the stochastic-related variables
		batch_indices = np.random.default_rng().choice(range(n), size=self.batch_size)
		new_x = x[batch_indices]
		new_y = y[batch_indices]
		new_y_hat = self.predict_all(new_x)
		
		# Create a list of the differences between y_hat and y
		diff = new_y_hat - new_y
		
		# Split the X matrix into feature columns
		cols = np.hsplit(new_x, features)
		# This turns [[1], [2], [3]] into [1, 2, 3]
		cols = np.array(cols).reshape((features, -1))
		
		# Calculate the gradient  (features is incremented to account for bias)
		gradient = np.zeros(features + 1)
		gradient[0] = np.sum(diff) / n
		# Loop through features, which is len(gradient)-1 (no bias accounted), and update i+1 (skip bias)
		for i in range(features):
			# Notice the i+1. This is parallel to the features+1 (accounts for bias)
			gradient[i+1] = np.dot(cols[i], diff.T) / n
		
		# Apply the gradient in the style of the chosen optimizer (return only for magnitude calculation)
		self.optimizer.update(self.w, gradient)
		
		# Calculate the magnitude of the gradient
		g_magnitude = abs(np.sum(gradient * gradient))
		
		# If this model is tracking metrics, calculate error and record error and gradient
		if self.track_metrics:
			self.__calculate_error(new_y_hat, new_y)
			self.error_hist.append(self.error)
			self.gradient_hist.append(g_magnitude)
		
		# Return the gradient magnitude so that the train method can know when to stop going
		return g_magnitude
	
	def __calculate_error(self, y_hat, y):
		"""Calculates the model's error by averaging each data point's error (y_hat - y) ^ 2"""

		# Square each difference and calculate half the average (half for derivative reasons)
		self.error = np.sum(np.square(y_hat - y)) / (2 * len(y))
	
	def __initialize_state(self, f_count):
		"""Initializes variables that depend on the feature count of the data passed in at train()"""
		
		# Initialize state variables with the number of features of X
		self.weight_count = f_count
		self.var_count = self.weight_count + 1
		self.w = np.array([0.0 for _ in range(self.var_count)])

		# Verify the headers (No change here = user-given header strings)
		if self.headers is None or len(self.headers) != self.var_count:
			# If no headers were passed in __init__, use defaults
			self.headers = ['x' + str(i) for i in range(1, self.var_count)]
			self.headers.insert(0, 'b')
	
	def to_json(self):
		
		# Set out all the straight forward properties of the json
		final_output = {
			"class_name": "LinearRegression",
			"weight_count": self.weight_count,
			"var_count": self.var_count,
			"error": self.error,
			"default_prints": self.default_prints,
			"track_metrics": self.track_metrics,
			"headers": self.headers,
			"w": self.w.tolist(),
			"optimizer": self.optimizer.to_json(),
			"is1D": self.is1D,
			"stochastic": 1 if self.update_weights == self.__update_weights_stochastic else 0,
			"batch_size": self.batch_size,
			"iterations": self.iterations
		}
		
		# Add in any optional fields
		if self.track_metrics:
			final_output['gradient_hist'] = self.gradient_hist
			final_output['error_hist'] = self.error_hist
			final_output['distances'] = self.distances
		
		return final_output


def build_from_json(json_obj):
	if json_obj['class_name'] != "LinearRegression":
		print("LinearRegression.build_from_json error: Model JSON not a LinearRegression!  Found",
			  json_obj['class_name'])
	lrm = LinearRegression(
			iterations=json_obj['iterations'],
			headers=json_obj['headers'],
			default_prints=json_obj['default_prints'],
			track_metrics=json_obj['track_metrics'],
			optimizer=optimizers.get_optimizer_from_json(json_obj['optimizer']),
			stochastic=(True if json_obj['stochastic'] == 1 else False),
			batch_size=json_obj['batch_size']
	)
	lrm.is1D = json_obj['is1D']
	lrm.weight_count = json_obj['weight_count']
	lrm.var_count = json_obj['var_count']
	lrm.w = np.array(json_obj['w'])
	lrm.error = json_obj['error']
	
	tM = json_obj['track_metrics']
	if type(tM) == bool and tM:
		lrm.gradient_hist = json_obj['gradient_hist']
		lrm.error_hist = json_obj['error_hist']
		lrm.distances = json_obj['distances']
	
	return lrm
	
	
