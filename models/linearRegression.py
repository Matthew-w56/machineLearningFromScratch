import numpy as np
import math


class SimpleLinearRegression:
	"""Machine Learning model that takes in a predicting variable, and predicts values for data points"""

	def __init__(self, min_gradient=0.0001, epsilon=0.00001, init_m=0, init_b=0, headers=None,
					default_prints=True, track_metrics=False):
		"""Builds a Linear Regression model with one input variable


		Parameters

		-min_gradient: float.  Defines the minimum magnitude of gradient that will still be
		applied before stopping the training process

		-epsilon: float.  Value added to S before square rooting it during the gradient application
		(Adagrad technique)

		-init_m: numerical.  Initial value of the weight, default is zero

		-init_b: numerical.  Initial value of the bias, default is zero

		-headers: array of len = 2, String.  Preferred names of the weight and bias respectively
		default is ['m', 'b']

		-default_prints: boolean.  Dictates whether the methods train() and evaluate() print their
		results by default.  This can be overridden in the method call directly

		-track_metrics: boolean.  Determines if the model calculates and stores things like an error
		history and gradient history, for later analysis.  Default is False, and setting to True
		increases run-time considerably (50% - 100%)
		"""

		# Instantiate headers variable
		if headers is None:
			# If none given, use defaults
			self.headers = ['m', 'b']
		else:
			# If one was given, check if it's a valid length
			if len(headers) == 2:
				# If so, use the given header names
				self.headers = headers
			else:
				# If headers given are invalid, go back to defaults
				print('Header list must be 2!')
				self.headers = ['m', 'b']

		# Initialize state variables
		self.m = init_m
		self.b = init_b
		self.s = np.array([0.0, 0.0])
		self.error = None
		self.default_prints = default_prints
		self.track_metrics = track_metrics

		# Metric state variables
		if track_metrics:
			self.gradient_hist = []		# History of model's past gradients, after learn_rate is applied
			self.error_hist = []		# History of model's error value each time it is calculated

		# Store Hyperparameters
		self.epsilon = epsilon
		self.min_gradient = min_gradient

	def train(self, x, y, prints=None):
		"""Trains the model to the given data by iterating gradient descent


		Parameters

		-x: 1-dimensional array of inputs (numerical)

		-y: 1-dimensional array of labels (numerical)

		-prints: Boolean.  Dictates whether this model will print the results at the end of the
		training process.  If left as None, the model.default_prints is used
		"""

		# Go through gradient descent until gradient magnitude is smaller than min_gradient
		while True:
			# Predict all inputs with new weights
			y_hat = self.predict_all(x)
			# Update the weights to be a little closer (gradient descent) (Returns gradient magnitude)
			g = self.__update_weights(x, y, y_hat)

			# If the minimum gradient is achieved, stop iterating gradient descent
			if g <= self.min_gradient:
				break

		# Calculate the error for the model
		self.__calculate_error(y_hat, y)

		# Print results
		if (prints is not None and prints) or (prints is None and self.default_prints):
			print('\nTraining done!')
			print('Error:', self.error)
			print('Training set size:', len(y))
			print('Final ' + self.headers[0] + ':', self.m)
			print('Final ' + self.headers[1] + ':', self.b)

	def evaluate(self, x, y, prints=None):
		"""Evaluates the performance of the model by predicting points in a dataset, and calculating error


		Parameters

		-x: 1-dimensional array of inputs (numerical)

		-y: 1-dimensional array of labels (numerical)

		-prints: Boolean.  Dictates whether this model will print the results at the end of the
		training process.  If left as None, the model.default_prints is used
		"""

		# Build prediction list from inputs given
		y_hat = self.predict_all(x)

		# Calculate error
		self.__calculate_error(y_hat, y)

		# Print results
		if (prints is not None and prints) or (prints is None and self.default_prints):
			print('\nEvaluation Results')
			print('Error:', self.error)
			print('Training set size:', len(y))

	def predict_all(self, x):
		"""Returns a list that contains one prediction for each input given

		Parameters

		-x: 1-dimensional array of inputs for the model to make predictions for and return
		"""

		# Build a list of predictions with y = mx + b applied through np array operations
		return self.m * x + self.b

	def __update_weights(self, x, y, y_hat):
		"""Iterate over gradient descent once, including calculating the gradient and applying it"""

		# Create a list of the differences between y_hat and y
		diff = y_hat - y

		# Calculate the gradients for the weight and bias respectively
		gradient = np.array([np.dot(x, diff) / len(x),
							 np.sum(diff)    / len(x)])

		# Calculate and update the S list, which inhibits the movement of the weights (ADAGRAD)
		self.s += gradient * gradient
		gradient /= np.sqrt(self.s + self.epsilon)

		# Apply the gradient
		self.m -= gradient[0]
		self.b -= gradient[1]

		# If this model is tracking metrics, calculate error and record error and gradient
		if self.track_metrics:
			self.__calculate_error(y_hat, y)
			self.error_hist.append(self.error)
			self.gradient_hist.append(gradient)

		# Return the magnitude of the gradient, so the training method can know when to stop
		return gradient.dot(gradient)

	def __calculate_error(self, y_hat, y):
		"""Calculates the model's error by averaging each data point's error (y_hat - y) ^ 2"""

		# Square each difference and calculate half the average (half for derivative reasons)
		self.error = np.sum(np.square(y_hat - y)) / (2 * len(y))


class MultipleLinearRegression:
	"""Machine Learning model that takes in predicting variables, and predicts values for data points"""

	def __init__(self, weight_count, epsilon=0.00001, min_gradient=0.0004, init_values=None, headers=None,
					default_prints=True, track_metrics=False):
		"""Builds a Linear Regression model with multiple input variables


		Parameters

		-weight_count: int.  Number of features in the dataset that will later be trained on

		-epsilon: float.  Value added to S before square rooting it during the gradient application
		(Adagrad technique)

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
		"""

		# Initialize state variables
		self.weight_count = weight_count
		self.var_count = weight_count + 1
		self.error = None
		self.default_prints = default_prints
		self.track_metrics = track_metrics
		self.s = np.array([0.0 for _ in range(self.var_count)])

		# Instantiate headers variable
		if headers is None:
			self.headers = ['x' + str(i) for i in range(1, self.var_count)]
			self.headers.insert(0, 'b')
		else:
			if len(headers) == self.var_count:
				self.headers = headers
			else:
				print('Header list must be ' + str(self.var_count) + '!')
				self.headers = ['x' + str(i) for i in range(1, self.var_count)]
				self.headers.insert(0, 'b')

		# Initialize weight/bias variables
		if init_values is None:
			self.w = np.array([0.0 for _ in range(self.var_count)])
		else:
			if len(init_values) != self.var_count:
				print('invalid init_values!  len is not', weight_count, 'plus 1 (for bias)')
				return
			else:
				self.w = np.array(init_values)

		# Metric state variables
		if track_metrics:
			self.gradient_hist = []		# History of model's past gradients, after S is applied
			self.error_hist = []		# History of model's error value each time it is calculated
			self.distances = []			# Distance that each guess is from dataset value (populated in evaluate())

		# Store Hyperparameters
		self.epsilon = epsilon
		self.min_gradient = min_gradient

	def train(self, x, y, prints=None):
		"""Trains the model to the given data by iterating gradient descent


		Parameters

		-x: 2-dimensional, numerical array of inputs of dimension [(data point count) x (feature count)]

		-y: 1-dimensional, numerical array of true values for inputs X of dimension [data point count]

		-prints: Boolean.  Dictates whether this model will print the results at the end of the
		training process.  If left as None, the model.default_prints is used
		"""

		# Go through gradient descent for the given iterations
		while True:
			# Predict all inputs with new weights
			y_hat = self.predict_all(x)
			# Update the weights to be a little closer (gradient descent)
			g = self.__update_weights(x, y, y_hat)
			# Break from loop if min_gradient is reached
			if g <= self.min_gradient:
				break

		# Calculate the error for the model
		self.__calculate_error(y_hat, y)

		# Print results
		if (prints is not None and prints) or (prints is None and self.default_prints):
			print('\nTraining done!')
			print('Error:', self.error)
			print('Training set size:', len(y))
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

	def __update_weights(self, x, y, y_hat):
		"""Iterate over gradient descent once, including calculating the gradient and applying it"""

		# Create a list of the differences between y_hat and y
		diff = y_hat - y

		# Split the X matrix into feature columns
		cols = np.hsplit(x, self.weight_count)
		# This turns [[1], [2], [3]] into [1, 2, 3]
		cols = np.array(cols).reshape((self.weight_count, len(x)))
		# Store this for use in a second
		n = len(x)

		# Calculate the gradient
		gradient = np.zeros(self.var_count)
		gradient[0] = np.sum(diff) / n
		for i in range(1, self.var_count):
			gradient[i] = np.dot(cols[i-1], diff.T) / n

		# Calculate and apply the S factor (ADAGRAD)
		g_squared = gradient * gradient
		self.s += g_squared
		gradient /= np.sqrt(self.s + self.epsilon)

		# Apply the gradient
		self.w -= gradient

		# Calculate the magnitude of the gradient vector
		g_magnitude = gradient.dot(gradient)

		# If this model is tracking metrics, calculate error and record error and gradient
		if self.track_metrics:
			self.__calculate_error(y_hat, y)
			self.error_hist.append(self.error)
			self.gradient_hist.append(g_magnitude)

		# Return the gradient magnitude so that the train method can know when to stop going
		return g_magnitude

	def __calculate_error(self, y_hat, y):
		"""Calculates the model's error by averaging each data point's error (y_hat - y) ^ 2"""

		# Square each difference and calculate half the average (half for derivative reasons)
		self.error = np.sum(np.square(y_hat - y)) / (2 * len(y))
