# Matthew Williams
# Moved into Legacy on 10/2/23

import numpy as np
from math import log


class BinaryLogisticRegression:
    """Basic class for Logistic Regression. Accepts one input feature X, and classifies it as 1 or 0

    >For multiple inputs and multiple classes, see MultinomialLogisticRegression"""

    def __init__(self, epsilon=0.00001, min_gradient=0.0004, init_m=0, init_b=0, headers=None, threshold=0.5):
        """Initializes the Regression Model.  It takes in one feature set, and predicts each as 1 or 0


        Parameters

        -epsilon: float.  Value added to S before square rooting it during the gradient application
		(Adagrad technique)

		-min_gradient: float.  Defines the minimum magnitude of gradient that will still be
		applied before stopping the training process

		-init_m: numerical.  Initial value of the weight, default is zero

		-init_b: numerical.  Initial value of the bias, default is zero

		-headers: array of len = 2, String.  Preferred names of the weight and bias respectively (b = [1])
		default is ['m', 'b']

		-threshold: float in range (0, 1) exclusive.  Level at which a data point's score after the sigmoid
		transformation must be in order to be classified as '1'.  Default is a neutral 0.5
        """

        # Instantiate base variables
        self.m = init_m
        self.b = init_b
        self.s = np.array([0.0, 0.0])

        # Instantiate headers variable
        if headers is None:
            # If no headers is specified, use default values
            self.headers = ['m', 'b']
        else:
            if len(headers) == 2:
                # If input is a valid length and is usable as headers, store them
                self.headers = headers
            else:
                # Input is an invalid length, use default values instead
                print('Header list must be 2!  Using default values instead')
                self.headers = ['m', 'b']

        # Instantiate Hyperparameters
        self.min_gradient = min_gradient
        self.threshold = threshold
        self.epsilon = epsilon

    def train(self, x, y, prints=True):
        """Trains the model to the given data by iterating gradient descent


		Parameters

		-x: 1-dimensional array of inputs (numerical)

		-y: 1-dimensional array of labels (binary)

		-prints: Boolean.  Dictates whether this model will print the results at the end of the training process
		"""
        # Simple input verification
        if len(x) == 0:
            print('Input array X must have data points!  length of x: 0')
            return
        if len(x) != len(y):
            print(f'X and Y must have an equal number of items!  X:{len(x)}, Y:{len(y)}')
            return
        if np.max(y) > 1 or np.min(y) < 0:
            print('Y must be a binary array that represents each data point\'s class!')
            return

        # Update weights the complete number of times
        while True:
            y_hat = self.predict_all(x)
            g = self.__update_weights(x, y, y_hat)
            if g <= self.min_gradient:
                break

        print('Shape of y_hat:', y_hat.shape)
        print('Shape of X:', x.shape)
        print('Shape of Y:', y.shape)
        print('Last G:', g)
        print('Shape of Gradient: (2)')

        # Calculate accuracy
        accuracy = self.__calculate_accuracy(y_hat, y)

        if prints:
            # Print results
            print('Training done!')
            print(('Final accuracy: %.2f' % (accuracy * 100)) + '%')
            print('Items in training data set:', len(y))
            print(self.headers[0] + ':', self.m)
            print(self.headers[1] + ':', self.b)

    def evaluate(self, x, y, prints=True):
        """Evaluates accuracy of the model


        Parameters

		-x: 1-dimensional array of inputs (numerical)

		-y: 1-dimensional array of labels (binary)

		-prints: Boolean.  Dictates whether this model will print the results at the end of the training process
		"""

        # Simple input verification
        if len(x) == 0:
            print('Input array X must have data points!  length of x: 0')
            return
        if len(x) != len(y):
            print(f'X and Y must have an equal number of items!  X:{len(x)}, Y:{len(y)}')
            return
        if np.max(y) > 1 or np.min(y) < 0:
            print('Y must be a binary array that represents each data point\'s class!')
            return

        # Create list of guesses, and calculate accuracy from it
        y_hat = self.predict_all(x)
        accuracy = self.__calculate_accuracy(y_hat, y)

        # Print results
        if prints:
            print('\n\nEvaluation results:')
            print(('Accuracy: %.2f' % (accuracy * 100)) + '%')
            print('Items in evaluation data set:', len(y))

    def predict_all(self, x):
        """Assembles a prediction list from the sigmoid of the model's prediction

        Parameters

        -x: 1-dimensional array of inputs (numerical)
        """

        return 1 / (1 + np.exp(- (x * self.m + self.b) ))

    def get_changing_point(self):
        """Returns the input point at which the prediction shifts from 0 to 1

        Formula can be attained from writing out the sigmoid, solving for z,
        and solving for where mx + b is equal to z given model m and b

        Returns:
        Value of X that serves as a partition between the inputs that the model
        would classify as '0', and the ones it would classify as '1'
        """

        # Set z equal to the inverse of sigmoid which is ln( (1-x) / x )
        z = -( log( (1.0-self.threshold) / self.threshold ) )
        # Steps backwards through y = mx + b by removing b (solve for x now)
        x = z - self.b
        # Takes next step by removing m
        x /= self.m

        # Return final answer, which is the number
        return x

    def __calculate_accuracy(self, y_hat, y):
        """Counts the number of times that the final guess of the model matches the label"""

        # Create an array of final guesses (with threshold applied)
        guesses = np.where(y_hat >= self.threshold, 1, 0)

        # Create an array of collisions between the guess and the label, and store it's length
        correct = len(np.where(guesses == y)[0])

        # Store the total number of data points given
        total = len(y)

        # Return the ratio of correct to total
        return correct / total

    def __update_weights(self, x, y, y_hat):
        """Calculates a gradient for the weights and applies it"""

        # Calculate differences between guesses and labels
        diff = y_hat - y

        # Instantiate gradient variable
        gradient = np.array([0.0, 0.0])

        # Calculate gradient
        gradient[0] = np.dot(x, diff) / len(x)
        gradient[1] = np.sum(diff) / len(x)

        # Calculate and apply the S factor (ADAGRAD)
        self.s += gradient * gradient
        gradient /= np.sqrt(self.s + self.epsilon)

        # Apply gradient
        self.m = self.m - gradient[0]
        self.b = self.b - gradient[1]

        return gradient.dot(gradient)
