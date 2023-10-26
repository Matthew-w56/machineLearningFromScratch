# Matthew Williams
# Moved into Legacy on 10/2/23

import numpy as np


class SimpleLinearRegression:
    """Machine Learning model that takes in a predicting variable, and predicts values for data points.

    This model uses the Adagrad Gradient Descent, and does not support the variable optimizers or gradient
    calculators that the LinearRegression does.  This class is basically depreciated at this point.  It was built
    only as a learning exercise.  Still functional, but the Linear Regression can handle Multiple as well as Simple
    Linear Regression."""
    
    def __init__(self, min_gradient=0.0001, epsilon=0.00001, init_m=0, init_b=0, headers=None, default_prints=True,
                 track_metrics=False):
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
        
        # Verify the inputs -----------------------
        if len(x) == 0:
            raise Exception('Cannot have an empty dataset X!')
        if len(x) != len(y):
            raise Exception(f'Data sets X and Y must be of equal length! ({len(x)} != {len(y)})')
        # Done verifying inputs ----------------------------
        
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
        
        # Verify inputs --------------------------------
        if len(x) == 0:
            raise Exception('Dataset X cannot be empty!')
        if len(x) != len(y):
            raise Exception(f'Datasets X and Y of different length! ({len(x)} != {len(y)})')
        # Done verifying inputs -------------------------------
        
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
