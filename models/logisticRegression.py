# Author: Matthew Williams

import numpy as np
from util import optimizers


def cast_to_one_hot(y):
    """The goal is that any input for class (integer class codes or one-hot encodings) is returned as a one-hot.
    Returns the input if that is already the case, and returns a converted representation if needed."""
    
    # If the first element of y is not iterable
    if not hasattr(y[0], '__iter__'):
        # If the list is (presumably, given the first element) comprised of numbers (class codes)
        try:
            # Test to see if it will let me cast y's elements to ints
            y = y.astype(int)
            # Record the maximum index needed to cover y's class codes
            max_index = max(y) + 1
            # Instantiate a blank list of 0's (What will become the one-hot
            final_list = np.zeros((len(y), max_index))
            # Loop through each row, and set the class code's column to 1
            for y_i,list_r in zip(y, final_list):
                final_list[y_i] = 1
            # Return the integer-casted one-hot representation
            return final_list.astype(int)
        except ValueError:
            raise Exception(f'Class codes (values in y) must be Integers!  {type(y[0])} won\'t work!')
    # If the first element in y is iterable, it is assumed that it is a valid class code array
    else:
        # If the array follows general, weakly checked conditions of a valid one-hot encoding
        if np.sum(y) == len(y) and np.max(y) == 1:
            # Simply return the input
            return y
        # Else, if the input is iterable but not a valid one-hot
        else:
            # Raise an exception and tell why
            raise Exception('Input Y has iterable elements, but is not a valid one-hot encoding!')


class LogisticRegression:
    """Logistic Regression model that takes in an input matrix X, and classifies each row as one of a given number of
    classes
    
    >For only one input and binary output, see BinaryLogisticRegression
    """
    
    def __init__(self, epsilon=0.0000001, min_gradient=0.015, optimizer=optimizers.Adagrad()):
        """Initializes the Regression Model.  It takes in a feature matrix, and predicts each input as 1 or 0


        Parameters

        -epsilon: float.  Value added to S before square rooting it during the gradient application
        (Adagrad technique)

        -min_gradient: float.  Defines the minimum magnitude of gradient that will still be
        applied before stopping the training process


        ((Note:  No threshold is needed for this model as it chooses the class with the highest probability of
        being in a given class, rather than if the epsilon of the class score is above a certain level, as in
        Binary regression.))
        """

        # State Variables
        self.weight_count = None
        self.var_count = None
        self.class_count = None
        self.w = None
        self.optimizer = optimizer
        
        self.accuracy = 0
        self.loss = 0
        
        # Indicates if this model's inputs are scalars rather than vectors
        self.is1D = False

        # Hyperparameters
        self.epsilon = epsilon
        self.min_gradient = min_gradient
    
    def train(self, x, y, prints=True):
        """Trains the model to the given data by iterating gradient descent


		Parameters

		-x: 2-dimensional array of numerical inputs with dimensions [(data point count) x (feature count)]

		-y: 1-dimensional array of binary labels with dimensions [(data point count) x (possible class count)].
		This is the one-hot representation of the class of the data points in input X

		-prints: Boolean.  Dictates whether this model will print the results at the end of the training process
		"""
        
        # Cast the input (any valid one-hot or class codes) into one-hot (This method is mostly input verification)
        y = cast_to_one_hot(y)
        
        # Simple input verification
        if len(x) == 0:
            raise Exception('Input matrix X must have data points!  length of x: 0')
        if len(x) != len(y):
            raise Exception(f'X and Y must have an equal number of rows!  X:{len(x)}, Y:{len(y)}')
        
        self.__initialize_state(x, y)
        
        # Signal to the console that the training has started
        print('Beginning the training process..')
        
        # Initialize the variables that rely on feature and class counts
        self.__initialize_state(x, y)
        
        # Add a column of 1's to X
        x = np.column_stack(([1 for _ in range(len(x))], x))
        
        while True:
            
            # Predict a class for each input
            y_hat = self.predict_all(x)
            
            # Record the magnitude of the gradient after applying it to the weights
            g = self.__update_weights(x, y, y_hat)

            if g <= self.min_gradient:
                break
        
        # Calculate the model's final accuracy and loss
        self.__calculate_accuracy(y, y_hat)
        self.__calculate_loss(y, y_hat)
        
        # Print results
        if prints:
            print('\nTraining done!')
            print('Loss: %.3f' % self.loss)
            print(('Accuracy: %.2f' % (self.accuracy * 100)) + '%')
            print('Items in training data set:', len(y), '\n')
    
    def evaluate(self, x, y, prints=True, details=False):
        """Evaluates accuracy of the model


        Parameters

		-x: 2-dimensional array of numerical inputs with dimensions [(data point count) x (feature count)]

		-y: 1-dimensional array of binary labels with dimensions [(data point count) x (possible class count)].
		This is the one-hot representation of the class of the data points in input X

		-prints: Boolean.  Dictates whether this model will print the results at the end of the training process

		-details: Boolean.  Dictates whether the confusion matrices will be printed at the end.  Default to False.
		"""
        
        # Cast the y values to one-hot
        y = cast_to_one_hot(y)

        # Simple input verification
        if len(x) == 0:
            print('Input matrix X must have data points!  length of x: 0')
            return
        if len(y) != len(x):
            print(f'X and Y must have an equal number of rows!  X:{len(x)}, Y:{len(y)}')
            return
        if hasattr(x[0], '__iter__') and len(x[0]) != self.weight_count:
            print(f'X must have as many columns as the original training set!  X:{len(x)} Original:{self.weight_count}')
            return
        if len(y[0]) != self.class_count:
            print(f'Y must have as many columns as the original training set!  Y:{len(y)} Original:{self.class_count}')
            return

        # Add a column of 1's to X
        x = np.column_stack(([1 for _ in range(len(x))], x))

        # Predict a class for each input row
        y_hat = self.predict_all(x)

        # Calculate the accuracy and loss of the model
        self.__calculate_accuracy(y, y_hat)
        self.__calculate_loss(y, y_hat)

        # Print results
        if prints:
            print('\nDone Evaluating!')
            print('Loss: %3f' % self.loss)
            print(('Accuracy: %.2f' % (self.accuracy * 100)) + '%')
            print(f'Items in evaluation data set:{len(y)}')

        # Print the confusion matrices
        if details:
            self.__print_detailed_accuracy(y, y_hat)
    
    def predict_all(self, x):
        """Assembles a prediction list from the given data set

        Parameters

        -x: 2-dimensional array of numerical inputs with dimensions [(data point count) x (feature count)]
        """

        # Start by multiplying each weight vector by X
        raw_y_hat = np.stack([
            x * self.w[k]
            for k in range(self.class_count)
        ], axis=1)

        # Sum each row to get a prediction score for that X and class
        raw_y_hat = np.sum(raw_y_hat, axis=2)

        # Subtract each row's max from the row's elements to add stability to e^x
        # This requires transposing y_hat to get the N dimension going horizontal, then
        # subtracting the maxes from it which automatically does that for each (now) column
        # which is the K dimension.  Then transposing that gets it back to the regular format
        y_hat_maxes = np.max(raw_y_hat, axis=1)  # This is a vector of length N
        raw_y_hat = (raw_y_hat.T - y_hat_maxes).T

        # Apply e^x to the array
        raw_y_hat = np.exp(raw_y_hat)

        # Sum each X's class scores and
        # Divide original numbers by row sum (normalize to sum(row)=1)
        # Here, each row is referring to a input X's array of class scores
        normalized_y_hat = (raw_y_hat.T / np.sum(raw_y_hat, axis=1).T).T

        # Restrict bounds of results from 0.0001 - 0.9999 (No '1' or '0')
        normalized_y_hat[np.where(normalized_y_hat > 0.9999)] = 0.99999
        normalized_y_hat[np.where(normalized_y_hat < 0.0001)] = 0.0001

        # Return list of probabilities per class per input
        return normalized_y_hat
    
    def get_weights(self, do_round=True):
        """Returns the weight matrix W of this model.  Each row is a class, and each column is a feature.  The first
        column is the bias.

        Parameters

        -do_round: Dictates whether the values are rounded to the nearest whole number or not.  Default is True"""

        # If rounded values is wanted
        if do_round:
            # Add 0.5 to each and return the floored version
            return np.around(self.w)
        # Assuming that didn't happen, return the true weights
        return self.w
    
    def __calculate_loss(self, y, y_hat):
        """Sums the log loss of each normalized guess (0/1)"""

        # Equation:
        #       Loss(x, y) = -(sum{i=1 -> N}(Yk[n] * log( (e^(W T x)) / sum{j=1 -> K}(e^(W T x)) )
        # AKA: Take each prediction matrix (one per possible class), and divide each data point's predictions
        # by the sum of that point's prediction in order to normalize the prediction outputs as percentages that
        # all add up to 1 for that point.  Then, take the log of each normalized prediction that correlates to the
        # true class (so if class=1 in the labels, only take the log of the prediction for class=1).  Add this
        # log to a running sum, and at the end, make the whole thing negative (it will always be negative at the
        # start, so multiplying by -1 makes it always positive, which is what we want here).

        # Instantiate the loss variable
        loss_total = 0

        # Loop through all data points
        for n in range(len(y)):
            # Add the datapoint's loss to the total
            loss_total += np.log(y_hat[n, y[n].argmax()])

        # Store the total
        self.loss = -loss_total
    
    def __calculate_accuracy(self, y, y_hat):
        """Counts the number of times that the final guess of the model matches the label"""

        # Counter variable for correct predictions
        correct = 0

        # For every data point (row), see if the y has a 1 in the same spot that y_hat has it's largest value
        for n in range(len(y_hat)):
            if y[n, y_hat[n].argmax()] == 1:
                # If so, increase the counter of correct predictions
                correct += 1

        # Store the ratio of correct predictions to total data points
        self.accuracy = correct / len(y)
    
    def __print_detailed_accuracy(self, y, y_hat):
        """Calculates then prints info like the confusion matrix, and other stats"""

        # Confusion matrix arrays, indexed by class number
        true_pos = np.zeros(self.class_count)
        false_pos = np.zeros(self.class_count)
        true_neg = np.zeros(self.class_count)
        false_neg = np.zeros(self.class_count)

        # For every data point (row), record the true class and prediction, then build a
        # confusion matrix for each class
        for n in range(len(y_hat)):
            # Record data
            y_ind = y[n].argmax()
            y_hat_ind = y_hat[n].argmax()
            # For every class that this model classifies
            for k in range(self.class_count):
                # If this is the prediction
                if k == y_hat_ind:
                    # If this is also the label
                    if k == y_ind:
                        # Add one to the true positives for this class
                        true_pos[k] += 1
                    else:
                        # If not, add one to the false positives for this class
                        false_pos[k] += 1
                else:
                    # If this prediction is 0
                    if k == y_ind:
                        # If the label is 1 though, add one to false negatives
                        false_neg[k] += 1
                    else:
                        # Otherwise, if the label really is 0, add one to true negatives
                        true_neg[k] += 1

        # Print the matrices for the classes
        print('\n\nConfusion Matrices per class\n')
        for k in range(self.class_count):
            print(f'class {k+1}\t |True\t|False')
            print(f'------------------------')
            print(f'Positive | {int(true_pos[k])}\t| {int(false_pos[k])}')
            print(f'Negative | {int(true_neg[k])}\t| {int(false_neg[k])}\n\n')

        # Do basically what the normal __calculate_accuracy() does, general stats
        total_true = 0
        for n in range(len(y)):
            if y[n, y_hat[n].argmax()] == 1:
                total_true += 1

        # Display basic accuracy
        print(f'{total_true} true predictions, and {len(y) - total_true} false ones.  That is a rate of:')
        print('%.2f' % ((total_true / len(y))*100) + '%!')
    
    def __update_weights(self, x, y, y_hat):
        """Runs one iteration of gradient descent by calculating the gradient, then applying it"""

        # Calculate Gradient --------------------------------------------------------
        diff = y_hat - y

        # Instantiate the inner term that is within the Reimann sum
        sum_inner = np.zeros((len(y), self.class_count, self.var_count))

        # For each class, build out 3D matrix for the gradient, before summing over data points
        for k in range(self.class_count):
            # Store column of difference matrix
            column = diff[:, k]
            # Multiply the column by each column of x
            result = np.array([x[:, f] * column for f in range(self.var_count)])
            # Add that matrix to the 3D matrix being calculated
            sum_inner[:, k, :] = result.T

        # Get the raw (un-normalized) gradient scores by summing over all data points,
        # and ending with just a matrix of gradients
        gradient = np.sum(sum_inner, axis=0)

        # Apply the gradient -------------------------------------------------------
        self.optimizer.update(self.w, gradient)

        # Return the Magnitude of the gradient matrix so the train() method knows when to stop
        return np.sqrt(abs(np.sum(gradient * gradient)))
    
    def __initialize_state(self, x, y):
        """Initializes state variables that rely on the number of features and classes"""
        # Record the number of features / weights, and classes passed in with X and Y
        
        # Get around the error of len(x[0]) == 1 crashing because z[0] isn't iterable, to set length to 1 anyways
        if not hasattr(x[0], '__iter__'):
            weight_count = 1
        else:
            weight_count = len(x[0])
        
        # Set class count to the length of the first element of y (a one-hot encoding of class)
        class_count = len(y[0])

        # Initialize the state variables that require knowledge of these counts
        self.weight_count = weight_count
        self.var_count = weight_count + 1
        self.class_count = class_count
        self.w = np.zeros((class_count, self.var_count))
        self.s = np.zeros((class_count, self.var_count))


def build_from_json(json_obj):
    pass
