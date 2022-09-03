import numpy as np
import math


class BinaryLogisticRegression:
    """Basic class for Logistic Regression. Accepts one input feature X, and classifies it as 1 or 0

    >For multiple inputs and binary output, see MultipleBinaryLogisticRegression
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

        return 1 / (1 + np.exp(- (x * self.m + self.b)))

    def get_changing_point(self):
        """Returns the input point at which the prediction shifts from 0 to 1

        Formula can be attained from writing out the sigmoid, solving for z,
        and solving for where mx + b is equal to z given model m and b

        Returns:
        Value of X that serves as a partition between the inputs that the model
        would classify as '0', and the ones it would classify as '1'
        """

        # Set z equal to the inverse of sigmoid which is ln( (1-x) / x )
        z = -( math.log( (1.0-self.threshold) / self.threshold ) )
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


class MultinomialLogisticRegression:
    """Logistic Regression model that takes in an input matrix X, and classifies each row as one of a given number of
    classes

    >For only one input and binary output, see BinaryLogisticRegression
    >For multiple inputs and binary output, see MultipleBinaryLogisticRegression
    """

    def __init__(self, weight_count, class_count, epsilon=0.0000001, min_gradient=0.03):
        """Initializes the Regression Model.  It takes in a feature matrix, and predicts each input as 1 or 0


        Parameters

        -weight_count: int.  Number of features in the dataset that will later be trained on

        -class_count: int.  Number of possible classes that this model will be predicting for

        -epsilon: float.  Value added to S before square rooting it during the gradient application
        (Adagrad technique)

        -min_gradient: float.  Defines the minimum magnitude of gradient that will still be
        applied before stopping the training process


        ((Note:  No threshold is needed for this model as it chooses the class with the highest probability of
        being in a given class, rather than if the epsilon of the class score is above a certain level, as in
        Binary regression.))
        """

        # State Variables
        self.weight_count = weight_count
        self.var_count = weight_count + 1
        self.class_count = class_count
        self.w = np.zeros((class_count, self.var_count))
        self.s = np.zeros((class_count, self.var_count))
        self.accuracy = 0
        self.loss = 0

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

        print('Beginning the training process..')

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

        self.__print_detailed_accuracy(y, y_hat)

    def evaluate(self, x, y, prints=True):
        """Evaluates accuracy of the model


        Parameters

		-x: 2-dimensional array of numerical inputs with dimensions [(data point count) x (feature count)]

		-y: 1-dimensional array of binary labels with dimensions [(data point count) x (possible class count)].
		This is the one-hot representation of the class of the data points in input X

		-prints: Boolean.  Dictates whether this model will print the results at the end of the training process
		"""

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
        # self.__print_detailed_accuracy(y, y_hat)

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

        # Alter, then apply the gradient -------------------------------------------------------

        # Add the square of the current gradient to self.S (Adagrad)
        self.s += (gradient * gradient)
        # Divide the gradient by self.S plus a small epsilon
        gradient /= np.sqrt(self.s + self.epsilon)
        # Apply the gradient to the model's weight matrix
        self.w -= gradient

        # Return the Magnitude of the gradient matrix so the train() method knows when to stop
        return np.sqrt(abs(np.sum(gradient * gradient)))
