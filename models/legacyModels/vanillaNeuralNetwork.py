# Author: Matthew Williams
# End Date: 7/13/2023
# Moved to Legacy on 10/2/23

import time
from math import ceil
import numpy as np
from util import optimizers, costFunctions, activationFunctions

"""
Few Notes on the quirks of this class:

Overall, this wasn't meant to be a super robust class.  It was created more as a "Can
I even make a working network?" kind of thing.  So there are a few artifacts left from
the fact that I didn't have a very full understanding of how neural nets even functioned
(mathematically or programmatically) before I started this class.

1) Layers lists
    These ended up a bit odd.  The first entry in the weight and bias matrices is a zero.
    This is because the input is treated as the activation of the "first layer", which is
    how a professor explained it in a lecture I watched.  I believe they meant to show
    how a layer really only needs to know about it's input to do it's job and that the
    input can be treated as any other activation.  But I ended up making it so that the
    first layer in the class definition was always ignored (and therefore should be of
    the form (input_size, None).  You'll also notice that loops that go through the layers
    often start at 1 to avoid these filler values that simulate the input being an activation.

2) Shared Optimizer
    While the more current classes that make up my neural networks have an independent
    optimizer for each layer (separately instantiated), this class uses one for the
    whole thing.

3) Robustness Checks
    While the more current classes include checks to make sure inputs are as they should
    be and what not, this class does not.  It just assumes that if nothing breaks, then it
    works, and if it does break, the user can deal with the error that appears down the
    stream as the function runs.

4) Activation Function Array Indexing
    You'll notice at the end of the back-propagation loop in the propagate() method that,
    in calculating the new delta value, it applies the derivative of the activation function
    indexed at [-l] with the z value at [-l-1].  Not sure how those line up or why it works,
    but it doesn't crash and it gives good answers so I'm not messing with it.  Just know that
    it's a little odd how that system works, and I don't fully remember (at the time of this
    documentation) how it got to be what it is.

5) Initialization
    While the more current classes use helper methods to initialize layers' weights and biases,
    this class has it baked in to the constructor.  Its methodology matches that of the method
    initializers.random_initializer().
"""


class VanillaNetwork:
    
    def __init__(self, layers: list[tuple], rand_range=2, rand_min=-1):
        """Initializes a Basic Neural Network.
        
        Builds the weight and bias matrices for the network according to the given layers.
        
        @param layers: list of tuples.  Representation of the layers in the form
            [(width, activation), (width, activation)]
            (Activation functions included as helper methods in util.activationFunctions.py (Int representations)
            (No activation function needed on layer 0, as that is assumed to be the input layer)
        
        Example Initialization layers list:
        [
        (5, None),
        (7, af.relu),
        (8, af.relu),
        (3, af.relu),
        (4, af.sigmoid),
        (1, af.v_linear)
        ]
        """
        
        # layers[n][0] = width of layer (int)
        # layers[n][1] = activation function of layer (int) (see lookup lists above)
        
        # Initialize layers list
        self.layers = layers
        
        # Initialize the cost function and it's derivative
        self.cost_func = costFunctions.wrong_cost_func
        self.d_cost_func = costFunctions.wrong_cost_func
        
        # Initialize array for activation functions and respective derivatives
        self.act_funcs = []
        self.d_act_funcs = []
        
        # Initialize array for the weights and biases respectively
        self.w = [0]
        self.c = [0]
        
        # Set up the blank optimizer field
        self.optimizer = None
        
        # Loop through each layer in order to build lists of activations and sizes of self.w from them
        # This skips layers[0] (input layer) because it's weight it caught by the (i-1) in the W part, and
        #       it doesn't have an activation function so that doesn't matter
        for i in range(1, len(layers)):
            # Create a random W and C matrix of the right size to fit between layers (i) and (i-1)
            self.w.append(np.random.rand( layers[i][0], layers[i-1][0] ) * rand_range + rand_min)
            self.c.append(np.random.rand( layers[i][0] ) * rand_range + rand_min)
            # Store the activation function and it's corresponding derivative function
            af, df = activationFunctions.get_activation_functions(layers[i][1])
            self.act_funcs.append(af)
            self.d_act_funcs.append(df)
    
    def compile(self, cost_function: int, optimizer=optimizers.AdamForNetworks()):
        """Gets the model ready to use by:
        Setting the cost function and it's corresponding derivative function

        ex: model.compile(cost=neuralNetwork.least_squares)"""
        
        self.cost_func, self.d_cost_func = costFunctions.get_cost_functions(cost_function)
        self.optimizer = optimizer

    def train(self, x, y, epochs=30, batch_size=32):
        """Runs batches of data points through the network forwards, find the error and gradients coming backwards,
        and applies those gradients to the weights and biases of the network.  Then repeats (epochs) number of times.
        Returns the array of costs (one for each epoch of training)."""
    
        # Record when this method started
        start_time = time.time()
    
        # Initialize the random generator (for batch selection), and store the size of the data input and the possible
        #   indices from 0 to the total_size
        rng = np.random.default_rng()
        index_range = range(0, len(x))
        costs = []
        # Loop through for each epoch
        for _ in range(epochs):
            # Prep each batch and initialize the running tallies for the gradients and biases
            batch = rng.choice(index_range, size=batch_size)  # Creates a random list of indices, not actual data points
            run_w, run_c = self.propagate(x[batch], y[batch])
        
            costs.append(self.get_cost(x, y))
            # Apply the gradients and biases (w and c respectively)
            self.optimizer.update(self.w[1:], self.c[1:], run_w, run_c)
    
        # Calculate the cost of this model with the data set
        cost = self.get_cost(x, y)
    
        print('\nTraining done!')
        print('Final cost:', ceil(cost))
        print('Epochs:', epochs)
        print('Batch Size:', batch_size)
        print('Time to train: %.1f seconds' % (time.time() - start_time), "\n")
        
        return costs
    
    def evaluate(self, x, y, batch_size=32):
        
        start_time = time.time()

        cost = self.get_cost(x, y)
        
        print('Evaluation done!')
        print('Evaluation cost:', ceil(cost))
        print('Batch size:', batch_size)
        print('Evaluation time: %.1f seconds' % (time.time() - start_time))
        print("\n\n")
    
    def propagate(self, x, y):
        """Feeds the item forwards, calculates error compared to y, then calculates gradients backwards.
        
        Parameters:
        -x: Matrix [batch size, input dims].  Input values to be propagated (batched)
        -y: Array of size {batch size}.  Numerical labels for each input row in x
        
        Returns
        -List of average gradient matrices for W
        -Array of average gradients for C
        """
        
        """
        Element Sizes (for future reference)
        -x: (n, d)      [2D]
        -y: (n,)        [1D]
        -a_s: (J, n, j) [3D] {Staggered}
        -z_s: (J, n, j) [3D] {Staggered}
        -delta: (n, j*) [2D] {j* = number of nodes in layer one step closer to result side}
        Note: Staggered arrays cannot be np.ndarray
        """
    
        # ---------------------[ Feed Forward \ Predict Stage ]-------------------------
        # Initialize the running arrays of the Z and A activations for each layer
        z_s = [0]  # 0 is an unimportant value, just a placeholder to align it to the a_s array
        a_s = [x]  # First activation is the input
        
        # Loop through the layers forwards and activate each one based off of the previous activation
        # (The loop starts at 1 instead of 0 so the input layer is not activated as a hidden layer)
        for l in range(1, len(self.layers)):
            z_s.append(np.array(a_s[l-1] @ self.w[l].T + self.c[l]))
            a_s.append(self.act_funcs[l-1](z_s[l]))
        # --------------------[ End Feed Forward Stage ]------------------------
    
        # ---------------------[ Back Propagation Stage ]---------------------------
        # Create the lists of matrices for the weights' and biases' gradients
        weight_gradients = [np.zeros(w.shape) for w in self.w[1:]]
        bias_gradients = [np.zeros(c.shape) for c in self.c[1:]]
    
        # Initialize the running error variable 'delta'
        delta = self.d_act_funcs[-1](z_s[-1]) * self.d_cost_func(a_s[-1], y.reshape(-1, 1))
        for l in range(1, len(self.layers)):
            # Calculate W'
            weight_gradients[-l] = delta.T.dot(a_s[-l - 1])
            # Calculate C'
            bias_gradients[-l] = delta
            # Calculate new delta
            delta = self.d_act_funcs[-l](z_s[-l - 1]) * delta.dot(self.w[-l])
        # --------------------[ End Back Propagation Stage ]------------------------
        
        # 'flatten' each weight_gradients and bias_gradients entry
        for i in range(0, len(weight_gradients)):
            # weight_gradients[i] = np.sum(weight_gradients[i], axis=0)
            bias_gradients[i] = np.sum(bias_gradients[i], axis=0)
            
        # Return the array of gradients for W, and the vector of C gradients
        return weight_gradients, bias_gradients
    
    def predict(self, x):
        """Returns the prediction from this network for a single data point.  Does not store activations along
        the way, or calculate or apply any gradients"""
        
        # Initialize the previous activation (a[l])
        # (This is word play since at the end, it is returned as the last (final) activation, but in the loop, it is the
        # last (previous) activation haha)
        last_a = x
        
        # Loop through the layers forwards and activate each one based off of the previous activation
        # (The loop starts at 1 instead of 0 so the input layer is not activated as a hidden layer)
        for l in range(1, len(self.layers)):
            # Calculate the activation of the current layer based off of the previous activation
            last_a = self.act_funcs[l-1]( self.w[l].dot(last_a) + self.c[l] )
        
        # Return the final layer's activation (final layer = output layer)
        return last_a
    
    def get_cost(self, x, y):
        total_cost = 0
        data_count = len(x)
        for i in range(data_count):
            total_cost += self.cost_func(self.predict(x[i]), y[i])
        return np.sum(total_cost)
