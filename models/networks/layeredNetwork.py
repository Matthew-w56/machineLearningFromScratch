# Author: Matthew Williams
# Start Date: 7/11/2023

import time
import copy
from math import ceil
import numpy as np
from util import optimizers, costFunctions
from util.layer import networkLayerDictionary
from util.layer.networkLayer import Layer
from util.optimizers import OptimizerForNetworks


class LayeredNetwork:
    
    def __init__(self, layers: list[Layer]):
        """
        Initializes a Neural Network comprised of a list of Layers.
        
        @param layers:  List of instances of the Layer class.  Sorted in order of earliest to latest
                        layers in the network.  The first layer will be activated using the input,
                        and the last layer's output will be the network's final output.
        """
        
        self.layers = layers
        
        # Initialize the cost function and it's derivative
        self.cost_func = costFunctions.wrong_cost_func
        self.d_cost_func = costFunctions.wrong_cost_func
        self.cost_func_index = -1
    
    def compile(self, cost_function: int, optimizer: OptimizerForNetworks = optimizers.AdamForNetworks()):
        """Gets the model ready to use by:
        Setting the cost function and it's corresponding derivative function
        Associating the layers together
        
        ex: model.compile(cost=neuralNetwork.least_squares)"""
        
        self.cost_func, self.d_cost_func = costFunctions.get_cost_functions(cost_function)
        self.cost_func_index = cost_function
        
        # Compile the layers
        for i in range(0, len(self.layers)):
            self.layers[i].compile(copy.deepcopy(optimizer))
    
    def train(self, x, y, epochs: int = 30, batch_size: int = 32, prints_friendly=True):
        """Runs batches of data points through the network forwards, find the error and gradients coming backwards,
        and applies those gradients to the weights and biases of the network.  Then repeats (epochs) number of times.
        Returns the array of costs (one for each epoch of training)."""
        
        # Record when this method started
        start_time = time.time()
        
        # Let all layers know that it is train time
        for layer in self.layers:
            layer.set_is_test_time(False)
        
        # Initialize the random generator (for batch selection), and store the size of the data input and the possible
        #   indices from 0 to the total_size
        rng = np.random.default_rng()
        index_range = range(0, len(x))
        costs = []
        # Loop through for each epoch
        for _ in range(epochs):
            # Prep each batch and initialize the running tallies for the gradients and biases
            batch = rng.choice(index_range, size=batch_size)  # Creates a random list of indices, not actual data points
            self.propagate(x[batch], y[batch])
            costs.append(self.get_cost(x, y))
            if not prints_friendly:
                print(costs[-1])
        
        # Calculate the cost of this model with the data set
        cost = self.get_cost(x, y)

        # Let all layers know that it is no longer train time
        for layer in self.layers:
            layer.set_is_test_time(True)
        
        if prints_friendly:
            # Print out how everything went
            print('\nTraining done!')
            print('Final cost:', ceil(cost))
            print('Epochs:', epochs)
            print('Batch Size:', batch_size)
            print('Time to train: %.1f seconds' % (time.time() - start_time), "\n")
        
        return costs
    
    def evaluate(self, x, y, batch_size: int = 32):
        
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
        
        # ----- Feed Forward Stage -----
        # Loop through the layers forwards and activate each one based off of the previous activation
        last_a = x
        for layer in self.layers:
            last_a = layer.activate(last_a)
        
        # ----- Back Propagation Stage -----
        # Initialize the running error variable 'delta'
        delta = self.d_cost_func(last_a, y.reshape(-1, 1))
        for l in range(1, len(self.layers)+1):
            delta = self.layers[-l].back_propagate(delta)
    
    def predict(self, x):
        """Returns the prediction from this network for a single data point.  Does not calculate or
        apply any gradients"""
        
        # This variable name is word play.  At the end, it is returned as the last (final) activation,
        # but in the loop, it is the last (previous) activation haha
        last_a = x
        for layer in self.layers:
            last_a = layer.activate(last_a)
        return last_a
    
    def get_cost(self, x, y):
        """Calculates the current cost of the model by predicting each point, and calculating the
        cost for each and summing them.  Expects a 2d array of data for the x parameter, so either
        pass in multiple points, or wrap the one point in another set of parenthesis."""
        
        total_cost = 0
        data_count = len(x)
        for i in range(data_count):
            total_cost += self.cost_func(self.predict(x[i]), y[i])
        return np.sum(total_cost)
    
    def to_json(self):
        final_output = {
            "class_name": "LayeredNetwork",
            "layers": [layer.to_json() for layer in self.layers],
            "cost_func_index": self.cost_func_index
        }
        return final_output


def build_from_json(json_obj):
    if json_obj['class_name'] != "LayeredNetwork":
        print("LayeredNetwork.build_from_json error: Model JSON not a LayeredNetwork!  Found",
              json_obj['class_name'])
    ln = LayeredNetwork(
            layers=[networkLayerDictionary.get_layer_from_json(layer_json) for layer_json in json_obj['layers']]
    )
    # Recreating compile logic here without compiling layers
    ln.cost_func, ln.d_cost_func = costFunctions.get_cost_functions(json_obj['cost_func_index'])
    ln.cost_func_index = json_obj['cost_func_index']
    
    return ln
