# Author: Matthew Williams
# Start Date: 7/18/2023
# Touched up: 10/2/23

import numpy as np
from util.layer.networkLayer import Layer
from util import optimizers


"""
Notes about using Batch Norm:

When using this as a layer in the network, it seems to like batch sizes of at least 48, and not as high
as 128 right now.  Also, it's NOT recommended to use drop-out in pair with batch norm.

"""


class BatchNormLayer(Layer):
    """A Batch Norm Layer is a layer whose only function is to normalize batches upon activation,
    and to pass through gradients on through the network, while learning the best way to normalize
    and scale the data it receives.  This acts as a normalization AFTER the previous activation
    function.  To implement batch normalization before an activation function, use a BatchNormDenseLayer.
    If you're not sure which to use, use BatchNormDenseLayer (most experts recommend normalizing before
    activation functions as opposed to after)."""
    
    def __init__(self, layer_size: int, epsilon: float = 0.00001, optimizer: optimizers.OptimizerForNetworks = None,
                 momentum=0.97):
        """Initializes a Batch Normalization Layer.  This does not apply any kind of activation function
        or other operations other than batch normalization."""
        
        super().__init__()
        
        self.is_test_time = False
        
        self.layer_size = layer_size
        self.epsilon = epsilon
        self.optimizer = optimizer
        
        self.running_avg = None
        self.running_std_dev = None
        self.momentum = momentum
        self.computation_cache = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        
        # Initialize these values to maintain the norm distribution, but with allowance to change
        self.gamma = np.ones(layer_size)
        self.beta = np.zeros(layer_size)
    
    def set_test_time(self, is_test_time: bool):
        """Updates whether the model is in training mode (is_test_time=False), or test mode (is_test_time=True).
        During test time, the stored exponential moving average of the average and standard deviation is used,
        rather than calculating them for the given batch."""
        
        self.is_test_time = is_test_time

    def compile(self, optimizer: optimizers.OptimizerForNetworks):
        """
        Stores the optimizer in this layer, if no optimizer was given in the constructor.

        @param optimizer: The class that will apply gradient updates to this layers.  See util.optimizers.py
                            If an optimizer is specified in the constructor, this value will be ignored.
        """
    
        # If no optimizer was given in the constructor, fill in with the network's optimizer
        if self.optimizer is None:
            self.optimizer = optimizer
    
    def activate(self, _input):
        """Takes in a batch of inputs, and normalizes them.  Takes in the average and standard deviation
        for test-time performance, and returns the normalized version of the input"""
        
        # Pick what avg and std_dev values to use
        if self.is_test_time and self.running_avg is not None and self.running_std_dev is not None:
            norm = (_input - self.running_avg) / self.running_std_dev
            print("using stored values")
        else:
            avg = np.sum(_input, axis=0) / _input.shape[0]
            x_Mu = _input - avg
            sq = np.square(x_Mu)
            std_dev_squared = np.sum(sq, axis=0) / _input.shape[0] + self.epsilon
            std_dev = np.sqrt(std_dev_squared)
            idev = 1. / std_dev
            norm = x_Mu * idev
            
            # If we're building off current batch, ingest that info for later
            self.__ingest_batch_values(avg, std_dev)
            self.computation_cache = (sq, std_dev_squared, x_Mu, idev, norm)
        
        # Return the scaled version
        return self.gamma * norm + self.beta
        
    def back_propagate(self, delta):
        # Source for math:
        # kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        
        sq,std_dev_squared,x_Mu,idev,norm = self.computation_cache
        
        gamma_gradient = np.sum(delta * norm, axis=0)
        beta_gradient = np.sum(delta, axis=0)
        
        # TODO: Compile this next section and look for more optimized solutions
        
        # This is wrt the pre-scaling output (normalized input)
        delta = delta * self.gamma
        
        # ivar is the inverted standard deviation
        d_idev = np.sum(delta * x_Mu, axis=0)
        # x_Mu1 is the input minus the average, part 1
        d_x_Mu1 = delta * idev
        
        # sqrt_dev is the standard deviation, before inverting it
        d_sqrt_dev = d_idev * (-1. / std_dev_squared)
        
        # About to start on step 5 in back-prop
        dvar = 0.5 * idev * d_sqrt_dev
        
        # Step 4
        d_sq = (np.ones(sq.shape) / sq.shape[0]) * dvar
        
        # step 3
        d_x_Mu2 = 2 * x_Mu * d_sq
        
        # step 2
        d_x1 = d_x_Mu1 + d_x_Mu2
        d_Mu = -np.sum(d_x1, axis=0)
        
        # step 1
        d_x2 = (np.ones(sq.shape) / sq.shape[0]) * d_Mu
        
        # step 0
        dx = d_x1 + d_x2
        
        # TODO: Optimize above this
        
        # Now that we're done, update, then return
        self.optimizer.update(self.gamma, self.beta, gamma_gradient, beta_gradient)
        
        return dx
    
    def __ingest_batch_values(self, avg, std_dev):
        """Records the given average and std_dev for later statistical analysis"""
        
        if self.running_avg is None:
            self.running_avg = np.zeros(avg.shape)
            self.running_std_dev = np.zeros(std_dev.shape)
    
        self.running_avg -= (self.running_avg - avg) * (1 - self.momentum)
        self.running_std_dev -= (self.running_std_dev - std_dev) * (1 - self.momentum)
    
    def to_json(self):
        final_output = {
            "class_name": "BatchNormLayer",
            "layer_size": self.layer_size,
            "epsilon": self.epsilon,
            "optimizer": self.optimizer.to_json(),
            "momentum": self.momentum,
            "gamma": self.gamma.tolist(),
            "beta": self.beta.tolist()
        }
        
        return final_output


def build_from_json(json_obj):
    if json_obj['class_name'] != "BatchNormLayer":
        print("BatchNormLayer.build_from_json error: Model JSON not a BatchNormLayer!  Found",
              json_obj['class_name'])
    
    bnl = BatchNormLayer(
            layer_size=json_obj['layer_size'],
            epsilon=json_obj['epsilon'],
            optimizer=optimizers.get_optimizer_from_json(json_obj['optimizer']),
            momentum=json_obj['momentum']
    )
    bnl.gamma = np.array(json_obj['gamma'])
    bnl.beta = np.array(json_obj['beta'])
    
    return bnl
