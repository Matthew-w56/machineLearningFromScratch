# Author: Matthew Williams
# Start Date: 7/11/2023

import numpy as np

from util.initializers import random_initializer, zero_initializer
from util.layer.networkLayer import Layer
import util.activationFunctions as af
from util import optimizers


class DenseLayer(Layer):
    """Dense layers are fully connected layers that do not implement
    any drop out, regularization, or other techniques."""
    
    def __init__(self, input_size: int, output_size: int, activation_function: int = af.leaky_relu,
                 optimizer: optimizers.OptimizerForNetworks = None, initializer=random_initializer):
        """
        Initializes a layer for use in a LayeredNetwork.  Dense layers are fully connected layers
        that do not implement any drop out, regularization, or other techniques.
        
        @param input_size: The size of the output of the previous layer.
        @param output_size: The size of this layer's desired output.
        @param activation_function: The function that should activate this function.  See util.activationFunctions.py
        @param optimizer: The class that will apply gradient updates to this layers.  See util.optimizers.py
                            If no optimizer is given, the network default is applied through the compile() method.
        @param initializer: The mode of initialization for this layer's weights and biases.  See util.initializers.py
        """
        
        super().__init__()
        
        # With transformation matrix A being nxm, input size is m and output size is n

        self.w = initializer(output_size, input_size)
        self.b = initializer(output_size)
        
        # Get the activation function, and that function's derivative from the util class
        (activation_func, derivative_func) = af.get_activation_functions(activation_function)
        self.act_func = activation_func
        self.d_act_func = derivative_func
        # Save the original index for json saving later
        self.act_func_index = activation_function
        
        # Optimizer choice given to allow specific layers to have different ones
        self.optimizer = optimizer
        
        # These values are stored each time activate() is called, and are utilized for
        # the gradient calculation in the back_propagate() method.
        self.last_input = None
        self.last_z = None
    
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
        """
        Activates this layer with the input given, and returns it's total activation (output).
        
        @param _input: The output of the previous layer (or the input to the network, if this
                        is the first layer of the network).
        @return: This layer's output, in the same format as the _input was given in, ideally.
        """
        
        self.last_input = _input
        # Notice how this follows the format y = mx + b (or here, xm + b)
        self.last_z = np.array(_input @ self.w.T + self.b)
        return self.act_func(self.last_z)
    
    def back_propagate(self, delta):
        """
        Runs the given derivative through this layer, applies the gradient to the weights and biases,
        then returns the running derivative to be used in the next layer down the network.
        
        @param delta: Running derivative variable.  Should represent the derivative of the network's
                        cost function with respect to the output given by this layer.
        @return: delta: Running derivative variable.  Represents the derivative of the network's cost
                        function with respect to the input given to this layer. (output of the previous
                        layer).
        """
        
        # Propagate delta to this layer
        delta = delta * self.d_act_func(self.last_z)
        
        # Calculate the gradients
        weight_gradient = delta.T.dot(self.last_input)
        bias_gradient = np.sum(delta, axis=0)
        
        # Propagate delta BEFORE altering self.w in the optimizer.update below
        delta = delta.dot(self.w)
        
        # Apply the gradient to the weights and biases
        self.optimizer.update(self.w, self.b, weight_gradient, bias_gradient)
        
        # Return delta so the next layer can utilize it
        return delta
    
    def to_json(self):
        final_output = {
            "class_name": "DenseLayer",
            "w": self.w.tolist(),
            "b": self.b.tolist(),
            "act_func_index": self.act_func_index,
            "optimizer": self.optimizer.to_json()
        }
        # Note: No need to store last_input or last_z
        
        return final_output


def build_from_json(json_obj):
    if json_obj['class_name'] != "DenseLayer":
        print("DenseLayer.build_from_json error: Model JSON not a DenseLayer!  Found", json_obj['class_name'])
        return None
    dl = DenseLayer(
                    input_size=int(2),
                    output_size=int(2),
                    activation_function=json_obj['act_func_index'],
                    initializer=zero_initializer)
    # Get the activation function, and that function's derivative from the util class
    (activation_func, derivative_func) = af.get_activation_functions(json_obj['act_func_index'])
    dl.act_func = activation_func
    dl.d_act_func = derivative_func
    dl.act_func_index = json_obj['act_func_index']
    dl.optimizer = optimizers.get_optimizer_from_json(json_obj['optimizer'])
    
    dl.w = np.array(json_obj['w'])
    dl.b = np.array(json_obj['b'])
    dl.compile(optimizers.get_optimizer_from_json(json_obj['optimizer']))
    
    return dl
