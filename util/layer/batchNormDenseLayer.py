# Author: Matthew Williams
# Start Date: 7/11/2023

import numpy as np
import copy
from util.initializers import random_initializer, zero_initializer
from util.layer.networkLayer import Layer
import util.activationFunctions as af
from util import optimizers
from util.layer import batchNormLayer


class BatchNormDenseLayer(Layer):
    """Batch Norm Dense Layers are Dense Layers that integrate Batch Normalization
    immediately BEFORE the activation of the layer.  For post-activation batch norm,
    use a BatchNorm layer immediately after the desired layer."""
    
    def __init__(self, input_size: int, output_size: int, activation_function: int = af.leaky_relu,
                 optimizer: type(optimizers.OptimizerForNetworks) = None, initializer=random_initializer):
        """
        Initializes a layer for use in a LayeredNetwork.  Batch Norm Dense layers are fully connected layers
        that implement Batch Normalization immediately BEFORE the activation function. (Between the z and a)

        @param input_size: The size of the output of the previous layer.
        @param output_size: The size of this layer's desired output.
        @param activation_function: The function that should activate this function.  See util.activationFunctions.py
        @param optimizer: The class that will apply gradient updates to this layers.  See util.optimizers.py
                            If no optimizer is given, the network default is applied through the compile() method.
        @param initializer: The mode of initialization for this layer's weights and biases.  See util.initializers.py
        """
        
        super().__init__()
        
        self.w = initializer(output_size, input_size)
        self.b = initializer(output_size)
        
        # Get the activation function, and that function's derivative from the util class
        (activation_func, derivative_func) = af.get_activation_functions(activation_function)
        self.act_func = activation_func
        self.d_act_func = derivative_func
        # Save the original index for json saving
        self.act_func_index = activation_function
        
        # Optimizer choice given to allow specific layers to have different ones
        self.optimizer = None if optimizer is None else optimizer()
        
        # These values are stored each time activate() is called, and are utilized for
        # the gradient calculation in the back_propagate() method.
        self.last_input = None
        self.last_z = None
        
        # This layer stores it's own 'internal' layer to utilize the batch norm layer's logic, rather than
        # writing it again.
        self.batch_norm_layer = batchNormLayer.BatchNormLayer(output_size)
    
    def compile(self, optimizer: optimizers.OptimizerForNetworks):
        """
        Stores the optimizer in this layer, if no optimizer was given in the constructor.

        @param optimizer: The class that will apply gradient updates to this layers.  See util.optimizers.py
                            If an optimizer is specified in the constructor, this value will be ignored.
        """
        
        # If no optimizer was given in the constructor, fill in with the network's optimizer
        if self.optimizer is None:
            self.optimizer = optimizer
        self.batch_norm_layer.compile(copy.deepcopy(self.optimizer))
    
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
        
        # Run it through the batch norm layer before activating
        pre_act_output = self.batch_norm_layer.activate(self.last_z)
        
        return self.act_func(pre_act_output)
    
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
        
        # Propagate delta to this layer (dc/da) -> (dc/dbn) (wrt the batch_norm output)
        delta = delta * self.d_act_func(self.last_z)

        # Run the gradient through the batch_norm layer.
        # This will return (dc/dz), which is also (dc/db) (wrt the z and bias)
        delta = self.batch_norm_layer.back_propagate(delta)
        
        # Calculate the gradients
        weight_gradient = delta.T.dot(self.last_input)
        bias_gradient = np.sum(delta, axis=0)
        
        # Propagate delta BEFORE altering self.w in the optimizer.update below (dc/dz) -> (dc/d_input) (wrt prev input)
        delta = delta.dot(self.w)
        
        # Apply the gradient to the weights and biases
        self.optimizer.update(self.w, self.b, weight_gradient, bias_gradient)
        
        # Return delta so the next layer can utilize it
        return delta
    
    def to_json(self):
        final_output = {
            "class_name": "BatchNormDenseLayer",
            "w": self.w.tolist(),
            "b": self.b.tolist(),
            "act_func_index": self.act_func_index,
            "optimizer": self.optimizer.to_json(),
            "batch_norm_layer": self.batch_norm_layer.to_json()
        }
        return final_output


def build_from_json(json_obj):
    if json_obj['class_name'] != "BatchNormDenseLayer":
        print("BatchNormDenseLayer.build_from_json error: Model JSON not a BatchNormDenseLayer!  Found",
              json_obj['class_name'])
    
    bdl = BatchNormDenseLayer(
            input_size=2, output_size=2,
            activation_function=json_obj['act_func_index'],
            initializer=zero_initializer
    )
    bdl.w = np.array(json_obj['w'])
    bdl.b = np.array(json_obj['b'])
    bdl.compile(
            optimizer=optimizers.get_optimizer_from_json(json_obj['optimizer'])
    )
    bdl.batch_norm_layer = batchNormLayer.build_from_json(json_obj['batch_norm_layer'])
    
    return bdl
