# Author: Matthew Williams
# Start Date: 7/11/2023


class Layer:
    
    def __init__(self):
        """
        Initializes this Layer.  The Layer class itself is meant only to be extended.
        For this purpose, the definition of a Layer is something that can be activated,
        and can have derivatives propagated through it.
        """
        pass
    
    def set_is_test_time(self, is_test_time: bool):
        """Notifies layer that the model is now in testing mode.  This affects some layers, though
        many will ignore this call."""
        pass

    def compile(self, optimizer):
        """
        Passes the network default optimizer to this layer, if no other optimizer was supplied
        in the constructor.
        
        @param optimizer: The class that will apply gradient updates to this layers.  See util.optimizers.py
                            If an optimizer is supplied in the constructor, this value will be ignored.
        """
        pass
    
    def activate(self, _input):
        """
        Takes in the given input, and activates this layer with it.  Returns the
        activation of this layer, to serve as the input for the next layer (or as the
        network's final output, depending on the layer's position in the network).
        
        @param _input: nd array.  The previous layer's activation, for use in this layer.
        """
        pass
    
    def back_propagate(self, delta):
        """
        Back-propagates the given derivative through this layer, and returns the
        running derivative value 'delta'.
        
        @param delta: nd array.  The running derivative value(s) from the previous layer's
                back-propagation.
        
        @return delta: The derivative of the cost function of the network, with respect
                to the layer's input.
        """
        pass
    
    def to_json(self):
        """Returns a python dictionary representing a JSON object"""
        pass
    