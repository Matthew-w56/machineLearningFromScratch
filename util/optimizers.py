# Author: Matthew Williams

import numpy as np

# TODO: Document this

# ----------[ Base Abstract Classes ]----------


class Optimizer:
    def __init__(self):
        pass
    
    def update(self, weight, gradient):
        pass
    
    def to_json(self):
        pass
        

class OptimizerForNetworks:
    def __init__(self):
        pass
    
    def update(self, weights, biases, w_gradients, c_gradients):
        pass
    
    def to_json(self):
        pass


# ----------[ End Abstract Classes ]----------


class DoNotOptimize(Optimizer, OptimizerForNetworks):
    def __init(self):
        pass
    
    def update(self, weight, gradient, w_gradients=None, c_gradients=None):
        pass
    
    def to_json(self):
        return {"class_name": "DoNotOptimize"}
    

class VanillaGradientDescent(Optimizer):
    """Optimizes by simply applying a learn rate to the gradient, then applying that to the weights given."""
    
    def __init__(self, learn_rate=0.0001):
        """Build the optimizer with the learn rate."""

        super().__init__()
        self.learn_rate = learn_rate
    
    def update(self, weight, gradient):
        """Applies the gradient to the weights after multiplying by the learn rate.  Simply as can be.

        Parameters

        -weight: ndArray.  The weight(s) to be adjusted by the optimizer.

        -gradient: ndArray.  The gradient(s) to apply to the weight(s)."""
        
        # Simply apply the gradient multiplied by the learn rate
        weight -= self.learn_rate * gradient
    
    def to_json(self):
        return {
            "class_name": "VanillaGradientDescent",
            "learn_rate": self.learn_rate
        }


class Adagrad(Optimizer):
    """Optimizes with the Adagrad technique.
    
    The object keeps it's own state matrix and adds the square of each gradient given to it into the state.
    Then, when applying the gradient, it divides the gradient by the square root of the state before applying it
    (plus a small epsilon to avoid a divide by zero error) to the weights.
    
    Note: This model will not work for any Network model (because of it's layers).  For that, see
    optimizers.AdagradForNetworks
    """
    
    def __init__(self, epsilon=0.00000001):
        """Builds the Adagrad optimizer with the epsilon and a blank state.
        
        Parameters
        
        -epsilon: float.  A small value to be added to the state before division to avoid a divide by zero error."""

        super().__init__()
        self.epsilon = epsilon
        self.s = None
    
    def update(self, weight, gradient):
        """Applies the gradient using the Adagrad technique.
        
        Parameters
        
        -weight: ndArray.  The weight(s) to be adjusted by the optimizer.
        
        -gradient: ndArray.  The gradient(s) to apply to the weight(s)."""
        
        # Calculate the State factor (or set it, if it is currently unset)
        if self.s is None:
            self.s = gradient * gradient
        else:
            self.s += gradient * gradient
        
        # Apply the S factor
        gradient /= np.sqrt(self.s) + self.epsilon
        
        # Apply the gradient
        weight -= gradient
    
    def to_json(self):
        return {
            "class_name": "Adagrad",
            "epsilon": self.epsilon,
            "s": None if self.s is None else self.s.tolist()
        }


class AdagradForNetworks(OptimizerForNetworks):
    """Acts just like the Adagrad optimizer, but keeps track of multiple layers.
    
    Separate 's' matrices are kept for each layer.
    
    See optimizers.Adagrad"""
    
    def __init__(self, epsilon=0.00001):
        super().__init__()
        self.epsilon = epsilon
        # S is the idea, sw is for weights, sc is for biases (c in code)
        self.sw = None
        self.sc = None
    
    def update(self, weights, biases, w_gradients, c_gradients):
        """Mimics the Adagrad.update() method, but does so for each implicit layer of a network.
        
        Note: This will break if given a model with no layers, but will not break with a 1-layer network.
        With some updates for efficiency, it is effectively:
        
        for layer in (layers in model):
            Adagrad.update()"""
        
        # Steps of operation:
        # Step 1)  Establish S exists, and update it's values
        # Step 2)  Go through each layer's W and apply it's gradient
        # Step 3)  Go through each layer's C and apply it's gradient
        
        # Step 1)
        # If the 's' isn't set yet, make a separate 's' for each layer
        if self.sw is None or self.sc is None:
            # Set 's' to a new list comprised of the squares of each layer's gradients
            self.sw = np.array([g * g for g in w_gradients])
            self.sc = np.array([g * g for g in c_gradients])
        else:
            # Add the square of each layer's gradients to the corresponding layer of 's'
            for s,g in zip(self.sw, w_gradients):
                s += (g * g)
            for s,g in zip(self.sc, c_gradients):
                s += (g * g)
        
        # Step 2)
        # Loop through each layer's (weight, gradient, and 's')
        for w,g,s in zip(weights, w_gradients, self.sw):
            # Apply the 's' factor to the gradient
            g_divided = g / (np.sqrt(s) + self.epsilon)
            # Apply the gradient to the weight
            w -= g_divided
        
        # Step 3)
        # Loop through each layer's (biases, bias gradients, and 's')
        for b,g,s in zip(biases, c_gradients, self.sc):
            # Apply the 's' factor to the gradient
            g_divided = g / (np.sqrt(s) + self.epsilon)
            # Apply the gradient to the bias
            b -= g_divided
    
    def to_json(self):
        return {
            "class_name": "AdagradForNetworks",
            "epsilon": self.epsilon,
            "sw": None if self.sw is None else self.sw.tolist(),
            "sc": None if self.sc is None else self.sc.tolist()
        }


class Momentum(Optimizer):
    """Optimizes using Gradient Descent with Momentum.

    The object has a momentum that is diminished by being multiplied by self.gamma (which lies on [0, 1)) each time the
    gradient is being applied.  A higher gamma means the momentum lasts longer and is stronger, and a lower gamma
    behaves more like Vanilla Gradient Descent.
    """
    
    def __init__(self, gamma=0.9, learn_rate=0.01, epsilon=0.00001):
        """Builds the Momentum optimizer with the gamma, and other inputs
        
        
        Parameters
        
        -gamma: float [0, 1).  Factor by which the previous running total of gradients (momentum) is multiplied by.
        
        -learn_rate: float.  The learning rate by which the model learns.  Should stay quite small.
        
        -epsilon: float.  A small value to be added to the state before division to avoid a divide by zero error."""

        super().__init__()
        self.gamma = gamma
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        self.s = None
    
    def update(self, weight, gradient):
        """Applies the gradient using Gradient Descent with Momentum.

        Parameters

        -weight: ndArray.  The weight(s) to be adjusted by the optimizer.

        -gradient: ndArray.  The gradient(s) to apply to the weight(s)."""
        
        # Set the momentum or Update the momentum
        if self.s is None:
            self.s = gradient * self.gamma
        else:
            self.s = (self.s * self.gamma) + gradient
        
        # Update the weight with the learning rate and gradient
        weight -= self.learn_rate * self.s
    
    def to_json(self):
        return {
            "class_name": "Momentum",
            "gamma": self.gamma,
            "learn_rate": self.learn_rate,
            "epsilon": self.epsilon,
            "s": None if self.s is None else self.s.tolist()
        }


class RMSProp(Optimizer):
    """Optimizes using the RMSProp technique.
    
    RMSProp uses a running tally of the squares of the gradients just like Adagrad, but each
    time, the existing tally is multiplied by 'beta', a float between 0 and 1.  This creates a
    diminishing effect, so if the gradient stays small for some time, the divisor also shrinks.
    Recommended value for beta is 0.9"""
    
    def __init__(self, beta=0.9, learn_rate=1, epsilon=0.00000001):
        """Builds the RMSProp optimizer.
        
        Parameters:
        
        -beta: float (0,1).  Determines how much the previous gradient aggregation (from Adagrad)
        diminishes.  closer to 0 means it diminishes quickly, and closer to 1 means that it is slower.
        
        -learn_rate: float.  Determines how quickly the descent begins, before effects from the divisor"""

        super().__init__()
        self.s = None
        self.beta = beta
        self.learn_rate = learn_rate
        self.epsilon = epsilon
    
    def update(self, weight, gradient):
        """Updates the gradient much like Adagrad does, but diminishes the existing S vector each time, allowing
        it to act more as an exponentially weighted average than a sum tally.
        
        Parameters

        -weight: ndArray.  The weight(s) to be adjusted by the optimizer.

        -gradient: ndArray.  The gradient(s) to apply to the weight(s)."""
        
        # Calculate the State factor (or set it, if it is currently unset)
        if self.s is None:
            self.s = gradient * gradient * self.beta
        else:
            self.s = self.s * self.beta  +  gradient * gradient * (1-self.beta)
        
        # Apply the S factor
        gradient /= np.sqrt(self.s + self.epsilon)
        
        # Apply the gradient
        weight -= gradient * self.learn_rate
    
    def to_json(self):
        return {
            "class_name": "RMSProp",
            "s": None if self.s is None else self.s.tolist(),
            "beta": self.beta,
            "learn_rate": self.learn_rate,
            "epsilon": self.epsilon
        }


class RMSPropForNetworks(OptimizerForNetworks):
    """Optimizes using the RMSProp technique.

    RMSProp uses a running tally of the squares of the gradients just like Adagrad, but each
    time, the existing tally is multiplied by 'beta', a float between 0 and 1.  This creates a
    diminishing effect, so if the gradient stays small for some time, the divisor also shrinks.
    Recommended value for beta is 0.9"""
    
    def __init__(self, beta=0.9, learn_rate=1, epsilon=0.00000001):
        """
        Builds the RMSProp optimizer.  This is different than the other RMSProp in that the
        gradients and weights are understood to be 3-d matrices (or arrays of 2d arrays), making
        this work for Neural Networks as opposed to less complicated models where gradients and
        weights are kept in a one, or two dimensional data structure.

        Parameters:

        -beta: float (0,1).  Determines how much the previous gradient aggregation (from Adagrad)
        diminishes.  closer to 0 means it diminishes quickly, and closer to 1 means that it is slower.

        -learn_rate: float.  Determines how quickly the descent begins, before effects from the divisor"""

        super().__init__()
        self.beta = beta
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        # S is the idea, sw is for weights, sc is for biases (c in code)
        self.sw = None
        self.sc = None
    
    def update(self, weights, biases, w_gradients, c_gradients):
        """
        Updates the gradient much like Adagrad does, but diminishes the existing S vector each time, allowing
        it to act more as an exponentially weighted average than a sum tally.

        Parameters

        -weight: 3D Array.  The weight(s) to be adjusted by the optimizer.

        -gradient: 3D Array.  The gradient(s) to apply to the weight(s)."""
        
        # Steps of operation:
        # Step 1)  Establish S exists, and update it's values
        # Step 2)  Go through each layer's W and apply it's gradient
        # Step 3)  Go through each layer's C and apply it's gradient
        
        # Step 1)
        # If the 's' isn't set yet, make a separate 's' for each layer
        if self.sw is None or self.sc is None:
            # Set 's' to a new list comprised of the squares of each layer's gradients
            self.sw = np.array([g * g * self.beta for g in w_gradients])
            self.sc = np.array([g * g * self.beta for g in c_gradients])
        else:
            # Add the square of each layer's gradients to the corresponding layer of 's'
            for s, g in zip(self.sw, w_gradients):
                s *= self.beta
                s += g * g * (1-self.beta)
            for s, g in zip(self.sc, c_gradients):
                s *= self.beta
                s += g * g * (1 - self.beta)
        
        # Step 2)
        # Loop through each layer's (weight, gradient, and 's')
        for w, g, s in zip(weights, w_gradients, self.sw):
            # Apply the 's' factor to the gradient
            g_divided = g / (np.sqrt(s) + self.epsilon)
            # Apply the gradient to the weight
            w -= g_divided
        
        # Step 3)
        # Loop through each layer's (biases, bias gradients, and 's')
        for b, g, s in zip(biases, c_gradients, self.sc):
            # Apply the 's' factor to the gradient
            g_divided = g / (np.sqrt(s) + self.epsilon)
            # Apply the gradient to the bias
            b -= g_divided
    
    def to_json(self):
        return {
            "class_name": "RMSPropForNetworks",
            "beta": self.beta,
            "learn_rate": self.learn_rate,
            "epsilon": self.epsilon,
            "sw": None if self.sw is None else self.sw.tolist(),
            "sc": None if self.sc is None else self.sc.tolist()
        }


class Adam(Optimizer):
    """Optimizes the model's weights with the Adam algorithm.
    
    Adam combines the momentum tracker of Momentum, and the adaptive restrictor of RMSProp.  Both
    are kept by exponential moving averages and are controlled by beta_v and beta_s respectively."""
    
    def __init__(self, beta_v=0.9, beta_s=0.999, epsilon=0.00000001, initial_lr=0.001):
        """Builds the Adam Optimizer.
        
        
        Parameters:
        
        -beta_v: float (0,1).  Determines how much previous gradients affect the current momentum.  Higher beta_v means
        a "rolling ball" with more momentum (higher "inertia").
        
        -beta_s: float (0,1).  Determines how much previous gradients affect the current RMSProp divisor.  Higher beta_s
        means that the ball slows down faster, while a lower beta_s means it maintains speed further into the descent.
        
        -epsilon: float.  Tiny number added to the square root of the beta_s variable to avoid a divide by zero error.
        
        -initial_lr: float.  Initial Learn Rate.  Determines the starting learn rate that the optimizer uses.
        """
        
        # Store parameters in state variables
        super().__init__()
        self.beta_v = beta_v
        self.beta_s = beta_s
        self.epsilon = epsilon
        self.learn_rate = initial_lr
        
        # Initialize the V and S vectors as None
        self.v = None
        self.s = None
    
    def update(self, weight, gradient):
        """Applies the gradient using the Adam technique.

        Parameters

        -weight: ndArray.  The weight(s) to be adjusted by the optimizer.

        -gradient: ndArray.  The gradient(s) to apply to the weight(s)."""
        
        # Set / Update the Moment vectors (v is the momentum component, s is the Adagrad/RMS_Prop component)
        if self.v is None or self.s is None:
            self.v = gradient * (1 - self.beta_v)
            self.s = (1 - self.beta_s) * (gradient * gradient)
        else:
            self.v = self.beta_v * self.v + (1 - self.beta_v) * gradient
            self.s = self.beta_s * self.s + (1 - self.beta_s) * (gradient * gradient)
        
        weight -= self.learn_rate * ( self.v / (np.sqrt(self.s) + self.epsilon) )

    def to_json(self):
        return {
            "class_name": "Adam",
            "beta_v": self.beta_v,
            "beta_s": self.beta_s,
            "epsilon": self.epsilon,
            "learn_rate": self.learn_rate,
            "v": None if self.v is None else self.v.tolist(),
            "s": None if self.s is None else self.s.tolist()
        }


class AdamForNetworks(OptimizerForNetworks):
    
    def __init__(self, beta_v=0.9, beta_s=0.999, epsilon=0.00000001, initial_lr=0.001):
        """Builds the Adam Optimizer.


        Parameters:

        -beta_v: float (0,1).  Determines how much previous gradients affect the current momentum.  Higher beta_v means
        a "rolling ball" with more momentum (higher "inertia").

        -beta_s: float (0,1).  Determines how much previous gradients affect the current RMSProp divisor.  Higher beta_s
        means that the ball slows down faster, while a lower beta_s means it maintains speed further into the descent.

        -epsilon: float.  Tiny number added to the square root of the beta_s variable to avoid a divide by zero error.

        -initial_lr: float.  Initial Learn Rate.  Determines the starting learn rate that the optimizer uses.
        """
        
        # Store parameters in state variables
        super().__init__()
        self.beta_v = beta_v
        self.beta_s = beta_s
        self.epsilon = epsilon
        self.learn_rate = initial_lr
        
        # Initialize the V and S vectors as None (For the weights and biases each)
        self.vw = None
        self.vc = None
        self.sw = None
        self.sc = None
    
    def update(self, weights, biases, w_gradients, c_gradients):
        """Applies the gradient using the Adam technique.

        Parameters

        -weight: ndArray.  The weight(s) to be adjusted by the optimizer.

        -gradient: ndArray.  The gradient(s) to apply to the weight(s)."""
        
        # Steps of operation:
        # Step 1)  Establish V and S exist, and update their values
        # Step 2)  Go through each layer's W and apply it's gradient
        # Step 3)  Go through each layer's C and apply it's gradient
        
        # Step 1)
        # If any of the 's's aren't set, set them.
        # (Assumption: one being set is mutually inclusive of all others being set, and vice versa)
        if self.sw is None:
            # Set V and S to a new list comprised of the squares of each layer's gradients
            self.vw = np.array([(1 - self.beta_v) * g for g in w_gradients])
            self.vc = np.array([(1 - self.beta_v) * g for g in c_gradients])
            self.sw = np.array([(1 - self.beta_s) * (g * g) for g in w_gradients])
            self.sc = np.array([(1 - self.beta_s) * (g * g) for g in c_gradients])
        else:
            # Add the square of each layer's gradients to the corresponding layer of 'v'
            for v, g in zip(self.vw, w_gradients):
                v *= self.beta_v
                v += (1 - self.beta_v) * g
            for v, g in zip(self.vc, c_gradients):
                v *= self.beta_v
                v += (1 - self.beta_v) * g
            # Add the square of each layer's gradients to the corresponding layer of 's'
            for s, g in zip(self.sw, w_gradients):
                s *= self.beta_s
                s += (1 - self.beta_s) * (g * g)
            for s, g in zip(self.sc, c_gradients):
                s *= self.beta_s
                s += (1 - self.beta_s) * (g * g)
        
        # Step 2)
        # Loop through each layer's (weight, gradient, and V and S)
        for w, g, s, v in zip(weights, w_gradients, self.sw, self.vw):
            # Apply the momentum and RMSProp items to the weights
            w -= self.learn_rate * (v / (np.sqrt(s) + self.epsilon))
        
        # Step 3)
        # Loop through each layer's (biases, bias gradients, and 's')
        for b, g, s, v in zip(biases, c_gradients, self.sc, self.vc):
            # Apply the momentum and RMSProp items to the biases
            b -= self.learn_rate * (v / (np.sqrt(s) + self.epsilon))

    def to_json(self):
        return {
            "class_name": "AdamForNetworks",
            "beta_v": self.beta_v,
            "beta_s": self.beta_s,
            "epsilon": self.epsilon,
            "learn_rate": self.learn_rate,
            "vw": self.vw.tolist(),
            "vc": self.vc.tolist(),
            "sw": self.sw.tolist(),
            "sc": self.sc.tolist()
        }


class AdamWithCorrection(Optimizer):
    """Optimizes the model's weights with the Adam algorithm.

    Adam combines the momentum tracker of Momentum, and the adaptive restrictor of RMSProp.  Both
    are kept by exponential moving averages and are controlled by beta_v and beta_s respectively.
    
    The Bias Correction mentioned is an attempt to reduce the effect that the first gradient has on the
    exponential averages kept by self.s and self.v"""
    
    def __init__(self, beta_v=0.9, beta_s=0.999, epsilon=0.0000001, initial_lr=0.001):
        """Builds the Adam Optimizer with Bias Correction.


        Parameters:

        -beta_v: float (0,1).  Determines how much previous gradients affect the current momentum.  Higher beta_v means
        a "rolling ball" with more momentum (higher "inertia").

        -beta_s: float (0,1).  Determines how much previous gradients affect the current RMSProp divisor.  Higher beta_s
        means that the ball slows down faster, while a lower beta_s means it maintains speed further into the descent.

        -epsilon: float.  Tiny number added to the square root of the beta_s variable to avoid a divide by zero error.

        -initial_lr: float.  Initial Learn Rate.  Determines the starting learn rate that the optimizer uses.
        """

        super().__init__()
        self.beta_v = beta_v
        self.beta_s = beta_s
        self.epsilon = epsilon
        self.learn_rate = initial_lr
    
        self.v = None
        self.s = None
        self.t = 0.01
    
    def update(self, weight, gradient):
        """Applies the gradient using the Adam technique.
        
        Parameters
        
        -weight: ndArray.  The weight(s) to be adjusted by the optimizer.
        
        -gradient: ndArray.  The gradient(s) to apply to the weight(s)."""
        
        # Set / Update the Moment vectors (v is the momentum component, s is the Adagrad/RMS_Prop component)
        if self.v is None or self.s is None:
            self.v = gradient * (1 - self.beta_v)
            self.s = (1 - self.beta_s) * (gradient * gradient)
        else:
            self.v = self.beta_v * self.v + (1 - self.beta_v) * gradient
            self.s = self.beta_s * self.s + (1 - self.beta_s) * (gradient * gradient)
        
        # Bias Correction Bit
        self.t = (self.t + 1 if self.t > 0.5 else 1)
        s_hat = self.s / (1 - self.beta_s ** self.t)
        v_hat = self.v / (1 - self.beta_v ** self.t)
        
        weight -= self.learn_rate * (v_hat / (np.sqrt(s_hat) + self.epsilon))

    def to_json(self):
        return {
            "class_name": "AdamWithCorrection",
            "beta_v": self.beta_v,
            "beta_s": self.beta_s,
            "epsilon": self.epsilon,
            "learn_rate": self.learn_rate,
            "v": None if self.v is None else self.v.tolist(),
            "s": None if self.s is None else self.s.tolist(),
            "t": self.t
        }


# TODO: Make Adam with correction for networks


classes = {
    "DoNotOptimize": DoNotOptimize,
    "VanillaGradientDescent": VanillaGradientDescent,
    "Adagrad": Adagrad,
    "AdagradForNetworks": AdagradForNetworks,
    "Momentum": Momentum,
    "RMSProp": RMSProp,
    "RMSPropForNetworks": RMSPropForNetworks,
    "Adam": Adam,
    "AdamForNetworks": AdamForNetworks,
    "AdamWithCorrection": AdamWithCorrection
}


def get_optimizer_from_json(json_obj):
    """Takes in a dictionary representing a json object that
    stores an optimizer.  Finds, instantiates, and returns the
    correct object in the correct class based off of an attribute
    in the json object called \"class_name\". """
    name = json_obj['class_name']
    optimizerClass = classes[name]
    opt = optimizerClass()
    for attr in json_obj:
        if type(json_obj[attr]) == list:
            setattr(opt, attr, np.array(json_obj[attr]))
        else:
            setattr(opt, attr, json_obj[attr])
    return opt


