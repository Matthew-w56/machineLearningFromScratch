import numpy as np

__leaky_coeff__ = 0.2


def f_relu(x):
    """Activation function that returns x back if it is more than 0, and 0 if x is negative"""
    if x.any() == np.nan:
        raise Exception('Nan in x!')
    return np.where(x > 0, x, 0)


def f_leaky_relu(x):
    if x.any() == np.nan:
        raise Exception("Nan in x!")
    return np.where(x >= 0, x, x * __leaky_coeff__)


def f_linear(x):
    """Linear 'activation function'.  Returns X"""
    
    return x


def f_sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # return np.piecewise(x, x < 0,
    #                    (lambda i: np.exp(i) / (1 + np.exp(i)), lambda i: 1 / (1 + np.exp(-i))))


def d_relu(x):
    """Derivative function of the ReLU activation.  Returns 1 if x > 0, otherwise 0"""
    
    return np.where(x > 0, 1, 0)


def d_leaky_relu(x):
    
    return np.where(x >= 0, 1, __leaky_coeff__)


def d_linear(x):
    """Returns the derivative of the linear activation function, which is 1."""
    return np.full(x.shape, 1)


def d_sigmoid(x):
    sig = f_sigmoid(x)
    return sig * (1 - sig)



relu = 0
linear = 1
sigmoid = 2
leaky_relu = 3

activationLookup = [f_relu, f_linear, f_sigmoid, f_leaky_relu]
derivativeLookup = [d_relu, d_linear, d_sigmoid, d_leaky_relu]


def get_activation_functions(index):
    return activationLookup[index], derivativeLookup[index]
