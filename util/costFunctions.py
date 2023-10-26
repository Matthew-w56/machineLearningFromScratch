import numpy as np


def f_least_squares(y_hat, y):
    return ((y_hat - y) ** 2) / 2


def f_bin_cross_entropy(y_hat, y):
    if y == 1:
        return np.log2(1 - y_hat)
    return np.log2(y_hat)

# -----------------------------------------------------------


def d_least_squares(y_hat, y):
    return y_hat - y


def d_bin_cross_entropy(y_hat, y):
    return (-y / y_hat) + ((1-y) / (1-y_hat))

# -----------------------------------------------------------


def wrong_cost_func(y_hat, y):
    print(f'[WCF]  {y_hat} and {y} inputted to the cost function before it was changed from the default!')
    raise Exception('Must set network\'s cost function through the compile() method before attempting to use!')


least_squares = 0
bin_cross_entropy = 1

costFunctionLookup =   [f_least_squares, f_bin_cross_entropy]
costDerivativeLookup = [d_least_squares, d_bin_cross_entropy]


def get_cost_functions(index):
    return costFunctionLookup[index], costDerivativeLookup[index]

