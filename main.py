# Author: Matthew Williams

import math
import random
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import optimizers
from models import decisionTree as dt
from models.legacyModels import simpleLinearRegression as slr
from models import linearRegression as lr
from models.legacyModels import binaryLogisticRegression as BlogR
from models import logisticRegression as logR
from models.legacyModels import vanillaNeuralNetwork as nn
from util import activationFunctions as af
from util import costFunctions as cf
import models.networks.layeredNetwork as ln
from util.layer.batchNormDenseLayer import BatchNormDenseLayer
from util.layer.denseLayer import DenseLayer


# ------------------[ Job Board ]---------------------------------------------------------------------------------------
# ------------------[ First Up ]-----------------------
# TODO: Add comments and doc comments to the Optimizers classes (Mainly RMSProp)
# ------------------[ Soon ]---------------------
# TODO: Add serialized model saving and loading
# ------------------[ Sometime ]-----------------------
# TODO: Optimize and look at Multiple Linear Regression, it is slow and I feel it can be more accurate
# TODO: Look into smarter ways to initialize weights and biases
# TODO: Implement random descent in a linear prediction setting
# TODO: Add error calculations to all logistic classes, rather than relying on solely accuracy
# TODO: Implement random descent in a univariate hyper-parameter setting (linear regression or threshold for logistic)
# TODO: Implement random descent in a multivariate hyper-parameter setting (the trees.)
# ----------------------------------------------------------------------------------------------------------------------


# ---------------------------------------- Matplotlib Helper Methods ---------------------------------------
def plot_pred_on_real(data_y, predictions, x_label=None, y_label=None):
    """Plots a graph where the X axis is the true values, and the Y is the predictions.

    Note:
      This is meant for continuous predictor models.  This does not translate to classifiers"""

    # Calculate where the x=y line should start and end for the graph's scope
    minimum = np.max((np.min(data_y), np.min(predictions)))
    maximum = np.min((np.max(data_y), np.max(predictions)))

    # Graph out the results
    plt.figure(figsize=(12, 8))
    plt.scatter(data_y, predictions, label="Point of [y,y^]", marker='x')
    plt.plot([minimum, maximum], [minimum, maximum], "r-", label="Line where y^=y")
    plt.grid()
    # Set the x and y label to either the default values, or the given ones
    if x_label is None:
        plt.xlabel('Real Value (Y)')
    else:
        plt.xlabel(x_label)
    if y_label is None:
        plt.ylabel('Predicted Value (Y^)')
    else:
        plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_error_density(data_y, predictions, num_bars, chop_last=0):
    """Plots a bar graph where each bar is a group of items and it's height is how many examples have a
    distance from the true value corresponding to the bar.

    Note:
       This actually plots the distance of a continuous guess from the true label, not the 'error' of it.
       As such, it does not display error in terms of the model, and it does not translate well to classifiers"""

    # Store a list of each data point's distance from the real value
    # and take the absolute value so that a distance of -1 and 1 is treated the same
    distances = abs(np.array(predictions - data_y))

    # Create a list to store each bar's intended height
    bar_data = []

    # Calculate the width (range of distances that fall into each bar) by dividing the
    # effective domain of errors (from 0 to the largest observed) by the desired number of bars
    bar_width = math.floor(np.max(distances) / num_bars)

    # Loop through each intended bar
    for i in range(num_bars):
        # Add to the list: the length of an array that contains only items that meet these requirements:
        # 1) The distance for this data point is smaller than the next bar marker ((i+1) * bar width)
        # 2) The distance for this data point is larger than the last bar marker (i * bar width)
        bar_data.append(len(distances[np.where((distances <= (i+1) * bar_width) & (distances > i * bar_width))]))

    # Crop the lists to what is wanted (take chop_last into account)
    bar_data = bar_data[0:len(bar_data)-chop_last]
    num_bars_shown = num_bars - chop_last

    # Get to the actual graphing, with Matplotlib
    # Here, set the size of the screen
    plt.figure(figsize=(12, 8))
    # Plot the bar
    plt.bar([i for i in range(num_bars_shown)], bar_data)
    plt.xticks(ticks=[i for i in range(num_bars_shown)],
               labels=[math.floor(bar_width * (i + 1)) for i in range(num_bars_shown)])
    plt.ylabel('Number of examples')
    plt.xlabel('Value distance from Predicted (<=X)')
    plt.title('Distribution of prediction distances from real values')
    plt.grid(axis="y")
    plt.show()


# ----------------------------------------- Other Helper Methods --------------------------------------
# Returns a subset of given sets chosen randomly with replacement
def bootstrap_sample(percentage, total_set_x, total_set_y):
    """Takes in a list of X's and Y's, with a percentage to choose, and randomly chooses
    floor(percentage * number of items in set)
    items randomly, with replacement, and returns the list of x and y data points, as a tuple (x, y)"""

    n = math.floor(len(total_set_x) * percentage)
    r = random.Random()
    final_x = []
    final_y = []
    max_i = len(total_set_x) - 1
    for i in range(n):
        r_n = r.randint(0, max_i)
        final_x.append(total_set_x[r_n])
        final_y.append(total_set_y[r_n])
    return np.array(final_x), np.array(final_y)


# Return two lists: first of the first 80 percent then second the last 2
def eighty_twenty_split(total_set):
    split_point = math.floor(len(total_set) * 0.8)
    return total_set[0:split_point], total_set[split_point:]


def data_normalize(data):
    return data / np.max(data)


def replace_all_dict(_list, _dict, default):
    for i in range(len(_list)):
        result = _dict.get(_list[i])
        if result is not None:
            _list[i] = result
        else:
            _list[i] = default


# -------------------------------------- Testing Methods --------------------------------------------
# Weather data (pred daily low from high)
def test_simple_linear():
    # Import the data for training and testing
    file = pd.read_csv('./datasets/WWII weather data/weather_clean.csv')
    data = np.array(file)

    # Split data into input and output (x and y)
    data_x = data[:80000, 0]
    data_y = data[:80000, 1]
    eval_x = data[80000:140000, 0]
    eval_y = data[80000:140000, 1]

    # Build the model
    model = slr.SimpleLinearRegression()

    # Train the model
    model.train(data_x, data_y)

    # Evaluate the model
    model.evaluate(eval_x, eval_y)

    # Store the model's predictions
    predictions = model.predict_all(eval_x)

    # plot_pred_on_real(eval_y, predictions, xlabel='Custom X Label', ylabel='Custom Y Label')
    plot_error_density(eval_y, predictions, 20, chop_last=9)


# Car dataset (pred price from car info)
def test_multiple_linear():
    # Import the data for training and testing
    # ./datasets/car_dataset.csv
    file = pd.read_csv('./datasets/car_dataset.csv')  # 205 entries, 8 variables, price at end
    data = np.array(file)
    
    # Split data into input and output (x and y)
    data_x = data[:150, 0:8]
    data_y = data[:150, 8] / 1000
    eval_x = data[150:, 0:8]
    eval_y = data[150:, 8] / 1000
    
    
    # Build the model
    model = lr.LinearRegression(optimizer=optimizers.AdamWithCorrection(), track_metrics=True, iterations=2000,
                                stochastic=True, batch_size=64)
    
    # Train the model
    model.train(data_x, data_y)
    
    # Evaluate the model
    model.evaluate(eval_x, eval_y)
    
    # Store all predictions
    predictions = model.predict_all(eval_x)
    
    # Graph out the results
    plot_pred_on_real(eval_y, predictions)
    
    # Plot the error over time ----------------------------
    # Get the array of past errors from the model
    errors = model.error_hist[10:]
    # Get ready to average errors into bins.  Define number of iterations that go into each bin
    bin_size = 200
    # Initialize the size of the bin array (divide total length of errors by bin size, and add one to account for
    # residual errors at end that don't completely fill bin (10/4=2.5; math.floor(2.5)=2; size+1=3; Done)
    bin_errors = [0 for _ in range(math.floor(len(errors) / bin_size) + 1)]
    # For each error, add it to it's bin (round down the quotient of i and bin_size)
    for i in range(len(errors)):
        bin_errors[math.floor(i / bin_size)] += errors[i]
    # For each bin (other than the last one), divide it's sum by the bin size (find it's average)
    for i in range(len(bin_errors)-1):
        bin_errors[i] /= bin_size
    # Divide the last bin by the number of items that fell into it (remainder from previous division)
    bin_errors[-1] /= len(errors) % bin_size
    # Done making the bin array! -----------------------------
    
    # Store the iteration bin markers (x axis values)
    x_marks = [i * bin_size for i in range(len(bin_errors))]
    # Plot the data
    plt.plot(x_marks, bin_errors, 'b')
    # Set the labels
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    # Show the plot
    plt.show()


# Weather data (pred daily low from high)
def test_multiple_linear_as_simple():
    # Import the data for training and testing
    file = pd.read_csv('./datasets/WWII weather data/weather_clean.csv')  # 205 entries, 1 variable, price at end
    data = np.array(file)

    # Split data into input and output (x and y)
    data_x = data[:50000, 0]
    data_y = data[:50000, 1]
    eval_x = data[50000:75000, 0]
    eval_y = data[50000:75000, 1]

    # Build the model
    model = lr.LinearRegression(optimizer=optimizers.Momentum(gamma=0.9, learn_rate=0.003))

    # Train the model
    model.train(data_x, data_y)

    # Evaluate the model
    model.evaluate(eval_x, eval_y)
    
    # Store a list of predictions for the evaluation dataset
    predictions = model.predict_all(eval_x)
    
    # Plot the error in a histogram in bins defined by the variance from y to y_hat
    plot_error_density(eval_y, predictions, 40, chop_last=25)
    plot_pred_on_real(eval_y, predictions)


# Drug 200 file (pred if datapoint is class Y or not)
def test_logistic():
    # Import the data for training and testing
    file = pd.read_csv('./datasets/drug200_Y.csv')
    data = np.array(file)

    # Split data into input and output (x and y)
    data_x = data[:150, 0]
    data_y = data[:150, 1]
    eval_x = data[150:, 0]
    eval_y = data[150:, 1]

    threshold = 0.5
    # iterations = 5000
    # Build the model
    model = BlogR.BinaryLogisticRegression(headers=['Weight', 'Bias'], threshold=threshold)

    # Train the model
    model.train(data_x, data_y)

    # Evaluate the model
    model.evaluate(eval_x, eval_y)

    # Store predictions
    predictions = model.predict_all(eval_x)

    # Store the point of X at which the model starts guessing 1
    cp = model.get_changing_point()

    # Graph out the results
    plt.figure(figsize=(12, 8))
    plt.scatter(eval_x, predictions)  # Model prediction confidence sigmoid
    plt.scatter(eval_x, eval_y)  # True values in dataset
    plt.plot([7, 36], [threshold, threshold])  # Horizontal threshold line
    plt.plot([cp, cp], [0, 1], '--')  # Vertical dashed changing point line
    plt.title('Predictions of whether drug Y will help')
    plt.ylabel('Confidence Score')
    plt.xlabel('NA to K ratio of patient')
    plt.show()


# Drug 200 file (pred each class)
def test_multinomial_logistic():
    # Import the data for training and testing
    file = pd.read_csv('./datasets/drug200_full_clean.csv')
    data = np.array(file)

    # Bootstrap a training set and eval set
    data_n = 0.8
    eval_n = 0.4
    data_x = data[:, :5]
    data_y = data[:, 5:]

    train_x, train_y = bootstrap_sample(data_n, data_x, data_y)
    eval_x, eval_y = bootstrap_sample(eval_n, data_x, data_y)

    # Build the model
    model = logR.LogisticRegression()

    # Train the model
    model.train(train_x, train_y)

    # Evaluate the model
    model.evaluate(eval_x, eval_y)

    """# Graph out the results
    plt.figure(figsize=(12, 8))
    plt.plot(predictions, eval_y)
    plt.show()"""
    """plt.scatter(eval_x, predictions)  # Model prediction confidence sigmoid
    plt.scatter(eval_x, eval_y)  # True values in dataset
    plt.plot([7, 36], [threshold, threshold])  # Horizontal threshold line
    plt.title('Predictions of whether drug Y will help')
    plt.ylabel('Confidence Score')
    plt.xlabel('NA to K ratio of patient')
    plt.show()"""


# Runs two models N times each and compares overall time
def compare_models(n):
    # Last time I compared these (10/2/23), simple took 27 seconds for 1000 iterations,
    # while the newer one only took 1.5 seconds for the same number of iterations!
    
    # Import the data for training and testing
    file = pd.read_csv('./datasets/WWII weather data/weather_clean.csv')
    data = np.array(file)

    # Split data into input and output (x and y)
    data_x = data[:50000, 0]
    data_y = data[:50000, 1]
    eval_x = data[50000:75000, 0]
    eval_y = data[50000:75000, 1]

    old_start = time.time()
    for i in range(n):
        model = slr.SimpleLinearRegression(default_prints=False)
        model.train(data_x, data_y)
        model.evaluate(eval_x, eval_y)
    old_end = time.time()

    new_start = time.time()
    for i in range(n):
        model = lr.LinearRegression(1, default_prints=False)
        print(data_x[:])
        model.train(data_x.transpose(), data_y)
        model.evaluate(eval_x, eval_y)
    new_end = time.time()

    old = int((old_end - old_start) * 1000)
    new = int((new_end - new_start) * 1000)
    diff = new - old

    print(f'Simple model took {old}ms to complete {n} training / evaluation runs.')
    print(f'Multiple model took {new}ms to complete {n} training / evaluation runs.')
    plus = '+'
    empty = ''
    print(f'That\'s a difference of {plus if diff > 0 else empty}{diff}ms from the old model to the new')

    print(f'On average, Old was {math.floor(old/n)}ms per run, '
          f'New was {math.floor(new/n)}ms per run, '
          f'and the difference was {plus if diff > 0 else empty}{math.floor(diff/n)}ms per run.')


# Method written to compete in a Kaggle competition.  I didn't do great, but judging that I used a linear model, I am
# happy with the results.
def compete():

    def fill_with_0(_list):
        for _i in range(len(_list)):
            if not _list[_i] > 0:
                _list[_i] = 0

    file = pd.read_csv('./datasets/kaggle/train.csv')

    ar = np.array(file)

    by_street = ar[:, 3]
    is_level = ar[:, 8]
    land_slope = ar[:, 11]
    is_by_railroad = ar[:, 13:15]
    is_by_railroad_finished = np.zeros(len(is_by_railroad))
    overall_quality = ar[:, 17]
    overall_condition = ar[:, 18]
    month_sold = ar[:, -5]
    year_sold = ar[:, -4]
    price = ar[:, len(ar[0]) - 1]

    fill_with_0(by_street)
    leveldict = dict()
    leveldict['Lvl'] = 1
    replace_all_dict(is_level, leveldict, 0)
    slopedict = dict()
    slopedict['Gtl'] = 0
    slopedict['Mod'] = 1
    replace_all_dict(land_slope, slopedict, 2)
    for i in range(len(is_by_railroad)):
        if (is_by_railroad[i, 0][0:2] == 'RR') or (is_by_railroad[i, 1][0:2] == 'RR'):
            is_by_railroad_finished[i] = 1
    min_year = np.min(year_sold)
    year_sold_finished = np.zeros(len(year_sold))
    for i in range(len(year_sold)):
        year_sold_finished[i] = year_sold[i] - min_year

    data_x = np.column_stack((by_street, is_level, land_slope, is_by_railroad_finished, overall_quality,
                              overall_condition, month_sold, year_sold_finished))
    data_y = np.array(price)

    # ---------------------------------------------------------

    file = pd.read_csv('./datasets/kaggle/test.csv')

    tar = np.array(file)
    ids = tar[:, 0]

    tstreet_against_property = tar[:, 3]
    tis_level = tar[:, 8]
    tland_slope = tar[:, 11]
    tis_by_railroad = tar[:, 13:15]
    tis_by_railroad_finished = np.zeros(len(tis_by_railroad))
    toverall_quality = tar[:, 17]
    toverall_condition = tar[:, 18]
    tmonth_sold = tar[:, -4]
    tyear_sold = tar[:, -3]

    fill_with_0(tstreet_against_property)
    replace_all_dict(tis_level, leveldict, 0)
    replace_all_dict(tland_slope, slopedict, 2)
    for i in range(len(tis_by_railroad)):
        if (tis_by_railroad[i, 0][0:2] == 'RR') or (tis_by_railroad[i, 1][0:2] == 'RR'):
            tis_by_railroad_finished[i] = 1
    tmin_year = np.min(tyear_sold)
    tyear_sold_finished = np.zeros(len(tyear_sold))
    for i in range(len(tyear_sold)):
        tyear_sold_finished[i] = tyear_sold[i] - tmin_year

    test_x = np.column_stack((tstreet_against_property, tis_level, tland_slope, tis_by_railroad_finished,
                              toverall_quality, toverall_condition, tmonth_sold, tyear_sold_finished))

    # ----------------------------------------------------------------------------

    model = lr.LinearRegression()

    train_x = data_x[:1000, :]
    eval_x = data_x[1000:, :]
    train_y = data_y[:1000] / 1000
    eval_y = data_y[1000:] / 1000

    # Transform data
    # t_train_x = data_normalize(train_x)
    # t_eval_x = data_normalize(eval_x)   # Transforming data with my normalize function tanked the model

    print('T, Train, X', train_x)

    model.train(train_x, train_y)

    predictions = model.predict_all(eval_x)

    plt.scatter(eval_y, predictions)
    plt.plot([0, max(predictions)], [0, max(predictions)], 'r--')
    plt.show()

    print('Done!')
    print(model.error)

    final_predictions = model.predict_all(test_x) * 1000

    output = np.column_stack((ids, final_predictions))
    output_file = pd.DataFrame(output)
    output_file.to_csv('output2.csv')


# Tests the Decision Tree on either the drug200 set, or the uci-occupancy set
def test_tree():
    # Import the data for training and testing
    # TODO: Write script that slowly increases the number of points for testing, and times each seperating them with a
    # threshold of significant difference of T, and finds the low and high point where the normal split function and
    # matrix one both have average finish times within one T of eachother, then take the average of those points for a
    # good, general-purpose threshold to use for hybrid
    # file = pd.read_csv('./datasets/uci-occupancy-detection/datatraining.csv')
    file = pd.read_csv('./datasets/drug200.csv')
    # file_test = pd.read_csv('./datasets/uci-occupancy-detection/datatest.csv')
    data = np.array(file)
    split = 150  # 6000
    # data_test = np.array(file_test)

    # Split data into input and output (x and y)
    data_x = data[:split, :5].tolist()
    data_y = data[:split, 5].tolist()
    eval_x = data[split:, :5].tolist()
    eval_y = data[split:, 5].tolist()

    model = dt.DecisionTree()

    model.train(data_x, data_y)

    model.evaluate(eval_x, eval_y)


# Tests a simple, Vanilla Neural Network
def test_old_vanilla_network():

    # Car dataset
    file = pd.read_csv('./datasets/uci-occupancy-detection/datatest.csv')  # 205 entries, 8 variables, price at end
    data = np.array(file)

    # Split data into input and output (x and y)
    data_x = data[:7500, 0:5]
    data_y = data[:7500, 5]
    eval_x = data[7500:, 0:5]
    eval_y = data[7500:, 5]
    
    for i in range(len(data_x[0])):
        data_max = np.max(data_x[:, i])
        eval_max = np.max(eval_x[:, i])
        data_x[:, i] = data_x[:, i] / data_max
        eval_x[:, i] = eval_x[:, i] / eval_max

    print("Creating Neural Network model..")
    
    model = nn.VanillaNetwork([
        (5, None),
        (7, af.relu),
        (8, af.relu),
        (3, af.relu),
        (4, af.sigmoid),
        (1, af.linear)
    ])
    
    print("Compiling Neural Network model..")
    model.compile(
            cost_function=cf.least_squares,
            optimizer=optimizers.AdamForNetworks(initial_lr=0.003)
        )
    
    print("\nTraining Neural Network model..")
    costs1 = model.train(data_x, data_y, epochs=25, batch_size=12)
    # Confirmed: Model works with multiple training stages
    costs2 = model.train(data_x, data_y, epochs=40, batch_size=48)
    
    print("\nEvaluating Neural Network model..")
    model.evaluate(eval_x, eval_y, batch_size=4)
    
    plt.plot(range(5, len(costs1)), costs1[5:], 'b')
    plt.plot(range(len(costs1), len(costs2) + len(costs1)), costs2, 'r-')
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.show()


# Tests the new set of classes that make up the LayeredNetwork system
def test_network():
    # Car dataset
    file = pd.read_csv('./datasets/uci-occupancy-detection/datatest.csv')  # 205 entries, 8 variables, price at end
    data = np.array(file)
    
    # Split data into input and output (x and y)
    data_x = data[:7500, 0:5]
    data_y = data[:7500, 5]
    eval_x = data[7500:, 0:5]
    eval_y = data[7500:, 5]
    
    for i in range(len(data_x[0])):
        data_max = np.max(data_x[:, i])
        eval_max = np.max(eval_x[:, i])
        data_x[:, i] = data_x[:, i] / data_max
        eval_x[:, i] = eval_x[:, i] / eval_max
    
    print("Creating Neural Network model..")
    model = ln.LayeredNetwork([
        BatchNormDenseLayer(5, 32),
        BatchNormDenseLayer(32, 128),
        BatchNormDenseLayer(128,128),
        BatchNormDenseLayer(128,128),
        BatchNormDenseLayer(128, 64, activation_function=af.sigmoid),
        BatchNormDenseLayer(64, 8, activation_function=af.sigmoid),
        DenseLayer(8, 1, activation_function=af.linear)
    ])
    
    print("Compiling Neural Network model..")
    model.compile(
            cost_function=cf.least_squares,
            optimizer=optimizers.AdamForNetworks(initial_lr=0.001)
    )
    
    print("\nTraining Neural Network model..")
    costs = model.train(data_x, data_y, epochs=25, batch_size=32, prints_friendly=False)
    
    print("\nEvaluating Neural Network model..")
    model.evaluate(eval_x, eval_y, batch_size=32)
    
    plt.plot(range(2, len(costs)), costs[2:], 'b')
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.show()


def test_json_output():
    # Car dataset
    file = pd.read_csv('./datasets/uci-occupancy-detection/datatest.csv')  # 205 entries, 8 variables, price at end
    data = np.array(file)
    
    # Split data into input and output (x and y)
    data_x = data[:7500, 0:5]
    data_y = data[:7500, 5]
    eval_x = data[7500:, 0:5]
    eval_y = data[7500:, 5]
    
    for i in range(len(data_x[0])):
        data_max = np.max(data_x[:, i])
        eval_max = np.max(eval_x[:, i])
        data_x[:, i] = data_x[:, i] / data_max
        eval_x[:, i] = eval_x[:, i] / eval_max
    
    print("Creating Neural Network model..")
    model = ln.LayeredNetwork([
        DenseLayer(5, 8),
        DenseLayer(8, 5),
        BatchNormDenseLayer(5, 4),
        DenseLayer(4, 1, activation_function=af.linear)
    ])
    
    print("Compiling Neural Network model..")
    model.compile(
            cost_function=cf.least_squares,
            optimizer=optimizers.AdamForNetworks(initial_lr=0.001)
    )
    
    print("\nTraining Neural Network model..")
    model.train(data_x, data_y, epochs=2, batch_size=8)
    
    print("\nEvaluating Neural Network model..")
    model.evaluate(eval_x, eval_y, batch_size=32)
    
    json_to_save = model.to_json()
    # Actually go in and write the file
    with open("testOutput.json", "w") as outputFile:
        json.dump(json_to_save, outputFile)
    
    print("Done!")


def test_json_input():
    # Car dataset
    file = pd.read_csv('./datasets/uci-occupancy-detection/datatest.csv')  # 205 entries, 8 variables, price at end
    data = np.array(file)
    
    # Split data into input and output (x and y)
    eval_x = data[7500:, 0:5]
    eval_y = data[7500:, 5]
    
    for i in range(len(eval_x[0])):
        eval_max = np.max(eval_x[:, i])
        eval_x[:, i] = eval_x[:, i] / eval_max
    
    fileName = "testOutput.json"
    f = open(fileName,)
    json_obj = json.load(f)
    f.close()
    model = ln.build_from_json(json_obj)
    print("Model:\t\t\t", json.dumps(model.to_json()))
    model.evaluate(eval_x, eval_y, batch_size=32)
    

# --------------------------- Call methods here -----------------------------
test_network()
