from models import logisticRegression as logR
from models import linearRegression as lr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import math
import random


# ---------------------------------------- Matplotlib Helper Methods ---------------------------------------
def plot_pred_on_real(data_y, predictions, xlabel=None, ylabel=None):
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
    if xlabel is None:
        plt.xlabel('Real Value (Y)')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel('Predicted Value (Y^)')
    else:
        plt.ylabel(ylabel)
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


# -------------------------------------- Testing Methods --------------------------------------------
# Weather data (pred daily low from high)
def test_simple_linear():
    # Import the data for training and testing
    file = pd.read_csv('./datasets/WWII weather data/weather_clean.csv')
    data = np.array(file)

    # Split data into input and output (x and y)
    data_x = data[:50000, 0]
    data_y = data[:50000, 1]
    eval_x = data[50000:75000, 0]
    eval_y = data[50000:75000, 1]

    # Build the model
    model = lr.SimpleLinearRegression()

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
    file = pd.read_csv('./datasets/car_dataset.csv')  # 205 entries, 8 variables, price at end
    data = np.array(file)

    # Split data into input and output (x and y)
    data_x = data[:150, 0:8]
    data_y = data[:150, -1]
    eval_x = data[150:, 0:8]
    eval_y = data[150:, -1]

    # Build the model
    model = lr.MultipleLinearRegression(8)

    # Train the model
    model.train(data_x, data_y)

    # Evaluate the model
    model.evaluate(eval_x, eval_y)

    predictions = model.predict_all(eval_x)

    # Graph out the results
    plot_pred_on_real(eval_y, predictions)


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
    model = lr.MultipleLinearRegression(1)

    # Train the model
    model.train(data_x, data_y)

    # Evaluate the model
    model.evaluate(eval_x, eval_y)
    
    # Store a list of predictions for the evaluation dataset
    predictions = model.predict_all(eval_x)
    
    plot_error_density(eval_y, predictions, 40, chop_last=20)


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
    iterations = 5000
    # Build the model
    model = logR.BinaryLogisticRegression(headers=['Weight', 'Bias'], threshold=threshold)

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
    model = logR.MultinomialLogisticRegression(5, 5)

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
        model = lr.SimpleLinearRegression(default_prints=False)
        model.train(data_x, data_y)
        model.evaluate(eval_x, eval_y)
    old_end = time.time()

    new_start = time.time()
    for i in range(n):
        model = lr.MultipleLinearRegression(1, default_prints=False)
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


# --------------------------- Call methods here -----------------------------

test_multinomial_logistic()
