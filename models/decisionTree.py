# Author: Matthew Williams
from dataclasses import dataclass
from operator import attrgetter
import numpy as np

"""
Notes to Understand:

1) In the Node class, all class labels are integers representing the class label.  That is all the Node knows.
     But the DecisionTree class keeps two dictionaries.  One to convert class labels to integers, and one to turn
     the integer back into the class label.  The tree deals with the user, and with the Strings. The Nodes deal with
     only the class integers (basically an index representation for the class)
"""


@dataclass
class DataPoint:
    x: list
    y: int


# ---------------------------------------------[ Entropy Calculation Methods ]------------------------------------
def calculate_node_entropy(leaf):
    """Calculates the entropy of the given node, and return it's entropy multiplied by the number of points in it, as
    well as the number of points in it"""

    """ -=[Completed.]=- """

    # Initialize a variable to store the total entropy for this node, across classes
    total = 0
    # Initialize a list to store how many points are from each class
    p_list = [p.y for p in leaf.data_points]
    cc = np.array(np.unique(p_list, return_counts=True)[1])
    # Take the sum of the list to get a total number of points (in my mind might be faster than len(data points) )
    cc_sum = np.sum(cc)
    # Divide the counts by the sum (so the sum(cc) = 1)
    cc = cc / cc_sum

    # Loop through each class
    for k in cc:
        # Add the probability of k multiplied by log base 2 of that probability (entropy)
        total += k * np.log2(k)

    # Return the negative entropy weighted by the number of points, and the number of points
    return -(total * cc_sum), cc_sum


def entropy_raw(node):
    """This is the recursive method for entropy().  Call that method instead."""

    """ -=[Completed.]=- """

    # If the recursion has hit a leaf, simply return it's weighted entropy and datapoint count
    if node.is_leaf:
        return calculate_node_entropy(node)

    # Otherwise, start up variables to hold the entropy and count of this node's children
    node_entropy = 0
    node_dp_count = 0

    # Loop through each child node
    for child in node.children:
        # Recurse through this method for the child
        en, c = entropy_raw(child)
        # Add the result to the entropy and dp count running  variables
        node_entropy += en
        node_dp_count += c

    # Return the final entropy sum and data point count
    return node_entropy, node_dp_count


def entropy(node):
    """Use this to calculate the entropy of a node.  Recursively calls children, too"""

    """ -=[Completed.]=- """

    # If the node is a leaf
    if node.is_leaf:
        # Return only the entropy of this node (leaf)
        return calculate_node_entropy(node)[0]

    # Otherwise, if this node has children
    # Make variables to hold all the entropy and the number of data points
    entropy_tally = 0
    dp_count = 0
    # Loop through all the node's children
    for child in node.children:
        # Call the recursive entropy_raw, and store it's final returning entropy and
        child_entropy, child_count = entropy_raw(child)
        entropy_tally += child_entropy
        dp_count += child_count

    # Divide the entropy tally by the data point count
    final_entropy = entropy_tally / dp_count
    # Return the weighted entropy
    return final_entropy


def sum_node_entropy(node):
    """Traverses the tree from the given node and returns the entropy already stored.
    This assumes that the nodes are already flattened."""

    if node.is_leaf:
        return node.entropy, node.data_count
    total_ent = 0
    total_count = 0
    for child in node.children:
        child_ent, child_count = sum_node_entropy(child)
        total_ent += child_ent
        total_count += child_count
    return total_ent, total_count
# ----------------------------------------------------------------------------------------------------------------


def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(j) for j in data])

    # Mask of valid places in each row
    mask = np.arange(max(lens)) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape)
    out[mask] = np.concatenate(data)
    return out


# ---------------------------------------------[ Node Class ]-----------------------------------------------------
class Node:

    def __init__(self):
        """Initializes a node, either as a leaf or as a normal node.


        Parameters

        -split_function: function.  Takes in a data point, and returns an index for which child gets that point

        -split_feature: int.  Index of the feature in X on which the points are being split

        -split_input: int.  For numerical splits, this is the threshold used.  For categorical, this is the class to
        split from the rest.  Or if categorical and split_input is -1, all classes are split into their own class

        -is_leaf: boolean.  Indicates whether this is a leaf node and therefore has no children but a prediction

        -prediction: int (class index).  Stored prediction for any given point that lands on this leaf

        -depth: int.  By default is 0, which indicates a root.  Each node added to another gets a depth of +1"""

        """ -=[Completed.]=- """

        # Variables for every node
        self.is_leaf = False
        self.depth = 0
        self.entropy = None
        self.data_count = None

        # Leaf Variables
        self.prediction = None  # Integer of class index
        self.data_points = []

        # Non-leaf variables
        self.split_function = None  # Function that splits data points into children indices
        self.split_feature = None  # Feature to split on
        self.split_input = None  # If numerical, this is the threshold.  If categorical, it is the class to split
        self.split_add = None
        self.children = []

    def set(self, dp_list):
        """Sets this node's list of points as the given list of points"""

        """ -=[Completed.]=- """

        self.data_points = dp_list.tolist()

    def give(self, dp):
        """Appends the given data point to this node's list of points"""

        """ -=[Completed.]=- """

        self.data_points.append(dp)

    def classify(self, dp):
        """Classifies the point by returning this node's prediction, or if it has children, the prediction of the
        corresponding child"""

        """ -=[Completed.]=- """

        # If this is a leaf node (no children, but has a prediction)
        if self.is_leaf:
            # Return that prediction
            return self.prediction
        # Otherwise, if this is a normal node (children, but no prediction)
        else:
            # Get the index of the corresponding child with this node's split function, and call it's classify method
            return \
                self.children[
                    self.split_function(dp, self.split_feature, self.split_input, self.split_add)].classify(dp)

    def set_children(self, nodes):
        """Store the list of nodes as this node's Children, and set each child's depth as this node's depth + 1"""

        """ -=[Completed.]=- """

        # Store the list of nodes as this node's Children
        self.children = nodes
        # Loop through each child
        for child in self.children:
            # Set the child depth to self.depth + 1
            child.depth = self.depth + 1

    def flatten(self):
        """Method for Leaf classes after data training is done.  Records the leaf's predicted class, and the
        probability of that class in that node (# of that class / # in this node total), and removes the
        array of data points from the node.  Done to increase efficiency in prediction."""

        """ -=[Completed.]=- """

        # Only proceed if this is a leaf.  If not, no action is needed.
        if self.is_leaf:
            # Separate the Y values from the data points, and record the unique classes found, and the number
            #  of times that each was seen in the list

            self.entropy, self.data_count = calculate_node_entropy(self)

            unique_classes, class_counts = np.unique([p.y for p in self.data_points], return_counts=True)

            # Set self.prediction as the predicted class (class with highest occurrence in node)
            if len(class_counts) == 0:
                print('Empty class counts!  Odd.  Prediction: 55')
                self.prediction = 55
            else:
                self.prediction = unique_classes[np.argmax(class_counts)]

            # Delete the array of data points held in this node
            self.data_points = None
# ----------------------------------------------------------------------------------------------------------------


# ---------------------------------------------[ Split Finding Methods ]------------------------------------------
def numerical_split(dp, feature, threshold, *args):
    # Return 1 if the point's value at this feature is more than the threshold
    return 1 if dp.x[feature] >= threshold else 0


def categorical_split(dp, feature, splitter, added=None):
    # If the 'splitter' is -1, which means to split each category into it's own node
    if splitter == -1:
        # Assume 'added' to be a dictionary of unique category types for this feature
        return added[dp.x[feature]]

    # Otherwise, return 1 if the data point's feature matches the splitter, or 0 if not
    return 1 if dp.x[feature] == splitter else 0


def try_split_virtual_vector(split_func, feature, split_in, added, data_points):
    # Build list of split indices
    split_indices = [split_func(p, feature, split_in, added) for p in data_points]
    max_index = 2 if added is None else len(added)

    # Get a matrix of class values per node
    y_s = [[] for _ in range(max_index)]
    for i in range(len(data_points)):
        y_s[split_indices[i]].append(data_points[i].y)

    # Deal with identifying unique classes and their occurrence count
    parts = []
    # Loop through each 'node'
    for r in y_s:
        # As long as this node got at least a point (don't deal with or pass on any row that got no points)
        if len(r) > 0:
            # Find unique class counts (No need for the actual labels)
            part = np.unique(r, return_counts=True)[1]
            parts.append(part)

    # Turn the parts list into a numpy array
    parts = np.array(numpy_fillna(parts))
    # Store each row (or node)'s sum
    parts_sums = np.sum(parts, axis=1)
    # Divide each item by that row's sum, as long as the sum isn't 0
    parts = np.divide(parts.T, parts_sums.T, where=(parts_sums > 0)).T
    # Sum up each point's entropy by row
    ent_sums = np.sum(parts * np.log2(parts, where=(parts > 0)), axis=1)
    # Multiply each row's entropy by the number of points in that row
    ent_sums *= parts_sums
    # Sum up each row's entropy
    total_ent = np.sum(ent_sums)

    # Return the entropy for this split (should be negative to allow minimizing rather than maximizing)
    return -total_ent


def find_best_split(data_points, feature, categories=None):

    if isinstance(data_points[0].x[feature], int) or isinstance(data_points[0].x[feature], float):
        # numerical split
        # Store all possible significant thresholds
        significant_numbers = []

        # Sort the points on the given feature
        list.sort(data_points, key=lambda item: item.x[feature])

        # Store the first seen y value, and store any values of x that result in a change in y
        current_y = data_points[0].y
        for p in data_points:
            if p.y != current_y:
                significant_numbers.append(p.x[feature])
                current_y = p.y

        # Store Running variables
        best_split_threshold = None
        best_split_entropy = None
        # For each significant number, try that split and see if it got a better entropy than the last
        for threshold in significant_numbers:
            ent = try_split_virtual_vector(numerical_split, feature, threshold, None, data_points)
            if (best_split_entropy is None) or (ent < best_split_entropy):
                best_split_threshold = threshold
                best_split_entropy = ent

        return numerical_split, best_split_threshold, None, best_split_entropy

    else:
        if isinstance(data_points[0].x[feature], str):
            # categorical split
            # Store all possible significant thresholds
            significant_splitters = []

            for c in categories:
                significant_splitters.append(c)
            significant_splitters.append(-1)

            # Store Running variables
            best_split_splitter = None
            best_split_entropy = None
            # For each significant number, try that split and see if it got a better entropy than the last
            for split in significant_splitters:
                ent = try_split_virtual_vector(categorical_split, feature, split, categories, data_points)
                if (best_split_entropy is None) or (ent < best_split_entropy):
                    best_split_splitter = split
                    best_split_entropy = ent

            return categorical_split, best_split_splitter, categories, best_split_entropy


def find_best_split_virtual_vector(data_points, feature, categories=None):
    if isinstance(data_points[0].x[feature], int) or isinstance(data_points[0].x[feature], float):
        # numerical split

        # Step 1)
        # Store all possible significant thresholds
        significant_numbers = []
        # Sort the points on the given feature
        list.sort(data_points, key=lambda item: item.x[feature])
        # Store the first seen y value, and store any values of x that result in a change in y
        current_y = data_points[0].y
        for p in data_points:
            if p.y != current_y:
                significant_numbers.append(p.x[feature])
                current_y = p.y

        # Step 2)
        # Build the class counts matrix
        # Since this feature is numerical, all splits are in [0, 1], so we don't need to calculate max_index
        # But, we still want the maximum Y value that is going to be dealt with
        max_y = max(data_points, key=attrgetter('y')).y + 1
        # Instantiate the empty list of class counts
        class_counts = np.array([[  # (split X node X class) matrix of counts
            [0 for _ in range(len(significant_numbers))] for _ in range(2)
        ] for _ in range(max_y)])
        # Now, actually build that list out by adding to each spot accordingly
        for thresh in range(len(significant_numbers)):
            for dp in data_points:
                class_counts[dp.y][numerical_split(dp, feature, significant_numbers[thresh], categories)][thresh] += 1

        # Step 3)
        # Sum the matrix by row, so a 2D matrix of node sums is resulted
        node_sums = np.sum(class_counts, axis=0)

        # Step 4)
        # Divide each element of the class counts by it's corresponding node sum (so sum(node's counts) = 1)
        normalized_counts = np.divide(class_counts, node_sums, where=(node_sums > 0)).T  # (split, index, class)

        # Step 5)
        # Sum each probability multiplied by the log2 of itself over each node
        entropy_sums = np.sum(normalized_counts * np.log2(normalized_counts, where=(normalized_counts > 0)), axis=2)

        # Step 6)
        # Multiply each node's entropy by it's corresponding sum
        entropy_sums *= node_sums.T

        # Step 7)
        # Total each prospective split's entropy
        total_entropy_sums = np.sum(entropy_sums, axis=1)

        # Step 8)
        # Find the best result and store info
        best_entropy = -np.max(total_entropy_sums)
        best_threshold_index = np.argmax(total_entropy_sums)

        # Step 9)
        # Return the result
        return numerical_split, significant_numbers[best_threshold_index], None, best_entropy

    else:
        if isinstance(data_points[0].x[feature], str):
            # categorical split

            # Step 1)
            # Store all possible significant thresholds
            significant_splitters = []
            # Store the first seen y value, and store any values of x that result in a change in y
            for c in categories:
                significant_splitters.append(c)
            significant_splitters.append(-1)

            # Step 2)
            # Build the class counts matrix
            # Since this feature is categorical, all splits are within the range of, at most, the length of 'categories'
            # But, we still want the maximum Y value that is going to be dealt with
            max_y = max(data_points, key=attrgetter('y')).y + 1
            # Instantiate the empty list of class counts  TODO Right Here Item 1
            class_counts = np.array([[  # (split X node X class) matrix of counts
                [0 for _ in range(len(significant_splitters))] for _ in range(len(categories))
            ] for _ in range(max_y)])
            # Now, actually build that list out by adding to each spot accordingly
            for split in range(len(significant_splitters)):
                for dp in data_points:
                    class_counts[dp.y][categorical_split(dp, feature, significant_splitters[split], categories)][
                        split] += 1

            # Step 3)
            # Sum the matrix by row, so a 2D matrix of node sums is resulted
            node_sums = np.sum(class_counts, axis=0)

            # Step 4)
            # Divide each element of the class counts by it's corresponding node sum (so sum(node's counts) = 1)
            normalized_counts = np.divide(class_counts, node_sums, where=(node_sums > 0)).T  # (split, index, class)

            # Step 5)
            # Sum each probability multiplied by the log2 of itself over each node
            entropy_sums = np.sum(normalized_counts * np.log2(normalized_counts, where=(normalized_counts > 0)), axis=2)

            # Step 6)
            # Multiply each node's entropy by it's corresponding sum
            entropy_sums *= node_sums.T

            # Step 7)
            # Total each prospective split's entropy
            total_entropy_sums = np.sum(entropy_sums, axis=1)

            # Step 8)
            # Find the best result and store info
            best_entropy = -np.max(total_entropy_sums)
            best_splitter_index = np.argmax(total_entropy_sums)

            # Step 9)
            # Return the result
            return categorical_split, significant_splitters[best_splitter_index], categories, best_entropy


hybrid_threshold = 300


def find_best_split_hybrid(data_points, feature, categories=None):
    if len(data_points) < hybrid_threshold:
        return find_best_split_virtual_vector(data_points, feature, categories)
    return find_best_split(data_points, feature, categories)


my_split_finder = find_best_split_hybrid  # Change this to affect the whole tree structure (alias variable)
# ----------------------------------------------------------------------------------------------------------------


# ---------------------------------------------[ Tree Class ]-----------------------------------------------------
class DecisionTree:

    def __init__(self, min_points=20, min_entropy=0.05):
        """Initializes a Decision Tree with a blank root node."""

        # Initialize state variables
        self.class_dict = {}
        self.index_dict = {}
        self.uniques_dict = {}  # Stores dictionaries of all categorical data's possible values, to pass on in grow
        self.root = Node()

        # Metric variables
        self.accuracy = 0
        self.entropy = 0

        # Hyperparameters
        self.min_points = min_points
        self.min_entropy = min_entropy

    def train(self, x, y, prints=True):
        """Trains the model on the data given

        Parameters

        -x: matrix.  Input matrix where rows are data points and columns are features.

        -y: array.  Input values' class label, in the form of the actual class name (String)"""

        """ -=[Completed.]=- """

        # Internalize the class data in the form of dictionaries to go from class label strings to indices and back
        self.__build_class_dict(x, y)

        # Give the root the array of DataPoints generated by self.__get_data_points()
        self.root.set(self.__get_data_points(x, y))

        # Begin the recursive, depth-first growth of the tree starting at the root
        self.__grow_node(self.root)

        # Record the resulting tree's predictions for each data point
        y_hat = self.predict(x, return_int=True)

        # Calculate the accuracy and entropy of the model
        self.__calculate_accuracy(y_hat, y)
        self.__calculate_entropy()

        if prints:
            # Print out the results of the training
            print('\n\nTraining done!')
            print('Accuracy: %.2f' % (self.accuracy * 100))
            print('Entropy: %.3f' % self.entropy)
            # TODO: put method in tree class that traverses tree and counts nodes, finds max-depth, etc. Print here.

    def evaluate(self, x, y, prints=True):
        """Predicts every point and evaluates how the model did

        Parameters

        -x: matrix.  Input matrix where rows are data points and columns are features.

        -y: array.  Input values' class label, in the form of the actual class name (String)"""

        """ -=[Completed.]=- """

        # Record the resulting tree's predictions for each data point
        y_hat = self.predict(x, return_int=True)

        # Calculate the accuracy and entropy of the model
        self.__calculate_accuracy(y_hat, y)
        self.__calculate_entropy()

        if prints:
            print('\n\nEvaluation done!')
            print('Accuracy: %.2f' % (self.accuracy * 100))
            print('Entropy: %.3f' % self.entropy)

            print('\n\n[evaluate] Class Dictionary:')
            for i in range(len(self.class_dict)):
                print(f'{i}: {self.class_dict[i]}')
            print('\n[evaluate]')
            # Print out the whole tree
            self.__print_node(self.root)
            print('\n\n')

    def predict(self, x, return_int=False):
        """Returns a class prediction for each input row in X"""

        """ -=[Completed.]=- """

        # If 'x' only includes one data point
        if len(x[0]) == 1:  # This is only used externally.  No internal use.
            print('[predict] Using the single predict() section of the method')

            # Return the root's classification of that point
            return self.root.classify(DataPoint(x, 0)) \
                if return_int else \
                self.class_dict[self.root.classify(DataPoint(x, 0))]

        # Otherwise, if 'x' has multiple points, return an array of the root's classifications
        return [(self.root.classify(DataPoint(point, 0))
                 if return_int else
                 self.class_dict[self.root.classify(DataPoint(point, 0))])
                for point in x]

    def __grow_node(self, node):

        n_ent = calculate_node_entropy(node)
        # Check if this node should split.  If not, flatten the node and return (AKA: set it to a leaf node)
        if len(node.data_points) < self.min_points or \
                n_ent[0] / n_ent[1] < self.min_entropy:
            # Set the node as a leaf, and flatten it
            node.is_leaf = True
            node.flatten()
            return

        # Loop through features and find the best split for each one.  Keep track of the best
        ex_p = node.data_points[0]  # Example point, for testing feature type and length easier
        feature_count = len(ex_p.x)
        # Store all the running 'best' variables
        best_split_func = None
        best_split_input = None
        best_split_add = None
        best_split_feature = None
        best_split_entropy = None
        # Loop through all the features in the point
        for f in range(feature_count):
            # Find the best split for that particular feature
            func, split_input, added, ent = my_split_finder(node.data_points, f,
                                                            self.uniques_dict[f] if f in self.uniques_dict else None)
            # If it is the best so far, store it's info
            if (best_split_entropy is None) or (ent < best_split_entropy):
                best_split_func = func
                best_split_input = split_input
                best_split_feature = f
                best_split_add = added
                best_split_entropy = ent

        # Make a list of the split_function's index given for each data point and store it to a variable
        split_list = [best_split_func(dp, best_split_feature, best_split_input, best_split_add)
                      for dp in node.data_points]

        # Build the list of children nodes of the length of the maximum index given by the split function
        max_split = max(split_list)
        nodes = [Node() for _ in range(max_split + 1)]

        # Give each data point to the node specified by the index ( node.give(dp) )
        for i in range(len(split_list)):
            nodes[split_list[i]].give(node.data_points[i])

        # Set the parent node's split variables
        node.split_function = best_split_func
        node.split_input = best_split_input
        node.split_feature = best_split_feature
        node.split_add = best_split_add

        # Give the nodes list to the node as it's children
        node.set_children(nodes)

        # Clear this node's data_points (=None)
        node.data_points = None

        # Call __grow_node() on each child created
        for child in node.children:
            self.__grow_node(child)

    def __build_class_dict(self, x, y):
        """Build a dictionary to go from index to class string (class_dict), and vice verse (index_dict)"""

        """ -=[Completed.]=- """

        # Lay out the unique classes
        classes = np.unique(y)
        # Build both dictionaries in one pass of the classes
        for i in range(len(classes)):
            self.index_dict[classes[i]] = i
            self.class_dict[i] = classes[i]

        # Store a list of each type of category for each categorical feature
        for i in range(len(x[0])):
            if isinstance(x[0][i], str):
                uni = np.unique([r[i] for r in x])
                temp_dict = {}
                for j in range(len(uni)):
                    temp_dict[uni[j]] = j
                self.uniques_dict[i] = temp_dict

    def __get_data_points(self, x, y=None):
        """Takes in an input matrix X and optionally a label array Y, and returns an array of DataPoints.
        When no Y is given, '0' is filled for all Y values"""

        """ -=[Completed.]=- """

        # If a Y was not given
        if y is None:
            # Fill y with '0', and create data points for each x.  Return list
            return np.array([(DataPoint(x[i], 0)) for i in range(len(x))])
        # If Y WAS given, create data points for each (x, y) pair, and return the list
        return np.array([(DataPoint(x[i], self.index_dict.get(y[i]))) for i in range(len(x))])

    def __calculate_accuracy(self, y_hat, y):
        """Finds where the y_hat is equal to y, and takes the percentage of times when that happens


        Parameters

        -y_hat: np array of ints (class index).  List of predictions in the form of class indices

        -y: array of Strings.  List of class labels from the train() y parameter.  Assumed to be Strings."""

        """ -=[Completed.]=- """

        # Create a new y array, encoded to indices rather than strings
        int_y = np.array([self.index_dict[item] for item in y])

        # Sum up the number of times that the integer form of y is equal to y_hat
        total_same = np.sum(np.where(int_y == y_hat, 1, 0))

        # Set self.accuracy to the percentage of the time that the predictions were right
        self.accuracy = total_same / len(y)

    def __calculate_entropy(self):
        """Calculates the entropy of the tree by recursively starting at the root.
        This assumes the tree creation process is done and tree nodes are flattened"""

        """ -=[Completed.]=- """

        v_e, v_c = sum_node_entropy(self.root)
        self.entropy = v_e / v_c

    def __print_node(self, node):
        tabs = ''.join(['|\t' for _ in range(node.depth)])
        if node.is_leaf:
            print(f'{tabs}Leaf Node  Prediction: {node.prediction};  Depth: {node.depth};  '
                  f'Entropy: {node.entropy};  Data count: {node.data_count};')
        else:
            print(f'{tabs}Parent Node  Children: {len(node.children)};  Depth: {node.depth};  '
                  f'Split Function: {node.split_function.__name__};  '
                  f'Feature: {node.split_feature};  '
                  f'Input: {node.split_input};  '
                  f'Addition: {node.split_add};')
            for child in node.children:
                self.__print_node(child)


def build_from_json(json_obj):
    pass
