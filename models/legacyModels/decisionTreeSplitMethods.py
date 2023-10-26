from models.decisionTree import *

# These are all the methods that I tried to see if they'd be faster than the current one used in
# the decision tree class (virtual_vector).  Breakout of how my testing went:

# (Best of 3 average)
# Normal (Loop):    11.7 seconds
# Virtual:           7.6 seconds
# Virtual vector:    7.6 seconds   [WINNER for me]
# Vector:           10.6 seconds
# Vector Padded:     8.9 seconds
# MY PERSONAL WINNER: Virtual Vector


def try_split(split_func, feature, split_in, added, data_points):
    # Build list of split indices
    split_indices = [split_func(p, feature, split_in, added) for p in data_points]

    # Make list of nodes of size: max(indices list)
    max_index = np.max(split_indices)
    nodes = [Node() for _ in range(max_index + 1)]

    # Give each data point to the corresponding node
    for i in range(len(split_indices)):
        nodes[split_indices[i]].give(data_points[i])

    # Give nodes to test node
    root = Node()
    root.set_children(nodes)

    # Set each child node as a leaf
    for child in root.children:
        child.is_leaf = True

    # use entropy() on the test node and return the result
    return entropy_raw(root)[0]


def try_split_virtual(split_func, feature, split_in, added, data_points):
    # Build list of split indices
    split_indices = [split_func(p, feature, split_in, added) for p in data_points]
    max_index = 2 if added is None else len(added)

    # Get a matrix of class values per node
    y_s = [[] for _ in range(max_index)]
    for i in range(len(data_points)):
        y_s[split_indices[i]].append(data_points[i].y)

    # Deal with the part of the equation that, if vectorized, would result in a staggered matrix size
    total_entropy = 0  # Total entropy for this split
    # Loop through each 'node'
    for r in y_s:
        # As long as this node got at least a point
        if len(r) > 0:
            # Find unique class counts (No need for the actual labels)
            part = np.unique(r, return_counts=True)[1]
            # Store their sum
            part_sum = np.sum(part)
            # Divide each count by the sum (so sum(part)=1)
            part = part / part_sum.astype(float)
            # For each item, multiply the class's probability by the log base 2 of that probability and sum total
            part_ent_sum = np.sum(part * np.log2(part))
            # Multiply that sum by the number of data points in this 'node'
            part_ent_sum *= part_sum
            # Append final entropy and it's count to the matrix
            total_entropy += part_ent_sum

    # Return the total weighted entropy of this split
    return -total_entropy


def try_split_vector(split_func, feature, split_in, added, data_points):
    split_indices = [split_func(p, feature, split_in, added) for p in data_points]
    max_y = max(data_points, key=attrgetter('y')).y
    max_index = max(split_indices)
    # Build list of split indices, maximum Y value seen, and maximum split index given

    # Get a list of class occurrences per 'node'
    y_s = np.zeros((max_index + 1, max_y + 1))
    for i in range(len(data_points)):
        y_s[split_indices[i], data_points[i].y] += 1

    # Store their sums
    y_sums = np.sum(y_s, axis=1)
    # Divide each count by the sum (so sum(part)=1)
    y_s = np.divide(y_s.T, y_sums.T, where=(y_sums > 0)).T
    # For each item, multiply the class's probability by the log base 2 of that probability and sum total
    ent_sums = np.sum(y_s * np.log2(y_s, where=(y_s > 0)), axis=1)
    # Multiply that sum by the number of data points in each node
    ent_sums *= y_sums
    # Calculate the total weighted entropy of this split
    total_ent = np.sum(ent_sums)

    # Return the entropy (negative)
    return -total_ent


def try_split_vector_over_pad(split_func, feature, split_in, added, data_points):
    padding_size = 20
    # Get a list of class occurrences per 'node'
    y_s = np.array([[0 for _ in range(padding_size)] for _ in range(padding_size)])
    for dp in data_points:
        y_s[dp.y, split_func(dp, feature, split_in, added)] += 1

    # Store their sums
    y_sums = np.sum(y_s, axis=0)
    # Divide each count by the sum (so sum(part)=1)
    y_s = np.divide(y_s, y_sums, where=(y_sums > 0)).T
    # For each item, multiply the class's probability by the log base 2 of that probability and sum total
    ent_sums = np.sum(y_s * np.log2(y_s, where=(y_s > 0)), axis=1)
    # Multiply that sum by the number of data points in each node
    ent_sums *= y_sums
    # Calculate the total weighted entropy of this split
    total_ent = np.sum(ent_sums)

    # Return the entropy (negative)
    return -total_ent



