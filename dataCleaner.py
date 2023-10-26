
import pandas as pd
import numpy as np

# -------------------------------- Helper methods and object creation ----------------------------------------
MF2num = dict()
MF2num['M'] = 0
MF2num['F'] = 1

str2int = dict()
str2int['two'] = 2
str2int['three'] = 3
str2int['four'] = 4
str2int['five'] = 5
str2int['six'] = 6
str2int['eight'] = 8
str2int['twelve'] = 12

str2date = dict()
str2date['Jan'] = 1
str2date['Feb'] = 2
str2date['Mar'] = 3
str2date['Apr'] = 4
str2date['May'] = 5
str2date['Jun'] = 6
str2date['Jul'] = 7
str2date['Aug'] = 8
str2date['Sep'] = 9
str2date['Oct'] = 10
str2date['Nov'] = 11
str2date['Dec'] = 12

level2num = dict()
level2num['LOW'] = 0
level2num['NORMAL'] = 1
level2num['HIGH'] = 2


def data_normalize(data):
    maxes = np.max(data, axis=0)
    return data / maxes


def replace_all_dict(_list, _dict, default):
    for i in range(len(_list)):
        result = _dict.get(_list[i])
        if result is not None:
            _list[i] = result
        else:
            _list[i] = default


def get_onehot_piece(_list, item):
    new_list = []
    for i in range(len(_list)):
        if _list[i] == item:
            new_list.append(1)
        else:
            new_list.append(0)
    return new_list

# Notation for this: ar_clean = ar[ar[:, 1] >= ar[:, 2]]
# --------------------------------------------------- WORKING CODE AREA ------------------------------------------


drug2index = dict()
drug2index['drugA'] = 0
drug2index['drugB'] = 1
drug2index['drugC'] = 2
drug2index['drugX'] = 3
drug2index['drugY'] = 4

file = pd.read_csv('./datasets/drug200_full_clean.csv')
print(file.head())
print(file.corr()['Drug A'])


"""Not part of data cleaner.  For use for multiple monitors"""


def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(j) for j in data])

    # Mask of valid places in each row
    mask = np.arange(max(lens)) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape)
    out[mask] = np.concatenate(data)
    return out


def try_split_virtual_vector(split_func, feature, split_in, added, data_points):
    # Build list of split indices  (7.6 seconds)
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






"""

Normal:
3 generators (data_points, node_count, node_count)
1 loop (data_points)
    to give each data point to it's corresponding node
1 loop (node count)
    within entropy_raw to recursively call functions
1 generator (data points)
1 loop (class count)
    to add totals multiplied by log2 to the entropy
Loops:
    Data Points: 1
    Node Count: 1
    Class Count: 1
Generators:
    Data Points: 2
    Node Count: 2
    Class Count: 0


Virtual:
1 generator (data_points)
1 loop (data_points)
    to add each point's y value to the virtual node row's y values
1 loop (node count)
    find uniques, get sum, regularize counts, calculate entropy, find total entropy for node
1 generator (node count)
Loops:
    Data Points: 1
    Node Count: 1
    Class Count: 0
Generators:
    Data Points: 1
    Node Count: 1
    Class Count: 0


Vector:
1 loop (data_points)
    build index list and record maximum y value
1 loop (data_points)
    build y_s values list from the data gathered in loop 1
Loops:
    Data Points: 2
    Node Count: 0
    Class Count: 0
Generators:
    Data Points: 0
    Node Count: 0
    Class Count: 0


Over_Pad:
1 loop (data_points)
    with over-sized matrix, set the element corresponding to split index and class to e+=1

TEST:
maybe start with the setup of virtual, then go into vector once line 285 starts on the virtual

"""




























