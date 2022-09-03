
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


