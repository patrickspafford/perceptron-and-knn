from copy import deepcopy
import numpy as np
import pandas as pd

BINARY_CLASSIFICATION = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a'
MULTICLASS_CLASSIFICATION = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale'


def get_binary_data():
    df = pd.read_table(BINARY_CLASSIFICATION, header=None, sep=' ')
    data = np.array(df.drop(df.columns[[15]], axis=1))
    new_data = []
    for row in data:
        new_row = [0] * 125
        new_row[0] = row[0]
        for cell in row:
            if isinstance(cell, str):
                new_row[int(cell.split(':')[0])] = 1
        new_data.append(new_row)
    return np.array(new_data)


def get_multiclass_data(is_np=False):
    df = pd.read_table(MULTICLASS_CLASSIFICATION, header=None, sep=' ')
    data = np.array(df.drop(df.columns[[5]], axis=1))
    new_data = []
    labels = set()
    for row in data:
        new_row = [0] * 5
        new_label = row[0]
        new_row[0] = new_label
        labels.add(new_label)
        for cell in row:
            if isinstance(cell, str):
                feature = int(cell.split(':')[0])
                value = float(cell.split(':')[1])
                new_row[feature] = value
        if is_np:
            new_row = np.array(new_row)
        new_data.append(new_row)
    return new_data, list(labels)


def one_versus_all(data, label):
    new_data = []
    for row in data:
        new_row = deepcopy(row)
        new_row[0] = 1 if label == row[0] else -1
        new_data.append(new_row)
    return new_data

    # print(get_multiclass_data()[:5])
