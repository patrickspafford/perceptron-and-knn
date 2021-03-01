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
