import numpy as np
from copy import deepcopy


def normalize(vector):
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
