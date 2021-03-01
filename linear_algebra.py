import numpy as np
from copy import deepcopy


def normalize(vector):
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def normalize_dict(weight_dict):
    new_weight_dict = deepcopy(weight_dict)
    for key in new_weight_dict.keys():
        new_weight_dict[key] = normalize(weight_dict[key])
    return new_weight_dict
