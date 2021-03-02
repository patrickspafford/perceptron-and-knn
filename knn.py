import numpy as np
from collections import Counter
from copy import deepcopy
from data import get_binary_data, get_multiclass_data


def distance(vector_1, vector_2):
    return np.linalg.norm(abs(vector_2 - vector_1))


def get_k_nearest_labels(training_data, test_datum, k):
    distances = list()
    for training_datum in training_data:
        training_feature_vector = training_datum[1:]
        training_label = training_datum[0]
        test_feature_vector = test_datum[1:]
        distances.append((training_label, distance(
            training_feature_vector, test_feature_vector)))
    k_closest_distances = sorted(distances, key=lambda x: x[1])[:k]
    return list(map(lambda x: x[0], k_closest_distances))


def get_most_common_label(labels):
    data = Counter(labels)
    return int(max(labels, key=data.get))


def knn(training_data, test_data, k):
    new_test_data = []
    for test_datum in test_data:
        new_test_datum = deepcopy(test_datum)
        new_test_datum[0] = get_most_common_label(
            get_k_nearest_labels(training_data, new_test_datum, k))
        new_test_data.append(new_test_datum)
    return new_test_data


def knn_accuracy(testing_data, predicted_data):
    errors = 0
    for i in range(len(testing_data)):
        label = testing_data[i][0]
        predicted_label = predicted_data[i][0]
        if not label == predicted_label:
            errors += 1
            print(f'Actual {label}, predicted {predicted_label}')
    return 1 - (errors / len(testing_data))


def run_knn(is_multiclass=False):
    data = get_binary_data() if not is_multiclass else get_multiclass_data(
        is_np=True)[0]
    np.random.shuffle(data)
    middle = round(len(data) / 2)
    training_data = data[:middle]
    testing_data = data[middle:]
    predicted_data = knn(training_data, testing_data, k=7)
    return knn_accuracy(testing_data, predicted_data)


print(run_knn(is_multiclass=True))
