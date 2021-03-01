import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import get_binary_data
from linear_algebra import normalize


def predict(feature_vector, weights):
    return 1 if feature_vector.dot(weights) > 0 else -1


def perceptron(training_data, learning_rate, epochs, multiplier):
    new_training_data = []
    for datum in training_data:
        datum = list(datum)
        datum.insert(1, 1)
        new_training_data.append(datum)
    new_training_data = np.array(new_training_data)
    weights = np.zeros(new_training_data.shape[1] - 1)
    for _ in range(epochs):
        for datum in new_training_data:
            learning_rate *= multiplier
            feature_vector = datum[1:]
            label = datum[0]
            predicted_label = 1 if feature_vector.dot(weights) else -1
            if not label == predicted_label:
                weights += (learning_rate * label * feature_vector)
    return normalize(weights)


def multiclass_predict(feature_vector, weights):
    dot_product_dict = {}
    for key in weights.keys():
        dot_product_dict[key] = feature_vector.dot(weights[key])


def multiclass_perceptron_with_intercept(training_data, learning_rate, epochs, multiplier, labels):
    new_training_data = []
    for datum in training_data:
        datum = list(datum)
        # insert a new constant feature so we can go through the origin
        datum.insert(1, 1)
        new_training_data.append(datum)
    new_training_data = np.array(new_training_data)
    weights = {}
    for label in range(labels):
        weights[label] = np.zeros(new_training_data.shape[1] - 1)
    for _ in range(epochs):
        for datum in new_training_data:
            learning_rate *= multiplier
            # Assume the last element in a data point is the label
            # The rest are the features
            feature_vector = datum[1:]
            for f_index in range(len(feature_vector)):
                if np.isnan(feature_vector[f_index]):
                    feature_vector[f_index] = 0
            label = datum[0]
            predicted_label = 1 if feature_vector.dot(
                weights[:len(feature_vector)]) > 0 else -1
            if not label == predicted_label:
                weights = weights + learning_rate * label * feature_vector
    return normalize(weights)


def perceptron_accuracy(testing_data, weights):
    errors = 0
    new_testing_data = []
    for datum in testing_data:
        datum = list(datum)
        # insert a new constant feature so we can go through the origin
        datum.insert(1, 1)
        new_testing_data.append(datum)
    new_testing_data = np.array(new_testing_data)
    for datum in new_testing_data:
        feature_vector = datum[1:]
        label = datum[0]
        predicted_label = 1 if feature_vector.dot(weights) else -1
        if not label == predicted_label:
            errors += 1
    return errors / len(testing_data)


def run_perceptron():
    data = get_binary_data()
    np.random.shuffle(data)
    middle = round(len(data) / 2)
    training_data = data[:middle]
    testing_data = data[middle:]
    weights = perceptron(training_data, 0.5, 86, 0.9)
    print(weights)
    return perceptron_accuracy(testing_data, weights)


print(run_perceptron())
