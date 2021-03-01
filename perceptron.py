import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import get_binary_data, get_multiclass_data
from linear_algebra import normalize, normalize_dict


def predict(feature_vector, weights):
    return 1 if feature_vector.dot(weights) > 0 else -1


def predict_multiclass(feature_vector, weightsDict):
    dotProductDict = {}
    for key in weightsDict.keys():
        dotProductDict[key] = feature_vector.dot(weightsDict[key])
    maxDotProduct = float('-inf')
    maxKey = None
    for key in dotProductDict.keys():
        if dotProductDict[key] > maxDotProduct:
            maxDotProduct = dotProductDict[key]
            maxKey = key
    return maxKey


def add_bias_term(data):
    new_training_data = []
    for datum in data:
        datum = list(datum)
        datum.insert(1, 1)
        new_training_data.append(datum)
    return np.array(new_training_data)


def perceptron(training_data, learning_rate, epochs, multiplier):
    new_training_data = add_bias_term(training_data)
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


def multiclass_perceptron(training_data, learning_rate, epochs, multiplier, labels):
    new_training_data = add_bias_term(training_data)
    weightsDict = {}
    for label in labels:
        weightsDict[label] = np.zeros(new_training_data.shape[1] - 1)
    for _ in range(epochs):
        for datum in new_training_data:
            learning_rate *= multiplier
            feature_vector = datum[1:]
            label = datum[0]
            predicted_label = predict_multiclass(feature_vector, weightsDict)
            if not label == predicted_label:
                update_vector = (learning_rate * feature_vector)
                weightsDict[label] += update_vector
                weightsDict[predicted_label] -= update_vector
    return normalize_dict(weightsDict)


def perceptron_accuracy(testing_data, weights):
    errors = 0
    new_testing_data = add_bias_term(testing_data)
    for datum in new_testing_data:
        feature_vector = datum[1:]
        label = datum[0]
        predicted_label = 1 if feature_vector.dot(weights) else -1
        if not label == predicted_label:
            errors += 1
    return errors / len(new_testing_data)


def multiclass_perceptron_accuracy(testing_data, weightsDict):
    errors = 0
    new_testing_data = add_bias_term(testing_data)
    for datum in new_testing_data:
        feature_vector = datum[1:]
        label = datum[0]
        predicted_label = predict_multiclass(feature_vector, weightsDict)
        if not label == predicted_label:
            errors += 1
    return errors / len(new_testing_data)


def run_perceptron():
    data = get_binary_data()
    np.random.shuffle(data)
    middle = round(len(data) / 2)
    training_data = data[:middle]
    testing_data = data[middle:]
    weights = perceptron(training_data, 0.5, 86, 0.9)
    print(weights)
    return perceptron_accuracy(testing_data, weights)


def run_multiclass_perceptron():
    data, labels = get_multiclass_data()
    np.random.shuffle(data)
    middle = round(len(data) / 2)
    training_data = data[:middle]
    testing_data = data[middle:]
    weights = multiclass_perceptron(training_data, 0.1, 20, 0.85, labels)
    print(weights)
    return multiclass_perceptron_accuracy(testing_data, weights)


# print(run_perceptron())
print(run_multiclass_perceptron())
