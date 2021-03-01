import numpy as np
from data import get_binary_data, get_multiclass_data, one_versus_all
from linear_algebra import normalize
from random import random, randint


def predict(feature_vector, weights):
    return 1 if feature_vector.dot(weights) > 0 else -1


def predict_multiclass(feature_vector, weightsDict):
    prediction = float('-inf')
    best_key = None
    for key in weightsDict.keys():
        product = feature_vector.dot(weightsDict[key])
        if product > prediction:
            prediction = product
            best_key = key
    return best_key


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
    weightsDict = {}
    for label in labels:
        new_training_data = one_versus_all(training_data, label)
        weightsDict[label] = perceptron(
            new_training_data, learning_rate, epochs, multiplier)
    return weightsDict


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
    weightsDict = multiclass_perceptron(training_data, 0.8, 50, 0.99, labels)
    print(weightsDict)
    return multiclass_perceptron_accuracy(testing_data, weightsDict)


def individual(training_data, labels, iter=10):
    sum = 0
    for _ in range(iter):
        sum += multiclass_perceptron(training_data, random(),
                                     randint(1, 100), min(1, random() + 0.5), labels)


print(run_multiclass_perceptron())
