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


def perceptron(training_data, learning_rate, epochs, multiplier, add_bias=True):
    new_training_data = training_data
    if add_bias:
        new_training_data = add_bias_term(training_data)
    weights = np.zeros(new_training_data.shape[1] - 1)
    for _ in range(epochs):
        for datum in new_training_data:
            learning_rate *= multiplier
            feature_vector = datum[1:]
            label = datum[0]
            predicted_label = predict(feature_vector, weights)
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
        predicted_label = predict(feature_vector, weights)
        if not label == predicted_label:
            print(f'Incorrect: Actual {label}, predicted {predicted_label}')
            errors += 1
    return 1 - (errors / len(new_testing_data))


def multiclass_perceptron_accuracy(testing_data, weightsDict):
    errors = 0
    new_testing_data = add_bias_term(testing_data)
    for datum in new_testing_data:
        feature_vector = datum[1:]
        label = datum[0]
        predicted_label = predict_multiclass(feature_vector, weightsDict)
        if not label == predicted_label:
            errors += 1
    return 1 - (errors / len(new_testing_data))


def run_perceptron():
    data = get_binary_data()
    np.random.shuffle(data)
    middle = round(len(data) / 2)
    training_data = data[:middle]
    testing_data = data[middle:]
    weights = perceptron(training_data, 0.5, 86, 0.9)
    print(weights)
    return perceptron_accuracy(testing_data, weights)


def run_multiclass_perceptron(learning_rate, epochs, multiplier):
    data, labels = get_multiclass_data()
    np.random.shuffle(data)
    middle = round(len(data) / 2)
    training_data = data[:middle]
    testing_data = data[middle:]
    weightsDict = multiclass_perceptron(
        training_data, learning_rate, epochs, multiplier, labels)
    return multiclass_perceptron_accuracy(testing_data, weightsDict)


def individual(learning_rate, epochs, multiplier, runs=7):
    sum = 0
    for _ in range(runs):
        sum += run_multiclass_perceptron(learning_rate, epochs, multiplier)
    return (sum / runs, learning_rate, epochs, multiplier)


def optimize_multiclass_perceptron(population_size):
    population = []
    for i in range(population_size):
        learning_rate = random()
        epochs = randint(1, 100)
        multiplier = min(1, random() + 0.5)
        print(f'Running organism {i + 1}')
        population.append(individual(learning_rate, epochs, multiplier))
    best_individual = float('-inf')
    best_learning_rate = float('-inf')
    best_epochs = float('-inf')
    best_multiplier = float('-inf')
    for organism in population:
        if organism[0] > best_individual:
            best_individual, best_learning_rate, best_epochs, best_multiplier = organism
    print(best_individual, best_learning_rate, best_epochs, best_multiplier)


optimize_multiclass_perceptron(50)
print(run_perceptron())
"""
learning rate: 
epochs:
multiplier:


"""
