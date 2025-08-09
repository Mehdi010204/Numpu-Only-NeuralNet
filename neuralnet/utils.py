import numpy as np

def one_hot_encode(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def predict(output):
    return np.argmax(output, axis=0)

def compute_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def normalize(X):
    return X / 255.0
