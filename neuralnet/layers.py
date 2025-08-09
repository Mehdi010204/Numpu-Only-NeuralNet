import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.b = np.zeros((output_size, 1))
    
    def forward(self, X):
        self.X = X
        return np.dot(self.W, X) + self.b
    
    def backward(self, dZ, learning_rate):
        m = self.X.shape[1]
        dW = np.dot(dZ, self.X.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return np.dot(self.W.T, dZ)

class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)
    
    def backward(self, dA):
        return dA * (self.Z > 0)

class Softmax:
    def forward(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)
