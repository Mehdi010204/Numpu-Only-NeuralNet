import numpy as np
from .layers import Dense, ReLU, Softmax
from .utils import one_hot_encode, compute_accuracy

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.fc1 = Dense(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Dense(hidden_size, output_size)
        self.softmax = Softmax()
        self.w1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.w2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.zeros((output_size, 1))
        self.accuracies = []

    def forward(self, X):
        Z1 = self.fc1.forward(X)
        A1 = self.relu.forward(Z1)
        Z2 = self.fc2.forward(A1)
        A2 = self.softmax.forward(Z2)
        self.cache = (Z1, A1, Z2, A2)
        return A2

    def backward(self, X, Y):
        Z1, A1, Z2, A2 = self.cache
        m = Y.size
        one_hot_Y = one_hot_encode(Y)
        dZ2 = A2 - one_hot_Y
        dA1 = self.fc2.backward(dZ2, self.learning_rate)
        dZ1 = self.relu.backward(dA1)
        self.fc1.backward(dZ1, self.learning_rate)

    def predict(self, X):
        A2 = self.forward(X)
        return np.argmax(A2, axis=0)

    def fit(self, X, Y, iterations):
        for i in range(iterations):
            self.forward(X)
            self.backward(X, Y)
            if i % 100 == 0:
                preds = self.predict(X)
                acc = compute_accuracy(preds, Y)
                self.accuracies.append(acc)
                print(f"Iteration {i}, Accuracy: {acc:.4f}")

    def save_model(self, filename="model.npz"):
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load_model(self, filename="model.npz"):
        data = np.load(filename)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
