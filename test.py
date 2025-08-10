import numpy as np
from neuralnet.model import NeuralNetwork

X_dummy = np.random.rand(784, 5)  # shape: (features, samples)
Y_dummy = np.array([1, 3, 4, 0, 7])  

model = NeuralNetwork(input_size=784, hidden_size=64, output_size=10)

predictions = model.predict(X_dummy)

print("Predictions :", predictions)
print("Predictions shape :", predictions.shape)

assert predictions.shape == (5,), "Output should be of shape (5,)"
