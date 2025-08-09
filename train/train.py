import pandas as pd
import numpy as np
from neuralnet.model import NeuralNetwork
from neuralnet.utils import normalize
from neuralnet.visualize import *

# Load data
train_data = pd.read_csv("data/mnist_train.csv").to_numpy()
test_data = pd.read_csv("data/mnist_test.csv").to_numpy()

# Preprocess
X_train = normalize(train_data[:, 1:].T)
Y_train = train_data[:, 0]
X_test = normalize(test_data[:, 1:].T)
Y_test = test_data[:, 0]

# Show sample images
show_sample_images(X_train.T, Y_train, samples=10)

# Train
model = NeuralNetwork(input_size=784, hidden_size=64, output_size=10, learning_rate=0.1)
model.fit(X_train, Y_train, iterations=1500)

# Test
acc = np.mean(model.predict(X_test) == Y_test)
print(f"Test Accuracy: {acc:.4f}")

# plot_accuracy_curve(model.accuracies)
losses = [1 - acc for acc in model.accuracies]  
# plot_loss_curve(losses)

# Visualize weights
# visualize_weights(model.fc1.W, num_neurons=10)


# Save model
#model.save_model("model_weights.npz")