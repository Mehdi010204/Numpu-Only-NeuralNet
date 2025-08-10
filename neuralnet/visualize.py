import matplotlib.pyplot as plt
import numpy as np

def show_sample_images(X, Y, samples=10):
    plt.figure(figsize=(10, 2))
    for i in range(samples):
        plt.subplot(1, samples, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap="gray")
        plt.title(str(Y[i]))
        plt.axis("off")
    plt.show()


def plot_accuracy_curve(accuracies):
    plt.figure(figsize=(6, 4))
    plt.plot(accuracies, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss_curve(losses):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_weights(weights, num_neurons=10):
    plt.figure(figsize=(10, 4))
    for i in range(num_neurons):
        plt.subplot(2, num_neurons // 2, i + 1)
        plt.imshow(weights[i].reshape(28, 28), cmap="seismic", interpolation="nearest")
        plt.axis("off")
        plt.title(f"N{i}")
    plt.show()