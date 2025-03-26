# utils/visualization.py
#
# Visualization utilities for plotting keypoints, brain networks, or EEG signals.

import matplotlib.pyplot as plt

def plot_keypoints(image, keypoints):
    """
    image: a 2D image or array
    keypoints: a list or array of (x, y) coordinates
    Displays the image with keypoints overlaid.
    """
    plt.imshow(image, cmap="gray")
    for x, y in keypoints:
        plt.scatter(x, y, c="red", s=10)
    plt.title("Keypoints Visualization")
    plt.show()

def plot_brain_activity(activation_map):
    """
    activation_map: a 2D array representing brain activity or EEG heatmap
    Displays a heatmap of the activation values.
    """
    plt.imshow(activation_map, cmap="hot")
    plt.colorbar(label="Activation")
    plt.title("Brain Activity Heatmap")
    plt.show()

def plot_training_curve(history):
    """
    history: dictionary or object holding training/validation losses or metrics
    For example: { "train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...] }
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.show()
