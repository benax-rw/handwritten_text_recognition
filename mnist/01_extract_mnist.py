import numpy as np
import os
import cv2  # OpenCV for saving images
from tensorflow.keras.datasets import mnist

# Load MNIST from disk (no need to download again)
mnist_path = os.path.expanduser("~/.keras/datasets/mnist.npz")

with np.load(mnist_path) as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

# Create folders to store images
train_folder = "mnist_train"
test_folder = "mnist_test"
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Function to save images
def save_images(images, labels, folder):
    for i in range(len(images)):
        filename = f"{folder}/{labels[i]}_{i}.png"
        cv2.imwrite(filename, images[i])  # Save as PNG
    print(f"Saved {len(images)} images in {folder}")

# Save training and test images
save_images(x_train, y_train, train_folder)
save_images(x_test, y_test, test_folder)

print("✅ MNIST images extracted and saved as PNG files!")