import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load raw MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Select a sample image (before preprocessing)
sample_index = 0  # First image in the dataset
raw_image = x_train[sample_index]

# Normalize and reshape the image (preprocessing)
normalized_image = raw_image / 255.0  # Normalize pixel values

# Display some pixel values before and after normalization
print("Pixel values BEFORE normalization (center region):")
print(raw_image[10:15, 10:15])  # Print a 5x5 section of the image

print("\nPixel values AFTER normalization (center region):")
print(normalized_image[10:15, 10:15])  # Print the same section after normalization

# Plot before and after preprocessing
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Before preprocessing
axes[0].imshow(raw_image, cmap='gray', vmin=0, vmax=255)  # Explicitly setting min/max values
axes[0].set_title("Before Preprocessing (0-255)")
axes[0].axis('off')

# After preprocessing
axes[1].imshow(normalized_image, cmap='gray', vmin=0, vmax=1)  # Explicitly setting min/max values
axes[1].set_title("After Preprocessing (0-1)")
axes[1].axis('off')

plt.show()