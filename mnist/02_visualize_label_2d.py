import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Find an image with label '5'
index = np.where(y_train == 5)[0][0]  # Get the first occurrence of label 5
digit = x_train[index]  # Extract the digit image

# Create a 2D visualization (before reshaping)
plt.figure(figsize=(5, 5))
plt.imshow(digit, cmap='gray')
plt.title("2D Visualization of MNIST Digit '5' (Before Reshaping)")
plt.axis('on')  # Hide axis for better visualization
plt.show()