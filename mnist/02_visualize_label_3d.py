import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Find an image with label '5'
index = np.where(y_train == 8)[0][0]  # Get the first occurrence of label 8
digit = x_train[index]  # Extract the digit image

# Normalize the grayscale image (0-1)
normalized_digit = digit / 255.0  

# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Generate X, Y coordinate grids
X, Y = np.meshgrid(range(28), range(28))

# Flip the Y-axis for correct orientation
ax.plot_surface(X, 28 - Y, normalized_digit, cmap='gray', edgecolor='k')

# Labels and title
ax.set_title("3D Visualization of MNIST Digit '8'")
ax.set_xlabel("Width (Pixels)")
ax.set_ylabel("Height (Pixels)")
ax.set_zlabel("Pixel Intensity (0-1)")
plt.axis('off')  # Hide axis for better visualization

plt.show()