# Import necessary libraries
import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Print dataset shape before preprocessing ⚡ ADDED
print(f"Original Training Data Shape: {x_train.shape}")
print(f"Original Testing Data Shape: {x_test.shape}")

# Normalize pixel values (0-255 → 0-1) ⚡ ADDED
x_train, x_test = x_train / 255.0, x_test / 255.0  

# Reshape images to add a single channel (for CNN input) ⚡ ADDED
x_train = x_train.reshape(-1, 28, 28, 1)  
x_test = x_test.reshape(-1, 28, 28, 1)  

# Print dataset shape after preprocessing ⚡ ADDED
print(f"Processed Training Data Shape: {x_train.shape}")
print(f"Processed Testing Data Shape: {x_test.shape}")

# Print sample labels ⚡ ADDED
print("Sample Labels (First 10 Training Labels):", y_train[:10])

