import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Display dataset shape
print(f"Training Data Shape: {x_train.shape}, Labels: {y_train.shape}")
print(f"Testing Data Shape: {x_test.shape}, Labels: {y_test.shape}")

# Visualize some sample digits
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i in range(5):
    axes[i].imshow(x_train[i], cmap='gray')
    axes[i].set_title(f"Label: {y_train[i]}")
    axes[i].axis('off')
plt.show()