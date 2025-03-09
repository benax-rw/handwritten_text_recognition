from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),  # ⚡ ADDED (First Convolution Layer)
    MaxPooling2D(2,2),  # ⚡ ADDED (First Pooling Layer)
    Conv2D(64, (3,3), activation='relu'),  # ⚡ ADDED (Second Convolution Layer)
    MaxPooling2D(2,2),  # ⚡ ADDED (Second Pooling Layer)
    Flatten(),  # ⚡ ADDED (Flattening for Dense Layer)
    Dense(128, activation='relu'),  # ⚡ ADDED (Fully Connected Layer)
    Dense(10, activation='softmax')  # ⚡ ADDED (Output Layer with 10 Classes)
])

# Compile the model ⚡ ADDED
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display model summary ⚡ ADDED
model.summary()