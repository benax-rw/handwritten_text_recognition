import cv2
import numpy as np
import tensorflow as tf
import time

# Load the trained model safely
try:
    model = tf.keras.models.load_model("mnist_model.keras", compile=False)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("Model loaded and recompiled successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Attempt to open webcam up to 30 times
max_attempts = 30
attempts = 0
cap = None

while attempts < max_attempts:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print(f"Camera opened successfully after {attempts + 1} attempt(s).")
        break
    print(f"⚠️ Camera failed to open. Retrying ({attempts + 1}/{max_attempts})...")
    time.sleep(1)  # Wait 1 second before retrying
    attempts += 1

# Exit if camera couldn't be opened
if not cap or not cap.isOpened():
    print("Failed to open camera after multiple attempts. Exiting...")
    exit()

# Define ROI coordinates (larger area for better accuracy)
ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT = 50, 50, 600, 600

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera frame not received. Retrying...")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw a transparent rectangle as the Region of Interest (ROI)
    overlay = frame.copy()
    cv2.rectangle(overlay, (ROI_X, ROI_Y), (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT), (0, 255, 0), 4)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Extract only the ROI pixels
    roi = gray[ROI_Y:ROI_Y + ROI_HEIGHT, ROI_X:ROI_X + ROI_WIDTH]

    # Preprocess for MNIST model
    roi = cv2.resize(roi, (28, 28))  # Resize to MNIST input size
    roi = cv2.bitwise_not(roi)  # Invert colors (MNIST expects white digits on black background)
    roi = roi / 255.0  # Normalize pixel values
    roi = roi.reshape(1, 28, 28, 1)  # Reshape for CNN input

    # Predict digit
    predictions = model.predict(roi)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Convert highest probability to percentage

    # Only display prediction if confidence is ≥ 75%
    if confidence >= 75:
        text = f"Predicted: {predicted_digit} ({confidence:.2f}%)"
    else:
        text = "Not confident enough"

    # Display result
    cv2.putText(frame, text, (ROI_X, ROI_Y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam feed with ROI overlay
    cv2.imshow("Live MNIST Digit Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()