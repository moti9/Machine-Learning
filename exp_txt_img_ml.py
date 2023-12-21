import cv2
import pytesseract
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# Load the image
image_path = "IELTS-template.jpg"
img = cv2.imread(image_path)

# Extract text using OCR
text = pytesseract.image_to_string(img)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to recognize facial expression using the trained model
def recognize_facial_expression(image):
    # Resize and preprocess the image
    img = cv2.resize(image, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = np.expand_dims(img, axis=-1)  # Add a channel dimension
    img = img / 255.0  # Normalize pixel values to between 0 and 1

    # Predict facial expression
    predictions = model.predict(np.expand_dims(img, axis=0))
    
    # Get the predicted emotion label
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    
    return predicted_emotion

# Build and train a simple facial expression recognition model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 classes for facial expressions
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load and preprocess your dataset (replace 'X_train' and 'y_train' with your data)
# ...

# Train the model
# model.fit(X_train, y_train, epochs=10, validation_split=0.2)
# Replace with actual training data when available

# Function to recognize facial expression using the trained model
def recognize_facial_expression(image):
    # Resize and preprocess the image
    img = cv2.resize(image, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = np.expand_dims(img, axis=-1)  # Add a channel dimension
    img = img / 255.0  # Normalize pixel values to between 0 and 1

    # Predict facial expression
    predictions = model.predict(np.expand_dims(img, axis=0))
    
    # Get the predicted emotion label
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    
    return predicted_emotion

# Perform facial expression recognition
facial_expression = recognize_facial_expression(img)

# Display the results
print("Extracted Text:", text)
print("Facial Expression:", facial_expression)

# (Optional) Save or display the image with annotated results
# ...

# (Optional) You can also create a video explaining the code and results
# ...
