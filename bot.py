import tensorflow as tf
import numpy as np
import pyautogui
import time
import win32api
import win32con

# Define the input shape
input_shape = (224, 224, 3)

# Define the learning rate
learning_rate = 0.001

# Define the batch size
batch_size = 32

# Define the number of epochs
num_epochs = 10

# Define the flag to switch between learning and bot phases
learn_mode = True

# Create a dictionary to map labels to integers
label_to_int = {}

# Create a function that takes a screenshot and preprocesses the image
def preprocess_image():
    # Take a screenshot
    screenshot = pyautogui.screenshot()
    # Get the current position of the mouse
    mouse_pos = win32api.GetCursorPos()
    # Convert the image to the RGB format
    rgb_image = np.array(screenshot)
    # Draw a red dot at the mouse position
    rgb_image[mouse_pos[1], mouse_pos[0], :] = [255, 0, 0]
    # Resize the image to a fixed size
    resized_image = tf.image.resize(rgb_image, size=(224, 224))
    # Normalize the pixel values
    normalized_image = resized_image / 255.0
    # Add a batch dimension to the image
    preprocessed_image = tf.expand_dims(normalized_image, axis=0)
    return preprocessed_image

# Create a neural network
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_to_int), activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define a threshold for the model confidence
confidence_threshold = 0.8

# Define a loop that collects data, trains the model, and evaluates its accuracy
while learn_mode:
    # Check if left mouse button is clicked
    if win32api.GetKeyState(win32con.VK_LBUTTON) < 0:
        preprocessed_image = preprocess_image()
        label = input("Enter the label for the current action: ")
        # Map the label to an integer
        label_int = label_to_int.setdefault(label, len(label_to_int))
        num_actions = len(label_to_int)
        label_one_hot = np.zeros((1, num_actions))
        label_one_hot[0, label_int] = 1
        # Update the output layer of the model to match the number of actions
        model.layers[-1] = tf.keras.layers.Dense(num_actions, activation='softmax')
        # Train the model on the data
        model.train_on_batch(preprocessed_image, label_one_hot)
    else:
        # Take a screenshot
        preprocessed_image = preprocess_image()
        # Make a prediction
        prediction = model.predict(preprocessed_image)
        predicted_label = None
        predicted_labels = {}
        for label, index in label_to_int.items():
            confidence = prediction[0][index
