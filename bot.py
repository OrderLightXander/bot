import cv2
import numpy as np
import keyboard
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from PIL import ImageGrab

# Define the resources that you want to identify
resources = ['fiber', 'stone', 'ore', 'wood']

# Load the training images and labels
X_train = []
y_train = []
for resource in resources:
    for i in range(8):
        img = cv2.imread(f'{resource}{i+1}.png', cv2.IMREAD_GRAYSCALE)
        X_train.append(img)
        y_train.append(resources.index(resource))

# Convert the training data to numpy arrays and normalize the pixel values
X_train = np.array(X_train).reshape(-1, 128, 128, 1) / 255.0
y_train = to_categorical(y_train)

# Define the neural network model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(resources), activation='softmax'))

# Compile the model and train it on the training data
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Additional function to save the image with the correct resource label
def save_corrected_image(screenshot, correct_resource):
    img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 4)  # Change thickness to 4
    if not os.path.exists(f'./{correct_resource}'):
        os.makedirs(f'./{correct_resource}')
    filename = f"{correct_resource}_{len(os.listdir(f'./{correct_resource}')) + 1}.png"
    cv2.imwrite(f"./{correct_resource}/{filename}", screenshot)
    print(f"Corrected image saved as {filename}")
    return filename



# Define the function to predict the resource in a scryeenshot and display a rectangle around the predicted resource
def predict_resource(screenshot):
    img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    img = img.astype(np.uint8)  # convert data type to uint8
    prediction = model.predict(np.array([img]))
    resource = resources[np.argmax(prediction)]
    print(f'Resource: {resource}')
    # Convert the image to binary format and find the contours of the predicted resource
    gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # convert back to BGR
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)  # convert BGR to grayscale
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(screenshot, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return screenshot



# Take a screenshot when the "p" key is pressed and predict the resource in the screenshot
# Modified on_press function
def on_press():
    global X_train, y_train

    print('Taking screenshot...')
    screenshot = np.array(ImageGrab.grab())
    predicted_screenshot = predict_resource(screenshot)
    cv2.imshow('Screenshot', predicted_screenshot)
    
    # Save the screenshot with the rectangle around the resource
    filename = f"screenshot_{len(os.listdir('./screenshots')) + 1}.png"
    cv2.imwrite(f"./screenshots/{filename}", predicted_screenshot)
    
    # Display the saved image and ask the user for input
    print(f"Is the prediction correct? (y/n)")
    user_input = input()
    if user_input.lower() == "n":
        print("What is the correct resource?")
        correct_resource = input().strip().lower()
        corrected_filename = save_corrected_image(predicted_screenshot, correct_resource)

        # Retrain the model with the new image
        img_path = f"./{correct_resource}/{corrected_filename}"
        print(f"Loading image from: {img_path}")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))  # Resize the image
        img = img.reshape(128, 128, 1)  # Add the channel dimension
        X_train = np.append(X_train, img[np.newaxis, :], axis=0)
        y_train = np.append(y_train, to_categorical(resources.index(correct_resource), num_classes=len(resources))[np.newaxis, :], axis=0)
        model.fit(X_train, y_train, epochs=1, batch_size=32)

        # Save the updated model
        model.save("updated_model.h5")

    print("Press 'p' to take another screenshot or 'q' to quit.")









cv2.namedWindow('Screenshot')
while True:
    screenshot = np.array(ImageGrab.grab())
    cv2.imshow('Screenshot', screenshot)
    key = cv2.waitKey(1)

    if keyboard.is_pressed('q'):
        break
    if keyboard.is_pressed('p'):
        on_press()
        print('pressed')

cv2.destroyAllWindows()
