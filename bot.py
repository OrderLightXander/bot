import cv2
import numpy as np
import keyboard
import os
import pyautogui
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
model.fit(X_train, y_train, epochs=30, batch_size=32)

# Additional function to save the image with the correct resource label
def save_corrected_image(screenshot, correct_resource):
    img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the amount of pixels that were cropped from the top and left of the image
    h, w = screenshot.shape[:2]
    resize_h, resize_w = img.shape[:2]
    top_crop = (h - resize_h) // 2
    left_crop = (w - resize_w) // 2

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Add the amount of pixels that were cropped to the x and y coordinates
        x += left_crop
        y += top_crop
        cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Change thickness to 2

    if not os.path.exists(f'./{correct_resource}'):
        os.makedirs(f'./{correct_resource}')
    filename = f"{correct_resource}_{len(os.listdir(f'./{correct_resource}')) + 1}.png"
    cv2.imwrite(f"./{correct_resource}/{filename}", screenshot)
    print(f"Corrected image saved as {filename}")
    return filename

def predict_resource(screenshot):
    # Calculate the dimensions of the screenshot and the square region of interest
    h, w, _ = screenshot.shape
    h_resize_factor = h / 128
    w_resize_factor = w / 128
    square_size = min(h, w)
    y_offset = (h - square_size) // 2
    x_offset = (w - square_size) // 2
    square = screenshot[y_offset:y_offset+square_size, x_offset:x_offset+square_size, :]

    # Resize the square region to 128x128 and convert to grayscale
    img = cv2.cvtColor(square, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=-1)

    # Normalize the pixel values
    img = img / 255.0

    # Predict the resource in the image
    prediction = model.predict(np.array([img]))
    resource = resources[np.argmax(prediction)]
    print(f'Resource: {resource}')

    # Draw the predicted rectangle on the screenshot
    gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    largest_contour_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > largest_contour_area:
            largest_contour_area = area
            largest_contour = c

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Add the amount of pixels that were cropped to the x and y coordinates
        x += int(x_offset / w_resize_factor)
        y += int(y_offset / h_resize_factor)
        # Adjust the coordinates based on the resize factor
        x = int(x / w_resize_factor)
        y = int(y / h_resize_factor)
        w = int(w / w_resize_factor)
        h = int(h / h_resize_factor)
        cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw the predicted resource label on the screenshot
    cv2.putText(screenshot, resource, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return screenshot

# Additional function to draw rectangles for predicted resources in real-time
def draw_rectangles(screenshot):
    # Get the dimensions of the window
    window_rect = cv2.getWindowImageRect('Screenshot')
    h_window = window_rect[3]
    w_window = window_rect[2]

    # Calculate the dimensions of the screenshot and the square region of interest
    h, w, _ = screenshot.shape
    h_resize_factor = h / h_window
    w_resize_factor = w / w_window
    square_size = min(h, w)
    y_offset = (h - square_size) // 2
    x_offset = (w - square_size) // 2
    square = screenshot[y_offset:y_offset+square_size, x_offset:x_offset+square_size, :]

    # Resize the square region to 128x128 and convert to grayscale
    img = cv2.cvtColor(square, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=-1)

    # Normalize the pixel values
    img = img / 255.0

    # Predict the resource in the image
    prediction = model.predict(np.array([img]))
    resource = resources[np.argmax(prediction)]
    print(f'Resource: {resource}')

    # Find the contours of the predicted resource in the original screenshot
    gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    largest_contour_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > largest_contour_area:
            largest_contour_area = area
            largest_contour = c

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Add the amount of pixels that were cropped to the x and y coordinates
        x += int(x_offset / w_resize_factor)
        y += int(y_offset / h_resize_factor)
        # Adjust the coordinates based on the resize factor
        x = int(x / w_resize_factor)
        y = int(y / h_resize_factor)
        w = int(w / w_resize_factor)
        h = int(h / h_resize_factor)
        cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Click left mouse button on the center of the largest predicted resource
        center_x = x + w // 2
        center_y = y + h // 2
        #pyautogui.click(center_x, center_y, button='left')

    return screenshot


# Real-time detection of predicted resources with live drawing of rectangles
cv2.namedWindow('Screenshot')
while True:
    screenshot = np.array(ImageGrab.grab())
    processed_screenshot = draw_rectangles(screenshot) # Process the screenshot
    cv2.imshow('Screenshot', processed_screenshot) # Update the live window with the processed screenshot
    key = cv2.waitKey(1)
    if keyboard.is_pressed('q'):
        break
    if keyboard.is_pressed('p'):
        print('Taking screenshot...')
        screenshot = np.array(ImageGrab.grab())
        screenshot = screenshot[200:-200, 200:-200, :]
        square_size = 200
        square = np.zeros((square_size, square_size, 3), dtype=np.uint8)
        h, w, _ = screenshot.shape
        y_offset = (h - square_size) // 2
        x_offset = (w - square_size) // 2
        screenshot[y_offset:y_offset+square_size, x_offset:x_offset+square_size, :] = square
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

cv2.destroyAllWindows()


