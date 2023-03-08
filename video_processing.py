import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained CNN model
model = load_model('asl_model.h5')

# Define a dictionary to map class labels to letters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
               8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
               15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
               22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Open a video capture device
cap = cv2.VideoCapture(1)

# Loop over video frames
while True:
    # Capture the current frame
    ret, frame = cap.read()

    # Preprocess the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0)

    # Make a prediction using the CNN model
    pred = model.predict(img)
    label = np.argmax(pred)

    # Draw the predicted letter on the frame
    cv2.putText(frame, labels_dict[label], (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('ASL Recognition', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close the display window
cap.release()
cv2.destroyAllWindows()