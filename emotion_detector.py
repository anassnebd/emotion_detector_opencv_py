import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore") 
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

# Loading the model 
# ps : you can use other existing models
model = load_model("best_model.h5")

# Load the Haar cascade classifier for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)
# cv2.namedWindow("Facial emotion analysis", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Facial emotion analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    """The line ret, test_img = cap.read() captures a frame from the video stream 
    and returns two values. The first value, ret, is a boolean indicating whether 
    or not the frame was successfully captured. The second value, test_img, is 
    the captured image. If the boolean value ret is False, it means the frame 
    wasn't captured correctly and the loop continues to the next iteration."""
    ret, test_img = cap.read() 
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Branding :) give us a visit
    ##NEED TO REDEPLOY
    cv2.putText(test_img, "Visit us : www.Tedora.info", (10, test_img.shape[0] - 10), font, 0.5, (0, 0, 0), 2)
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    
    # Draw a rectangle around the detected faces
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)
        roi_gray = gray_img[y:y + w, x:x + h]  
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # Make predictions
        predictions = model.predict(img_pixels)

        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'happy', 'fear', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        
        # Display the predicted emotion on the image
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize the image for display
    resized_img = cv2.resize(test_img, (1000, 700))
    # cv2.setWindowProperty("Facial emotion analysis", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Facial emotion analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.imshow('Facial emotion analysis ', resized_img)


    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(10) == ord('q') or cv2.waitKey(30) == 27:  
        break
    
   
# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows

# If you have any question ask me for FREE 
# Instagram : nebd_anass , tedora_design
