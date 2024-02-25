import cv2
import mediapipe as mp
import time
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from tkinter import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

window = Tk()

cap = cv2.VideoCapture(0)
#change if we want t.py model v which is called model_enhanced.h5 for better accuracy
model = load_model("./model.h5")

def predict_smile(img, model):
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array.astype('float32')
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction[0][0] >= 0.5:
        print("smiling")
        return 1
    else:
        print("nope")
        return -1
#predict_smile("./smile/James_Jones_0001.jpg",model)

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        elapsed_time = time.time() - start_time
        # Format elapsed time as minutes and seconds
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        resized_frame = cv2.resize(frame, (150, 150))

        # Convert to RGB
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        print("seconds")
        print(seconds)
        if seconds >= 10 and predict_smile(rgb_frame,model) == -1:
            window.quit()
            os.system('python /Users/anerie/Desktop/lunahacks/smileScreen.py')
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(48, 25, 52), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        #Attempt to calculate landmarks to test if the person is sitting
        '''
        landmarks = []
        #count landmarks for until 2 mins have past for starting position
        while minutes < 1:
            landmarks.append(results.pose_landmarks)
            #landmarks.append([mp_drawing.draw_landmarks[0], mp_drawing.draw_landmarks[7], mp_drawing.draw_landmarks[8]])
        
        for landmark in landmarks:
            for result in landmark:
                print(result.y)
        print("done")
        break

        noseStart = landmarks([-1], [-3])
        print(noseStart)
        '''
        # Display timer on the image with a rectangle background
        timer_text = f'Timer: {minutes:02d}:{seconds:02d}'
        text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        
        #timer movement
        while minutes > 45:
            window.quit()
            os.system('python /Users/anerie/Desktop/lunahacks/timerScreen.py')

        # Rectangle background for the timer text
        rectangle_color = (48, 25, 52)  # Background color
        cv2.rectangle(image, (5, 5), (text_size[0] + 20, text_size[1] + 20), rectangle_color, -1)  # Draw filled rectangle
        
        # Display timer text on top of the rectangle
        cv2.putText(image, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        

        #WATCH TIMER FOR LONGER THAN 30 MINS + EYE POSITION IN 5+- TRUE/FALSE
        #take points 0, 7, 8 after 3 secs
        
        #TAKE PICS FOR MODEL
        cv2.imshow('ZENMATE', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
