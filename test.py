import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime
from win32com.client import Dispatch

# Speak function using Windows SAPI
def speak(message):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(message)

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load saved face data and labels
if os.path.exists('data/names.pkl') and os.path.exists('data/faces_data.pkl'):
    with open('data/names.pkl', 'rb') as name_file:
        LABELS = pickle.load(name_file)
    with open('data/faces_data.pkl', 'rb') as faces_file:
        FACES = pickle.load(faces_file)
else:
    print("Face data files not found. Ensure 'names.pkl' and 'faces_data.pkl' are in the 'data' folder.")
    exit()

print('Shape of Faces matrix --> ', FACES.shape)

# Train KNN classifier
knn = cv2.KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Attendance file columns
COL_NAMES = ['NAME', 'TIME']

# Main loop for real-time face recognition
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop and preprocess face
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        
        # Predict using KNN
        output = knn.predict(resized_img)
        
        # Get timestamp
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        
        # Prepare attendance file
        attendance_dir = "Attendance"
        if not os.path.exists(attendance_dir):
            os.makedirs(attendance_dir)
        attendance_file = os.path.join(attendance_dir, f"Attendance_{date}.csv")
        attendance = [str(output[0]), timestamp]
        
        # Display recognized face with name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 255, 0), -1)
        cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)

    if key == ord('o'):  # 'o' to log attendance
        speak("Attendance Taken.")
        if os.path.exists(attendance_file):
            with open(attendance_file, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open(attendance_file, "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

    if key == ord('q'):  # 'q' to quit
        break

# Release resources
video.release()
cv2.destroyAllWindows()
