import cv2
import numpy as np
import pickle
import time
import os
import csv
from datetime import datetime
from win32com.client import Dispatch

# Load face names from names.pkl file
with open('data/names.pkl', 'rb') as f:
    known_face_names = pickle.load(f)

# Initialize video capture and setup
video = cv2.VideoCapture(0) 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('data/trainer.yml')  # Load the pre-trained model

def speak(message):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(message)

# Attendance marking function
def markAttendance(name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Attendance marked for {name} at {timestamp}")

# Define the confidence threshold (lower is more sensitive)
threshold = 100  # Lower threshold for confidence

# Background image
if os.path.exists("background.png"):
    imgBackground = cv2.imread("background.png")
else:
    imgBackground = np.zeros((720, 1280, 3), dtype=np.uint8)  # Placeholder background

# List to store names of people who have marked attendance for the current session
marked_names = []  # Track names already marked for attendance
all_attendance = []  # Persistent list to store attendance across frames

while True:
    success, frame = video.read()
    frame = cv2.flip(frame, 1)
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        print(f"Detected face at position: x={x}, y={y}, w={w}, h={h}")
        face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        label, confidence = recognizer.predict(gray_face)
        print(f"Predicted label: {label}, Confidence: {confidence}")

        if 0 <= label < len(known_face_names):  # Ensure label is valid
            name = known_face_names[label]
            if confidence < 90:  # Adjust confidence threshold
                print(f"Recognized: {name}")
                if name not in marked_names:  # Only add if not already recorded
                    all_attendance.append([name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                    marked_names.append(name)  # Add to the marked list
            else:
                print(f"Low confidence ({confidence}). Skipping...")
                name = "Unknown"  # Label as "Unknown" for low confidence
        else:
            print(f"Invalid label: {label}. Skipping...")
            name = "Unknown"  # Label as "Unknown" for low confidence

        # Debug persistent attendance
        print("All Attendance:", all_attendance)

        # Draw rectangle and label around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    k = cv2.waitKey(1)

    # Take attendance when 'o' is pressed
    if k == ord('o'):
        if all_attendance:  # Check persistent attendance list
            speak("Attendance Taken.")
            time.sleep(2)
            attendance_file = f"Attendance/Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
            if os.path.exists(attendance_file):
                with open(attendance_file, "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(all_attendance)
            else:
                with open(attendance_file, "w", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Name", "Timestamp"])
                    writer.writerows(all_attendance)
            all_attendance = []  # Clear after saving            
        else:
            speak("No new attendance to record.")

    # Quit program
    if k == ord('q'):  # 'q' to quit the program
        print("Exiting program...")
        break


# Release video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()


