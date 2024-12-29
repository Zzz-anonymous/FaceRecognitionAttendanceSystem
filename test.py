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

# Function to speak a message
def speak(message):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(message)

# Attendance file path for the current date
attendance_file = f"Attendance/Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"

# Load previous attendance if the file exists and is not empty
if os.path.exists(attendance_file):
    with open(attendance_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # Skip the header if it exists
        marked_names = [row[0] for row in reader if row]  # Get all previously marked names, handle empty rows
else:
    marked_names = []  # Initialize as empty if the file doesn't exist
# Persistent attendance list
all_attendance = []

# Background image (optional)
if os.path.exists("background.png"):
    imgBackground = cv2.imread("background.png")
else:
    imgBackground = np.zeros((720, 1280, 3), dtype=np.uint8)  # Placeholder background

while True:
    success, frame = video.read()
    frame = cv2.flip(frame, 1)
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        label, confidence = recognizer.predict(gray_face)
        if 0 <= label < len(known_face_names):
            name = known_face_names[label]
            if confidence < 95:  # Adjust confidence threshold
                if name not in marked_names:  # Check if already marked
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    all_attendance.append([name, timestamp])
                    marked_names.append(name)  # Add to persistent marked list
                    print(f"Attendance marked for {name} at {timestamp}")
            else:
                name = "Unknown"
        else:
            name = "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    k = cv2.waitKey(1)

    # Save attendance when 'o' is pressed
    if k == ord('o'):
        if all_attendance:
            speak("Attendance Taken.")
            if os.path.exists(attendance_file):
                with open(attendance_file, "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(all_attendance)
            else:
                os.makedirs(os.path.dirname(attendance_file), exist_ok=True)
                with open(attendance_file, "w", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Name", "Timestamp"])
                    writer.writerows(all_attendance)
            all_attendance = []  # Clear temporary attendance after saving
        else:
            speak("No new attendance to record.")

    if k == ord('q'):  # Exit the program
        print("Exiting program...")
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()