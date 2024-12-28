from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(message):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(message)

# Ensure necessary directories exist
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

# Load pre-trained face data and labels
try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = np.array(pickle.load(f))
except FileNotFoundError:
    print("Data files not found. Ensure 'names.pkl' and 'faces_data.pkl' exist in the 'data' folder.")
    exit()

print('Shape of Faces matrix --> ', FACES.shape)

# Dynamically set n_neighbors based on dataset size
n_neighbors = min(3, len(LABELS))
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(FACES, LABELS)
print(f"Training completed with n_neighbors={n_neighbors}")

# Background image
if os.path.exists("background.png"):
    imgBackground = cv2.imread("background.png")
else:
    imgBackground = np.zeros((720, 1280, 3), dtype=np.uint8)  # Placeholder background

COL_NAMES = ['NAME', 'TIME']

# Function to load attendance records for the day
def load_attendance(date):
    attendance_file = f"Attendance/Attendance_{date}.csv"
    if os.path.exists(attendance_file):
        with open(attendance_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            return {row[0] for row in reader if row}
    return set()

# Video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to read frame from video stream. Exiting...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        detected_attendance = []
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        attendance_file = f"Attendance/Attendance_{date}.csv"

        # Load existing attendance for the day
        existing_attendance = load_attendance(date)

        # for (x, y, w, h) in faces:
        #     crop_img = frame[y:y + h, x:x + w, :]
        #     resized_img = cv2.resize(crop_img, (100, 100)).flatten().reshape(1, -1)

        #     # Validate KNN input dimensions
        #     if resized_img.shape[1] != FACES.shape[1]:
        #         print("Mismatch in input dimensions for KNN. Skipping this detection.")
        #         continue

        #     output = knn.predict(resized_img)[0]
        #     timestamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        #     # Check if the person has already taken attendance
        #     if output not in existing_attendance:
        #         detected_attendance.append([output, timestamp])
        #         existing_attendance.add(output)

        #         # Draw rectangles and text
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #         cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        #         cv2.putText(frame, str(output), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        for (x, y, w, h) in faces:
            # Crop and resize the face
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (100, 100)).flatten().reshape(1, -1)

            # Ensure the input matches the training data type and shape
            resized_img = resized_img.astype(np.float32)  # Convert to match training data type

            # Validate KNN input dimensions
            if resized_img.shape[1] != FACES.shape[1]:
                print(f"Input dimension mismatch: Expected {FACES.shape[1]}, got {resized_img.shape[1]}. Skipping...")
                continue

            # Get the prediction and distances from KNN
            distances, indices = knn.kneighbors(resized_img, n_neighbors=1)
            distance = distances[0][0]  # Distance to the nearest neighbor
            output = knn.predict(resized_img)[0]

            # Assign label based on distance threshold
            threshold = 0.6  # Adjust this value based on your dataset
            label = output if distance < threshold else "Unknown"

            timestamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')

            # Check if the person has already taken attendance
            if label not in existing_attendance:
                detected_attendance.append([label, timestamp])
                existing_attendance.add(label)

                # Draw rectangles and text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, str(label), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)


                # Embed frame into background image
        if imgBackground.shape[0] >= frame.shape[0] and imgBackground.shape[1] >= frame.shape[1]:
            imgBackground[162:162 + frame.shape[0], 55:55 + frame.shape[1]] = frame
        else:
            imgBackground = frame  # Fallback if background is smaller

        cv2.imshow("Frame", imgBackground)
        k = cv2.waitKey(1)

        if k == ord('o'):  # Take attendance for all detected faces
            if detected_attendance:
                speak("Attendance Taken.")
                time.sleep(2)
                if os.path.exists(attendance_file):
                    with open(attendance_file, "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(detected_attendance)
                else:
                    with open(attendance_file, "w", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(COL_NAMES)
                        writer.writerows(detected_attendance)
            else:
                speak("No new attendance to record.")

        if k == ord('q'):  # Quit
            print("Exiting program...")
            break
finally:
    video.release()
    cv2.destroyAllWindows()
