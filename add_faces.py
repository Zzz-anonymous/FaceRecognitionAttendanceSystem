import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

name = input("Enter your name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3, 5)

    # face detect with red rectangle
    for(x,y,w,h) in faces:
        # crop image of the face and resize it
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50,50))
        
        if len(faces_data) <= 100 and i%10 == 0:
           faces_data.append(resized_img)
        i = i+1
        


        # x = starting point of x-axis
        # y = starting point of y-axis
        # w = ending point of x-axis
        # y = ending point of y-axis        
        cv2.rectangle(frame,(x,y), (x+w, y+h), (50,50,255), 1)

    cv2.imshow('Frame', frame)
    k = cv2.waitKey(1)
    # press 'q' to exit the camera
    if k == ord('q') or len(faces_data)==100:
        break

video.release()
cv2.destroyAllWindows()

# store into pickle file
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# create names.pkl if there is no names.pkl file or store names into names.pkl file
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    # wb = write mode
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    # rb = read mode
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
        # handle duplicate name
        names = names + [name] * 100
        # store the name back to the file
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# store data to a faces_data.pkl file
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis = 0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(names, f)