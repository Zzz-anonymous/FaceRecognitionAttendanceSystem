import streamlit as st
import cv2
import json
import numpy as np
import os
from PIL import Image

# Function to process and store face data
def process_face_data(face_image, name):
    # Convert the image to grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:  # No faces detected
        st.error("No face detected in the image. Please try again!")
        return  # Exit the function

    faces_data = []
    names = []

    for (x, y, w, h) in faces:
        # Draw a rectangle around each face
        cv2.rectangle(face_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle with 2px thickness

        # Crop face image
        crop_img = face_image[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (100, 100))  # Resize to a fixed size
        faces_data.append(resized_img.flatten().tolist())  # Flatten the image before adding

    # Load or initialize existing face data
    if os.path.exists('data/faces_data.json'):
        with open('data/faces_data.json', 'r') as f:
            existing_faces = json.load(f)
    else:
        existing_faces = []  # Initialize if file doesn't exist

    existing_faces.extend(faces_data)

    # Save updated face data
    with open('data/faces_data.json', 'w') as f:
        json.dump(existing_faces, f)

    # Process and save names
    if os.path.exists('data/names.json'):
        with open('data/names.json', 'r') as f:
            existing_names = json.load(f)
    else:
        existing_names = []

    new_names = [name] * len(faces_data)
    existing_names.extend(new_names)

    with open('data/names.json', 'w') as f:
        json.dump(existing_names, f)

    st.success("Face captured and data saved successfully!")
    st.image(face_image, caption="Image with Face Highlighted", use_container_width=True)


# Streamlit app layout
st.title("Face Recognition Data Collection")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Upload Image", "View Captured Faces", "Real-time Face Capture"])

if page == "Upload Image":
    # Upload image file through Streamlit file uploader
    image_file = st.file_uploader("Upload a .jpg image", type=["jpg", "jpeg"])

    # Input name associated with the face
    name = st.text_input("Enter the name for the face")

    # Button to process image
    if st.button("Process Image"):
        if image_file is None:
            st.error("Please upload an image.")
        elif not name:
            st.error("Please enter your name.")
        else:
            # Load existing data
            if os.path.exists('data/faces_data.json') and os.path.exists('data/names.json'):
                with open('data/faces_data.json', 'r') as f:
                    faces_data = json.load(f)
                with open('data/names.json', 'r') as f:
                    names = json.load(f)
            else:
                faces_data = []  # Initialize if files do not exist
                names = []

            # Open and process the image
            image = Image.open(image_file)
            image = np.array(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Load Haar Cascade for face detection
            faceDetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

            # Detect faces in the image
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                st.write("No face detected in the image.")
            else:
                for (x, y, w, h) in faces:
                    # Crop and resize the face
                    crop_img = image[y:y + h, x:x + w, :]
                    resized_img = cv2.resize(crop_img, (100, 100))

                    # Flatten the image and add to dataset
                    resized_img = resized_img.flatten().tolist()
                    faces_data.append(resized_img)
                    names.append(name)

                    # Only add the first face detected
                    st.write(f"Face added for {name}.")
                    break

                # Save updated data
                with open('data/faces_data.json', 'w') as f:
                    json.dump(faces_data, f)

                with open('data/names.json', 'w') as f:
                    json.dump(names, f)

                st.write(f"Updated dataset saved successfully!")

elif page == "View Captured Faces":
    st.header("Captured Faces and Names")

    # Check if the data files exist
    if os.path.exists('data/faces_data.json') and os.path.exists('data/names.json'):
        with open('data/faces_data.json', 'r') as f:
            faces_data = json.load(f)
        with open('data/names.json', 'r') as f:
            names = json.load(f)

        # Validate data consistency
        if len(faces_data) != len(names):
            st.error("Mismatch between the number of faces and names. Data might be corrupted.")

            # Truncate data to match the shorter length
            min_length = min(len(faces_data), len(names))
            faces_data = faces_data[:min_length]
            names = names[:min_length]
            st.warning("Mismatched data has been truncated to the smallest valid size.")

        # Display the captured faces and names
        for i in range(len(names)):
            try:
                # Reshape flattened face to image format
                face_image = np.array(faces_data[i]).reshape(100, 100, 3)  # Assuming RGB channels
                face_image = Image.fromarray(face_image.astype(np.uint8))  # Convert to PIL Image

                # Display name and image
                st.image(face_image, caption=names[i], use_container_width=True)
            except Exception as e:
                st.error(f"Error processing face at index {i}: {e}")
    else:
        st.write("No faces data found. Please ensure 'faces_data.json' and 'names.json' exist in the 'data' folder.")

elif page == "Real-time Face Capture":
    st.header("Real-time Face Capture")

    # Name input for webcam capture
    webcam_name = st.text_input("Enter your name for webcam capture:")

    # Capture image from webcam
    image = st.camera_input("Capture your face:")

    if image is not None:
        if not webcam_name:
            st.error("Please enter your name for the webcam capture.")
        else:
            # Convert to OpenCV format
            img = Image.open(image)
            img_array = np.array(img)

            # Check if image is in the correct shape
            if img_array.ndim == 2:  # Grayscale image
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels (BGR)
            
            # Ensure the image is in uint8 format
            img_array = img_array.astype(np.uint8)

            # Process the captured image to detect and store faces
            process_face_data(img_array, webcam_name)
            st.image(img_array, caption="Captured Image", use_container_width=True)
