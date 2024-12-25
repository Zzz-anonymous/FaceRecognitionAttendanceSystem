import streamlit as st
import cv2
import pickle
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
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize to a fixed size
        faces_data.append(resized_img.flatten())  # Flatten the image before adding

    # Load or initialize existing face data
    if 'faces_data.pkl' in os.listdir('data/'):
        with open('data/faces_data.pkl', 'rb') as f:
            existing_faces = pickle.load(f)
    else:
        existing_faces = np.empty((0, 7500))  # Initialize with correct dimensions if file doesn't exist

    # Convert new face data to array and ensure consistent dimensions
    faces_data = np.asarray(faces_data)
    if faces_data.size > 0:  # Only append if new faces are detected
        if existing_faces.size == 0:
            existing_faces = faces_data
        else:
            existing_faces = np.vstack([existing_faces, faces_data])  # Use vstack for consistent shape

    # Save updated face data
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(existing_faces, f)

    # Process and save names
    if 'names.pkl' in os.listdir('data/'):
        with open('data/names.pkl', 'rb') as f:
            existing_names = pickle.load(f)
    else:
        existing_names = []

    new_names = [name] * len(faces_data)
    existing_names.extend(new_names)

    with open('data/names.pkl', 'wb') as f:
        pickle.dump(existing_names, f)

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
            if os.path.exists('data/faces_data.pkl') and os.path.exists('data/names.pkl'):
                with open('data/faces_data.pkl', 'rb') as f:
                    faces_data = pickle.load(f)
                with open('data/names.pkl', 'rb') as f:
                    names = pickle.load(f)
            else:
                faces_data = np.empty((0, 7500))  # Initialize with correct dimensions
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
                    resized_img = cv2.resize(crop_img, (50, 50))

                    # Flatten the image and add to dataset
                    resized_img = resized_img.flatten()
                    faces_data = np.append(faces_data, [resized_img], axis=0)
                    names.append(name)

                    # Only add the first face detected
                    st.write(f"Face added for {name}.")
                    break

                # Save updated data
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces_data, f)

                with open('data/names.pkl', 'wb') as f:
                    pickle.dump(names, f)

                st.write(f"Updated dataset saved successfully!")

elif page == "View Captured Faces":
    st.header("Captured Faces and Names")
    
    if os.path.exists('data/faces_data.pkl') and os.path.exists('data/names.pkl'):
        with open('data/faces_data.pkl', 'rb') as f:
            faces_data = pickle.load(f)
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        
        for i in range(len(names)):
            # Convert flattened faces back to image format
            face_image = faces_data[i].reshape(50, 50, 3)  # Reshape the flattened array back to 50x50x3
            face_image = Image.fromarray(face_image.astype(np.uint8))  # Convert to a PIL Image
            
            # Display name and image
            st.image(face_image, caption=names[i], use_container_width=True)
    else:
        st.write("No faces data found.")

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
