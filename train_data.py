import streamlit as st
import cv2
import pickle
import numpy as np
import os
from PIL import Image

# Function to process and store face data
# def process_face_data(face_image, name):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
#     faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     faces = faceDetect.detectMultiScale(gray, 1.3, 5)

#     if len(faces) == 0:  # No faces detected
#         st.error("No face detected in the image. Please try again!")
#         return  # Exit the function

#     faces_data = []
#     for (x, y, w, h) in faces:
#         # Draw a rectangle around each face
#         cv2.rectangle(face_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle with 2px thickness

#         # Crop face image
#         crop_img = face_image[y:y + h, x:x + w]
#         resized_img = cv2.resize(crop_img, (100, 100))  # Resize to a fixed size
#         faces_data.append(resized_img.flatten())  # Flatten the image before adding

#     # Load or initialize existing face data
#     if os.path.exists('data/faces_data.pkl'):
#         with open('data/faces_data.pkl', 'rb') as f:
#             existing_faces = pickle.load(f)
#     else:
#         existing_faces = []  # Initialize if file doesn't exist

#     existing_faces.extend(faces_data)

#     # Save updated face data
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(existing_faces, f)

#     # Process and save names
#     if os.path.exists('data/names.pkl'):
#         with open('data/names.pkl', 'rb') as f:
#             existing_names = pickle.load(f)
#     else:
#         existing_names = []

#     new_names = [name] * len(faces_data)
#     existing_names.extend(new_names)

#     with open('data/names.pkl', 'wb') as f:
#         pickle.dump(existing_names, f)

#     st.success("Face captured and data saved successfully!")
#     st.image(face_image, caption="Image with Face Highlighted", use_container_width=True)

# def process_face_data(faces_data, name):
#     # Process and save faces as in the second code block
#     faces_data_flattened = [img.flatten() for img in faces_data]  # Flatten images before saving

#     # Load or initialize existing face data
#     if os.path.exists('data/faces_data.pkl'):
#         with open('data/faces_data.pkl', 'rb') as f:
#             existing_faces = pickle.load(f)
#     else:
#         existing_faces = []

#     existing_faces.extend(faces_data_flattened)

#     # Save updated face data
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(existing_faces, f)

#     # Process and save names
#     if os.path.exists('data/names.pkl'):
#         with open('data/names.pkl', 'rb') as f:
#             existing_names = pickle.load(f)
#     else:
#         existing_names = []

#     new_names = [name] * len(faces_data)
#     existing_names.extend(new_names)

#     with open('data/names.pkl', 'wb') as f:
#         pickle.dump(existing_names, f)

#     st.success("Face captured and data saved successfully!")

# # Initialize video capture
# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# faces_data = []
# i = 0

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y + h, x:x + w]
#         resized_img = cv2.resize(crop_img, (50, 50))

#         if len(faces_data) <= 100 and i % 10 == 0:
#             faces_data.append(resized_img)

#         i += 1
#         cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)

#     # If user presses 'q' or we reach 100 face samples, process and save the data
#     if k == ord('q') or len(faces_data) == 100:
#         name = "Person_Name"  # Get name dynamically, for example from a user input
#         process_face_data(faces_data, name)
#         break

# video.release()
# cv2.destroyAllWindows()

# # Streamlit app layout
# st.title("Face Recognition Data Collection")

# # Sidebar for navigation
# page = st.sidebar.selectbox("Choose a page", ["Upload Image", "View Captured Faces", "Real-time Face Capture"])

# if page == "Upload Image":
#     # Upload image file through Streamlit file uploader
#     image_file = st.file_uploader("Upload a .jpg image", type=["jpg", "jpeg"])

#     # Input name associated with the face
#     name = st.text_input("Enter the name for the face")

#     # Button to process image
#     if st.button("Process Image"):
#         if image_file is None:
#             st.error("Please upload an image.")
#         elif not name:
#             st.error("Please enter your name.")
#         else:
#             # Open and process the image
#             image = Image.open(image_file)
#             image = np.array(image)
#             process_face_data(image, name)

# elif page == "View Captured Faces":
#     st.header("Captured Faces and Names")

#     # Check if the data files exist
#     if os.path.exists('data/faces_data.pkl') and os.path.exists('data/names.pkl'):
#         with open('data/faces_data.pkl', 'rb') as f:
#             faces_data = pickle.load(f)
#         with open('data/names.pkl', 'rb') as f:
#             names = pickle.load(f)

#         # Validate data consistency
#         if len(faces_data) != len(names):
#             st.error("Mismatch between the number of faces and names. Data might be corrupted.")

#             # Truncate data to match the shorter length
#             min_length = min(len(faces_data), len(names))
#             faces_data = faces_data[:min_length]
#             names = names[:min_length]
#             st.warning("Mismatched data has been truncated to the smallest valid size.")

#         # Display the captured faces and names
#         for i in range(len(names)):
#             try:
#                 # Reshape flattened face to image format
#                 face_image = np.array(faces_data[i]).reshape(100, 100, 3)  # Assuming RGB channels
#                 face_image = Image.fromarray(face_image.astype(np.uint8))  # Convert to PIL Image

#                 # Display name and image
#                 st.image(face_image, caption=names[i], use_container_width=True)
#             except Exception as e:
#                 st.error(f"Error processing face at index {i}: {e}")
#     else:
#         st.write("No faces data found. Please ensure 'faces_data.pkl' and 'names.pkl' exist in the 'data' folder.")

# elif page == "Real-time Face Capture":
#     st.header("Real-time Face Capture")

#     # Name input for webcam capture
#     webcam_name = st.text_input("Enter your name for webcam capture:")

#     # Capture image from webcam
#     image = st.camera_input("Capture your face:")

#     if image is not None:
#         if not webcam_name:
#             st.error("Please enter your name for the webcam capture.")
#         else:
#             # Convert to OpenCV format
#             img = Image.open(image)
#             img_array = np.array(img)

#             # Check if image is in the correct shape
#             if img_array.ndim == 2:  # Grayscale image
#                 img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels (BGR)

#             # Ensure the image is in uint8 format
#             img_array = img_array.astype(np.uint8)

#             # Process the captured image to detect and store faces
#             process_face_data(img_array, webcam_name)
#             st.image(img_array, caption="Captured Image", use_container_width=True)

import streamlit as st
import cv2
import pickle
import numpy as np
import os
from PIL import Image

# Function to process and store face data
def process_face_data(faces_data, name):
    try:
        # Flatten images before saving
        faces_data_flattened = [np.array(img).flatten() for img in faces_data]

        # Ensure 'data' folder exists
        if not os.path.exists('data'):
            os.makedirs('data')

        # Load or initialize existing face data
        if os.path.exists('data/faces_data.pkl'):
            with open('data/faces_data.pkl', 'rb') as f:
                try:
                    existing_faces = pickle.load(f)
                except EOFError:  # Handle empty file
                    existing_faces = []
        else:
            existing_faces = []

        existing_faces.extend(faces_data_flattened)

        # Save updated face data
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(existing_faces, f)

        # Debugging
        print(f"Faces saved: {len(existing_faces)}")

        # Process and save names
        if os.path.exists('data/names.pkl'):
            with open('data/names.pkl', 'rb') as f:
                try:
                    existing_names = pickle.load(f)
                except EOFError:
                    existing_names = []
        else:
            existing_names = []

        new_names = [name] * len(faces_data)
        existing_names.extend(new_names)

        with open('data/names.pkl', 'wb') as f:
            pickle.dump(existing_names, f)

        st.success("Face captured and data saved successfully!")

    except Exception as e:
        st.error(f"Error processing face data: {e}")


# Initialize video capture and setup
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces_data = []
i = 0

# Initialize session state variable to track face capture process
if 'captured' not in st.session_state:
    st.session_state['captured'] = False

# Page logic
st.title("Face Recognition Data Collection")
page = st.sidebar.selectbox("Choose a page", ["Upload Image", "View Captured Faces", "Real-time Face Capture"])

# Define a threshold value for face recognition
threshold = 0.6  # Adjust this based on your training and testing

if page == "Real-time Face Capture":
    st.header("Real-time Face Capture")

    # Name input for webcam capture
    webcam_name = st.text_input("Enter your name for webcam capture:")

    if webcam_name and not st.session_state['captured']:
        st.session_state['captured'] = True  # Mark as captured

        # Capture image from webcam
        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue  # Skip invalid regions

                crop_img = frame[y:y + h, x:x + w]
                resized_img = cv2.resize(crop_img, (100, 100))  # Resize to 100x100
                resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                

                if len(faces_data) <= 50 and i % 10 == 0:
                   faces_data.append(resized_img_rgb)  # Append the RGB image
                i += 1
                
                cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)


            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1)

            # If user presses 'q' or we reach 50 face samples, process and save the data
            if k == ord('q') or len(faces_data) == 50:
                process_face_data(faces_data, webcam_name)
                break

        video.release()
        cv2.destroyAllWindows()

    elif not webcam_name:
        st.error("Please enter your name for the webcam capture.")

elif page == "Upload Image":
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
            # Open and process the image
            image = Image.open(image_file)
            image = np.array(image)
            resized_img = cv2.resize(image, (100, 100))  # Resize to 100x100 pixels
            faces_data = [resized_img]  # Store resized image
            process_face_data(faces_data, name)

elif page == "View Captured Faces":
    st.header("Captured Faces and Names")

    # Ensure the 'data' folder exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Create the pkl files if they don't exist
    if not os.path.exists('data/faces_data.pkl'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump([], f)  # Initialize with an empty list

    if not os.path.exists('data/names.pkl'):
        with open('data/names.pkl', 'wb') as f:
            pickle.dump([], f)  # Initialize with an empty list

    # Load faces and names data
    with open('data/faces_data.pkl', 'rb') as f:
        faces_data = pickle.load(f)
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)

    # Validate data consistency
    if len(faces_data) != len(names):
        st.error("Mismatch between the number of faces and names. Data might be corrupted.")
        min_length = min(len(faces_data), len(names))
        faces_data = faces_data[:min_length]
        names = names[:min_length]
        st.warning("Mismatched data has been truncated to the smallest valid size.")

    # Display the captured faces and names
    for i in range(len(names)):
        try:
            face_image = np.array(faces_data[i])  # Ensure correct format
            face_image = face_image.reshape(100, 100, 3)  # Ensure correct shape
            face_image = Image.fromarray(face_image.astype(np.uint8))  # Convert to PIL Image
            st.image(face_image, caption=names[i], use_container_width=True)
        except Exception as e:
            st.error(f"Error processing face at index {i}: {e}")

