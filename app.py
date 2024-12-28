import streamlit as st
import os
import csv
import pickle

# Function to read attendance records from a specific file
def read_attendance(file_path):
    attended_names = set()
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip the header row if present
            for row in reader:
                if row:
                    attended_names.add(row[0])
    except Exception as e:
        st.error(f"Error reading file {file_path}: {e}")
    return attended_names

# Function to read all names from names.pkl
def read_all_names(names_file):
    try:
        with open(names_file, 'rb') as f:
            return set(pickle.load(f))
    except Exception as e:
        st.error(f"Error reading file {names_file}: {e}")
        return set()

# Streamlit app layout
st.title("Attendance Records Viewer")

# Check if the Attendance folder exists
attendance_folder = "Attendance"
names_file = "data/names.pkl"
if not os.path.exists(attendance_folder):
    st.warning("No attendance records found. Please ensure the 'Attendance' folder exists and contains attendance files.")
else:
    # List all attendance files
    attendance_files = [f for f in os.listdir(attendance_folder) if f.endswith('.csv')]

    if len(attendance_files) == 0:
        st.write("No attendance records available.")
    else:
        # Dropdown to select an attendance file
        selected_file = st.selectbox("Select an attendance file to view:", attendance_files)

        if selected_file:
            file_path = os.path.join(attendance_folder, selected_file)
            attended_names = read_attendance(file_path)

            # Read all names from names.pkl
            all_names = read_all_names(names_file)

            if all_names:
                # Separate names into attended and not attended
                attended = sorted(attended_names)
                not_attended = sorted(all_names - attended_names)

                # Display attendance records
                st.write(f"### Attendance Records: {selected_file}")

                st.write("#### Names Who Have Taken Attendance:")
                if attended:
                    st.table(attended)
                else:
                    st.write("No one has taken attendance yet.")

                st.write("#### Names Who Have Not Taken Attendance:")
                if not_attended:
                    st.table(not_attended)
                else:
                    st.write("Everyone has taken attendance.")
            else:
                st.write("No names found in the system.")
