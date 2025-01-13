# Real-time Video Capture
<div align="center">
  <img src="https://github.com/user-attachments/assets/322a6729-8804-4a06-8210-624a397c62e3" width="350" title="Real-time Video Capture">
  <p align="justify">
    The system captures up to 50 face images in a single session. These images are used to train the Local Binary Patterns Histograms (LBPH) model for better face recognition accuracy. Figure 4.1.1 demonstrates       the webcam capturing faces in real time. Once 50 images are collected (illustrated in Figure 4.1.2), the system prompts the user to input their name, as shown in Figure 4.1.3.
  </p>
</div>

<hr>

# Face Recognition & Attendance Capture
<div align="center">
  <img src="https://github.com/user-attachments/assets/7c91ef55-8c2c-4161-a83d-7945bda6e382" width="350" title="System recognises a single face">
  <p align="justify">
      The face recognition system can recognize a single face when the camera is on. After that, the system will take the attendance for the registered face and name when the user presses the ‘o’ button.  
  </p>
</div>


<div align="center">
  <img src="https://github.com/user-attachments/assets/47451408-b03f-41ea-ac42-58adf79fcaed" width="350" title=" System recognises multiple faces">
  <p align="justify">
      The system can recognize multiple faces that are present on the camera by authenticating the faces with the data that is stored in the pkl file. If the registered faces are verified, the system will take           attendance for those registered.  
  </p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/c90b632c-1495-4104-9ec2-0f0c1fbfdd08" width="350" title="System recognizes faces at different angles">
  <p align="justify">
      The system can recognize the registered faces at different angles. Figure 4.2.3 shows the angle of the face is slightly upward but the system still can recognize the face. Figure 4.2.4 shows the face of           “Hao” is slightly downward but the system can accurately recognize the face.
  </p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/80d6aa28-de1e-4aa3-b955-a33b6b2ee70d" width="350" title="Data in the CSV matched with recognized name and timestamps">
  <p align="justify">
      The system will show the name of the registered face again after taking the attendance as shown in Figure 4.2.5 and the system will generate an audio “Attendance is taken”. The system will also generate an        audio “No new attendance to record”. Figure 4.2.6 shows the attendance record of the register names and the timestamp of attendance.  
  </p>
</div>


<div align="center">
  <img src="https://github.com/user-attachments/assets/2d6c5daa-aa7e-4d60-9182-c45597c8658b" width="350" title="Face recognized in low light environment">
  <p align="justify">
      The system can recognize the registered face in the low light environment accurately. Figure 4.2.7 shows the environment is in dark mode and the system still can detect and recognize the registered face           correctly. 
  </p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/5ec9da08-ec0a-40e7-a146-6d248457085b" width="350" title="The system can recognize registered faces and unregistered faces.">
  <p align="justify">
      The system can recognize the registered faces and unregistered faces accurately without any mistake. Figure 4.2.8 shows that if the faces are unregistered then the system will show the unknown for the             faces. If the faces are registered, then the system will show the name of the registered faces such as “Wang”. 
  </p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/3801c8d1-6fac-4275-83a6-abad5e80ecd8" width="350" title="The system does not log attendance for unregistered faces.">
  <p align="justify">
      The system will not take attendance for unregistered faces. Figure 4.2.9 shows that the face on the camera is unregistered so it will be defined as unknown and the unregistered faces cannot take attendance. 
  </p>
</div>

<hr>

# Attendance Records Viewer

<div align="center">
  <img src="https://github.com/user-attachments/assets/b5b27d11-43eb-421b-adae-ba0225410851" width="350" title="Attendance record interface ">
  <p align="justify">
      Users can view the attendance record through the interface (Streamlit). The attendance list will display who has taken attendance and who has not taken attendance. The figure above shows that 3 people have        taken attendance and one person has not taken attendance.
  </p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/59d75d79-e66c-4b4f-957d-d821f9b6777d" width="350" title="All attended attendance record">
  <p align="justify">
      Based on figure 4.3.3, when nobody is absent, the system will display the “Everyone has taken attendance” message. Figure 4.3.4 shows that all people have taken attendance.
  </p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/3d550733-f4ac-4da7-b65e-2f06ebaa3863" width="350" title="All absent attendance record">
  <p align="justify">
      Based on Figure 4.3.5, when nobody is attending, the system will display a “No one has taken attendance yet” message. Figure 4.3.6 shows the attendance list does not have any record..
  </p>
</div>




