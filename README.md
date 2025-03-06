# **Mediapipe Hand Tracking with Blender using Sockets**

### **Overview**
This project enables real-time hand tracking in **Blender** using **Mediapipe** and **Sockets** for animation. The system captures hand movements via **Mediapipe**, sends the data through sockets, and updates the Blender animation in real time.

---

## **Table of Contents**
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

---

## **Features**
✅ Real-time hand tracking with **Mediapipe**  
✅ Sends hand tracking data via **Sockets**  
✅ Controls Blender animations dynamically  
✅ Supports multiple hand gestures  

---

## **Requirements**
Make sure you have the following installed before running the project:

- **Python 3.8+**  
- **Blender 3.x**  
- **Mediapipe** (`pip install mediapipe`)  
- **OpenCV** (`pip install opencv-python`)  
- **Socket Programming Support** (built-in with Python)  

---

## **🖥️ How It Works**

📌 Step 1: Capturing Hand Data (mediapipe_server.py)
Tracks hand movements in real-time.
Extracts joint positions and sends data via a socket.
📌 Step 2: Creating Joint Points (frames.py)
Converts raw hand-tracking data into structured frame points.
Defines key joint locations for the hand rig in Blender.
📌 Step 3: Creating IK Relations (ik_constraints.py)
Uses Inverse Kinematics (IK) Constraints to simulate natural hand motion.
Ensures fingers bend correctly based on tracking data.
📌 Step 4: Updating Hand in Blender (hand_track.py)
Receives hand tracking data from mediapipe_server.py.
Updates the hand rig bones in Blender dynamically.

1. **Clone the repository**
   ```sh
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
