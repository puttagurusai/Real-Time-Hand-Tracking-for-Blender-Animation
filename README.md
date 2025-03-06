# **🎮 Real-Time Hand Tracking in Blender using Sockets**  

This project enables **real-time hand tracking** in **Blender** using **Mediapipe** and **Socket communication**. It allows users to animate a Blender hand rig dynamically based on live hand movements.  

---

## **📜 Table of Contents**  
- [🚀 Features](#-features)  
- [📦 Requirements](#-requirements)  
- [⚙️ Installation](#-installation)  
- [🖥️ How It Works](#-how-it-works)  
- [📂 Folder Structure](#-folder-structure)  
- [🎬 Demo](#-demo)  
- [🤝 Contributing](#-contributing)  
- [📜 License](#-license)  

---

## **🚀 Features**  
✅ **Real-time** hand tracking with **Mediapipe**  
✅ **Socket communication** for fast data transfer  
✅ **Inverse Kinematics (IK)** for realistic motion  
✅ **Custom hand rig** integration in Blender  
✅ **Supports multiple hand gestures** for animation control  

---

## **📦 Requirements**  

Ensure you have the following installed before running the project:  
##**🖥️ How It Works**

<br /> This system works using a server-client model where hand tracking data is captured and sent to Blender for animation.
<br /> 📌 Step 1: Capturing Hand Data (mediapipe_server.py)
Uses Mediapipe to detect hand landmarks.
Extracts joint positions and normalizes them.
Sends real-time socket data to Blender.
<br /> 📌 Step 2: Creating Joint Points (frames.py)
Converts raw hand tracking data into structured frame points.
Defines key joint locations in Blender’s rig.
<br /> 📌 Step 3: Creating IK Relations (ik_constraints.py)
Uses Inverse Kinematics (IK) Constraints for realistic finger motion.
Ensures fingers bend naturally based on tracking data.
<br /> 📌 Step 4: Updating Hand in Blender (hand_track.py)
Receives hand tracking data from mediapipe_server.py.
Dynamically updates the hand rig bones in Blender.
### 🔧 **Software Requirements**  
- **Python 3.8+**  
- **Blender 3.x**  

### 📦 **Python Dependencies**  
Install the required packages using:  
```bash
pip install -r requirements.txt
