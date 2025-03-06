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
## **✨ Special Features**

### 🔹 **Smoothing Hand Motion**
To prevent jittery or unstable hand tracking, we use a **smoothing algorithm** to blend previous and current frames.

## **✨ Special Features**

### 🔹 **Smoothing Hand Motion**
To prevent jittery or unstable hand tracking, we use a **smoothing algorithm** to blend previous and current frames.

```python
SMOOTHING_FACTOR = 0.6  # Weight for smoothing (0.0 to 1.0, higher is more reactive)


---
## **📦 Requirements**  

Ensure you have the following installed before running the project:  
## **🖥️ How It Works**

This project follows a structured pipeline to **track hand movements** and **apply them to a Blender rig** dynamically. Below is a step-by-step breakdown of how each script contributes to the system:

---

### 📌 **Step 1: Capturing Hand Data** (`mediapipe_server.py`)  
- Uses **Mediapipe** to detect hand landmarks in real-time.  
- Extracts **joint positions** and normalizes them for Blender compatibility.  
- Sends the processed hand-tracking data **through a socket** to Blender.  

---

### 📌 **Step 2: Creating the Hand Armature in Blender** (Manual Setup)  
- Before applying tracking data, we first **create a structured hand rig (armature)** in Blender.  
- The armature consists of **bones for each finger joint**, ensuring proper movement.  
- Each bone is named to match the **tracking data points**, allowing seamless updates later.  

---

### 📌 **Step 3: Creating Joint Points** (`frames.py`)  
- Converts raw **hand-tracking data** into structured frame points.  
- Defines **key joint locations** that will be used in the Blender rig.  
- Each joint point corresponds to a specific **bone in the hand rig**.  

---

### 📌 **Step 4: Creating IK Relations** (`ik_constraints.py`)  
- Uses **Inverse Kinematics (IK) Constraints** to ensure **realistic finger movement**.  
- Ensures that **finger joints bend naturally** instead of moving linearly.  
- Allows the Blender rig to respond dynamically to **live tracking input**.  

---

### 📌 **Step 5: Updating Hand in Blender** (`hand_track.py`)  
- Receives real-time **hand tracking data** from `mediapipe_server.py`.  
- Updates the **Blender armature bones** based on the received data.  
- Continuously refreshes the Blender viewport, so the hand moves dynamically.  

---

This pipeline ensures that **live hand movements** are accurately reflected in Blender **with smooth and natural motion**. 🎯🔥

### 🔧 **Software Requirements**  
- **Python 3.8+**  
- **Blender 3.x**  

### 📦 **Python Dependencies**  
Install the required packages using:  
```bash
pip install -r requirements.txt
