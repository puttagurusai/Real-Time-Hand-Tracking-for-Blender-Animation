import mediapipe as mp
import cv2
import socket
import json

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Bone-to-landmark mapping
bone_to_landmarks = {
    "thumb_cmc_to_thumb_mcp": ("THUMB_CMC", "THUMB_MCP"),
    "thumb_mcp_to_thumb_ip": ("THUMB_MCP", "THUMB_IP"),
    "thumb_ip_to_thumb_tip": ("THUMB_IP", "THUMB_TIP"),
    "index_mcp_to_index_pip": ("INDEX_FINGER_MCP", "INDEX_FINGER_PIP"),
    "index_pip_to_index_dip": ("INDEX_FINGER_PIP", "INDEX_FINGER_DIP"),
    "index_dip_to_index_tip": ("INDEX_FINGER_DIP", "INDEX_FINGER_TIP"),
    "middle_mcp_to_middle_pip": ("MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP"),
    "middle_pip_to_middle_dip": ("MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP"),
    "middle_dip_to_middle_tip": ("MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP"),
    "ring_mcp_to_ring_pip": ("RING_FINGER_MCP", "RING_FINGER_PIP"),
    "ring_pip_to_ring_dip": ("RING_FINGER_PIP", "RING_FINGER_DIP"),
    "ring_dip_to_ring_tip": ("RING_FINGER_DIP", "RING_FINGER_TIP"),
    "pinky_mcp_to_pinky_pip": ("PINKY_MCP", "PINKY_PIP"),
    "pinky_pip_to_pinky_dip": ("PINKY_PIP", "PINKY_DIP"),
    "pinky_dip_to_pinky_tip": ("PINKY_DIP", "PINKY_TIP"),
    "wrist_to_thumb_cmc": ("WRIST", "THUMB_CMC"),
    "wrist_to_index_mcp": ("WRIST", "INDEX_FINGER_MCP"),
    "wrist_to_middle_mcp": ("WRIST", "MIDDLE_FINGER_MCP"),
    "wrist_to_ring_mcp": ("WRIST", "RING_FINGER_MCP"),
    "wrist_to_pinky_mcp": ("WRIST", "PINKY_MCP"),
}

# Socket setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
blender_address = ('127.0.0.1', 5004)

# Webcam setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check connection to Blender before starting
def check_connection():
    try:
        # Send a test message to Blender to see if the connection works
        test_message = {"status": "ready"}
        sock.sendto(json.dumps(test_message).encode(), blender_address)
        return True
    except Exception as e:
        print(f"Connection error: {e}")
        return False

# Wait for Blender to be ready
print("Waiting for connection to Blender...")
while not check_connection():
    print("Waiting for Blender to start...")
    cv2.waitKey(1000)  # Wait 1 second before retrying

print("Connected to Blender. Sending bone data... Press ESC to exit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame for visualization
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Prepare the data
            points_data = {}
            for point_name in mp_hands.HandLandmark:
                landmark = hand_landmarks.landmark[point_name]
                points_data[point_name.name] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                }
            
            # Send the data as JSON to Blender
            sock.sendto(json.dumps(points_data).encode(), blender_address)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key pressed
        # Send stop signal to Blender
        stop_message = "stop"
        sock.sendto(stop_message.encode(), blender_address)
        print("Sending stop signal...")
        break

cap.release()
cv2.destroyAllWindows()
sock.close()