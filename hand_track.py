#import bpy
#import socket
#import json
#from mathutils import Vector

## Socket setup
#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sock.bind(('127.0.0.1', 5004))  # Adjust this to match your sender's address and port
#sock.setblocking(False)  # Non-blocking socket

## IK target naming convention (matches joint names in incoming data)
#ik_targets = {
#    "THUMB_CMC": "THUMB_CMC",
#    "THUMB_MCP": "THUMB_MCP",
#    "THUMB_IP": "THUMB_IP",
#    "THUMB_TIP": "THUMB_TIP",
#    "INDEX_FINGER_MCP": "INDEX_FINGER_MCP",
#    "INDEX_FINGER_PIP": "INDEX_FINGER_PIP",
#    "INDEX_FINGER_DIP": "INDEX_FINGER_DIP",
#    "INDEX_FINGER_TIP": "INDEX_FINGER_TIP",
#    "MIDDLE_FINGER_MCP": "MIDDLE_FINGER_MCP",
#    "MIDDLE_FINGER_PIP": "MIDDLE_FINGER_PIP",
#    "MIDDLE_FINGER_DIP": "MIDDLE_FINGER_DIP",
#    "MIDDLE_FINGER_TIP": "MIDDLE_FINGER_TIP",
#    "RING_FINGER_MCP": "RING_FINGER_MCP",
#    "RING_FINGER_PIP": "RING_FINGER_PIP",
#    "RING_FINGER_DIP": "RING_FINGER_DIP",
#    "RING_FINGER_TIP": "RING_FINGER_TIP",
#    "PINKY_MCP": "PINKY_MCP",
#    "PINKY_PIP": "PINKY_PIP",
#    "PINKY_DIP": "PINKY_DIP",
#    "PINKY_TIP": "PINKY_TIP",
#}

#bone_to_landmarks = {
#    "thumb_cmc_to_thumb_mcp": ("THUMB_CMC", "THUMB_MCP"),
#    "thumb_mcp_to_thumb_ip": ("THUMB_MCP", "THUMB_IP"),
#    "thumb_ip_to_thumb_tip": ("THUMB_IP", "THUMB_TIP"),
#    "index_mcp_to_index_pip": ("INDEX_FINGER_MCP", "INDEX_FINGER_PIP"),
#    "index_pip_to_index_dip": ("INDEX_FINGER_PIP", "INDEX_FINGER_DIP"),
#    "index_dip_to_index_tip": ("INDEX_FINGER_DIP", "INDEX_FINGER_TIP"),
#    "middle_mcp_to_middle_pip": ("MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP"),
#    "middle_pip_to_middle_dip": ("MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP"),
#    "middle_dip_to_middle_tip": ("MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP"),
#    "ring_mcp_to_ring_pip": ("RING_FINGER_MCP", "RING_FINGER_PIP"),
#    "ring_pip_to_ring_dip": ("RING_FINGER_PIP", "RING_FINGER_DIP"),
#    "ring_dip_to_ring_tip": ("RING_FINGER_DIP", "RING_FINGER_TIP"),
#    "pinky_mcp_to_pinky_pip": ("PINKY_MCP", "PINKY_PIP"),
#    "pinky_pip_to_pinky_dip": ("PINKY_PIP", "PINKY_DIP"),
#    "pinky_dip_to_pinky_tip": ("PINKY_DIP", "PINKY_TIP"),
#    "wrist_to_thumb_cmc": ("WRIST", "THUMB_CMC"),
#    "wrist_to_index_mcp": ("WRIST", "INDEX_FINGER_MCP"),
#    "wrist_to_middle_mcp": ("WRIST", "MIDDLE_FINGER_MCP"),
#    "wrist_to_ring_mcp": ("WRIST", "RING_FINGER_MCP"),
#    "wrist_to_pinky_mcp": ("WRIST", "PINKY_MCP"),
#}

## Reference length landmarks
#REFERENCE_START = "WRIST"
#REFERENCE_END = "MIDDLE_FINGER_MCP"
#REFERENCE_DISTANCE = 0.386  # Reference distance in meters (adjust based on your system)

## Dictionary to store initial bone lengths
#initial_bone_lengths = {}

## Store the initial lengths of bones at script start
#def store_initial_bone_lengths():
#    armature = bpy.data.objects.get("Armature")  # Replace 'Armature' with your armature's name
#    if not armature or not armature.pose:
#        print("Armature not found or invalid. Ensure the correct armature name.")
#        stop_script()
#        return

#    bpy.context.view_layer.objects.active = armature
#    bpy.ops.object.mode_set(mode='POSE')

#    for bone_name in bone_to_landmarks.keys():
#        bone = armature.pose.bones.get(bone_name)
#        if bone:
#            initial_bone_lengths[bone_name] = bone.length
#        else:
#            print(f"Bone {bone_name} not found in armature. Skipping...")
#            stop_script()

## Timer function to update IK targets
#def receive_data():
#    try:
#        data, _ = sock.recvfrom(4096)
#        message = data.decode('utf-8')

#        # Stop script if "stop" signal received
#        if message.strip().lower() == "stop":
#            print("Received stop signal. Exiting...")
#            stop_script()
#            return None

#        # Parse JSON data
#        coordinates = json.loads(message)

#        # Ensure reference landmarks exist
#        if REFERENCE_START not in coordinates or REFERENCE_END not in coordinates:
#            print(f"Reference landmarks {REFERENCE_START} or {REFERENCE_END} missing. Skipping frame.")
#            stop_script()
#            return

#        # Calculate scale normalization factor
#        start_pos = Vector((
#            coordinates[REFERENCE_START]["x"],
#            coordinates[REFERENCE_START]["z"],
#            -coordinates[REFERENCE_START]["y"]
#        ))
#        end_pos = Vector((
#            coordinates[REFERENCE_END]["x"],
#            coordinates[REFERENCE_END]["z"],
#            -coordinates[REFERENCE_END]["y"]
#        ))
#        current_distance = (end_pos - start_pos).length
#        scale_factor = REFERENCE_DISTANCE / current_distance if current_distance > 0 else 1.0

#        # Update IK targets
#        for landmark, target_name in ik_targets.items():
#            if landmark not in coordinates:
#                print(f"Landmark {landmark} missing in data. Skipping...")
#                stop_script()
#                continue

#            target_empty = bpy.data.objects.get(target_name)
#            if not target_empty:
#                print(f"Target empty {target_name} not found in the scene. Skipping...")
#                stop_script()
#                continue

#            # Normalize position and update the target
#            raw_pos = Vector((
#                coordinates[landmark]["x"],
#                coordinates[landmark]["z"],
#                -coordinates[landmark]["y"]
#            ))
#            normalized_pos = (raw_pos - start_pos) * scale_factor
#            target_empty.location = normalized_pos

#            # Adjust bone length to match initial length
#            bone_length = initial_bone_lengths.get(target_name, None)
#            if bone_length:
#                armature = bpy.data.objects.get("Armature")  # Replace 'Armature' with your armature name
#                if armature and armature.pose:
#                    bone = armature.pose.bones.get(target_name)
#                    if bone:
#                        direction = (target_empty.location - bone.head)
#                        bone.tail = bone.head + direction.normalized() * bone_length

#    except BlockingIOError:
#        pass  # No data received, continue
#    except json.JSONDecodeError as e:
#        print(f"Error decoding JSON: {e}")
#    except Exception as e:
#        print(f"Unexpected error: {e}")

#    return 0.1  # Reschedule the function to run again in 0.1 seconds

## Function to stop the script
#def stop_script():
#    try:
#        bpy.app.timers.unregister(receive_data)
#        sock.close()
#        print("Script stopped.")
#    except ValueError:
#        print("Script already stopped.")

## Store initial bone lengths
#store_initial_bone_lengths()

## Register the timer function
#bpy.app.timers.register(receive_data)

#print("Blender script is running and listening for hand tracking data...")

import bpy
import socket
import json
from mathutils import Vector

# Socket setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('127.0.0.1', 5004))  # Adjust this to match your sender's address and port
sock.setblocking(False)  # Non-blocking socket

# IK target naming convention (matches joint names in incoming data)
ik_targets = {
    "THUMB_CMC": "THUMB_CMC",
    "THUMB_MCP": "THUMB_MCP",
    "THUMB_IP": "THUMB_IP",
    "THUMB_TIP": "THUMB_TIP",
    "INDEX_FINGER_MCP": "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP": "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP": "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP": "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP": "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP": "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP": "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP": "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP": "RING_FINGER_MCP",
    "RING_FINGER_PIP": "RING_FINGER_PIP",
    "RING_FINGER_DIP": "RING_FINGER_DIP",
    "RING_FINGER_TIP": "RING_FINGER_TIP",
    "PINKY_MCP": "PINKY_MCP",
    "PINKY_PIP": "PINKY_PIP",
    "PINKY_DIP": "PINKY_DIP",
    "PINKY_TIP": "PINKY_TIP",
}

# Parameters
SMOOTHING_FACTOR = 0.6  # Weight for smoothing (0.0 to 1.0, higher is more reactive)
UPDATE_THRESHOLD = 0.001  # Minimum distance change required for updates (in meters)

# Reference length landmarks
REFERENCE_START = "WRIST"
REFERENCE_END = "MIDDLE_FINGER_MCP"
REFERENCE_DISTANCE = 0.386  # Reference distance in meters (adjust based on your system)

# Dictionary to store initial bone lengths and previous positions for smoothing
initial_bone_lengths = {}
previous_positions = {}

# Store the initial lengths of bones at script start
def store_initial_bone_lengths():
    armature = bpy.data.objects.get("Armature")  # Replace 'Armature' with your armature's name
    if not armature or not armature.pose:
        print("Armature not found or invalid. Ensure the correct armature name.")
        stop_script()
        return

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    for bone_name in ik_targets.keys():
        previous_positions[bone_name] = None  # Initialize previous positions for smoothing

# Timer function to update IK targets
def receive_data():
    try:
        data, _ = sock.recvfrom(4096)
        message = data.decode('utf-8')

        # Stop script if "stop" signal received
        if message.strip().lower() == "stop":
            print("Received stop signal. Exiting...")
            stop_script()
            return None

        # Parse JSON data
        coordinates = json.loads(message)

        # Print the parsed JSON data
#       print("Received JSON data:", json.dumps(coordinates, indent=4))  # Pretty print the JSON
        # Ensure reference landmarks exist
        if REFERENCE_START not in coordinates or REFERENCE_END not in coordinates:
            print(f"Reference landmarks {REFERENCE_START} or {REFERENCE_END} missing. Skipping frame.")
            return

        # Calculate scale normalization factor
        start_pos = Vector((
            coordinates[REFERENCE_START]["x"],
            coordinates[REFERENCE_START]["z"],
            -coordinates[REFERENCE_START]["y"]
        ))
        end_pos = Vector((
            coordinates[REFERENCE_END]["x"],
            coordinates[REFERENCE_END]["z"],
            -coordinates[REFERENCE_END]["y"]
        ))
        current_distance = (end_pos - start_pos).length
        scale_factor = REFERENCE_DISTANCE / current_distance if current_distance > 0 else 1.0

        # Update IK targets
        for landmark, target_name in ik_targets.items():
            if landmark not in coordinates:
                continue

            target_empty = bpy.data.objects.get(target_name)
            if not target_empty:
                continue

            # Normalize position and apply scale factor
            raw_pos = Vector((
                coordinates[landmark]["x"],
                coordinates[landmark]["z"],
                -coordinates[landmark]["y"]
            ))
            normalized_pos = (raw_pos - start_pos) * scale_factor

            # Get the previous position for smoothing
            prev_pos = previous_positions.get(landmark)
            if prev_pos is None:
                smoothed_pos = normalized_pos
            else:
                # Calculate change in position
                change = (normalized_pos - prev_pos).length
                if change < UPDATE_THRESHOLD:
                    continue  # Skip minor changes below the threshold

                # Smooth the position
                smoothed_pos = prev_pos.lerp(normalized_pos, SMOOTHING_FACTOR)

            # Update the target location and store the smoothed position
            target_empty.location = smoothed_pos
            previous_positions[landmark] = smoothed_pos

    except BlockingIOError:
        pass  # No data received, continue
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return 0.1  # Reschedule the function to run again in 0.1 seconds

# Function to stop the script
def stop_script():
    try:
        bpy.app.timers.unregister(receive_data)
        sock.close()
        print("Script stopped.")
    except ValueError:
        print("Script already stopped.")

# Store initial bone lengths
store_initial_bone_lengths()

# Register the timer function
bpy.app.timers.register(receive_data)

print("Blender script is running and listening for hand tracking data...")
