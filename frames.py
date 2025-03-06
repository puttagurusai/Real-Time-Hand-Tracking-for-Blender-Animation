import bpy

# Bone-to-landmark naming for all joints
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

# Armature name
ARMATURE_NAME = "Armature"
armature = bpy.data.objects.get(ARMATURE_NAME)

if not armature:
    raise ValueError(f"Armature '{ARMATURE_NAME}' not found. Check your Blender file.")

# Ensure armature is the active object
bpy.context.view_layer.objects.active = armature

# Switch to Edit Mode to access the bones
bpy.ops.object.mode_set(mode='EDIT')

# Iterate over all bones and their corresponding landmark names
for bone_name, (start_point, end_point) in bone_to_landmarks.items():
    # Get the bone by name
    bone = armature.data.edit_bones.get(bone_name)
    if not bone:
        print(f"Bone '{bone_name}' not found. Skipping...")
        continue

#    # Create empty at the bone's head (joint location) and name it after the start_point
#    empty = bpy.data.objects.new(start_point, None)
#    bpy.context.collection.objects.link(empty)
#    empty.empty_display_size = 0.02
#    empty.empty_display_type = 'SPHERE'
#    empty.location = armature.matrix_world @ bone.head  # World space location

    # Create empty at the bone's tail (finger tip location) and name it after the end_point
    empty_tip = bpy.data.objects.new(end_point, None)
    bpy.context.collection.objects.link(empty_tip)
    empty_tip.empty_display_size = 0.02
    empty_tip.empty_display_type = 'SPHERE'
    empty_tip.location = armature.matrix_world @ bone.tail  # World space location

# Switch back to Object Mode
bpy.ops.object.mode_set(mode='OBJECT')

print("Empties created at each joint with appropriate names.")

# Create an empty at the origin (0, 0, 0)
empty = bpy.data.objects.new("WRIST", None)
bpy.context.collection.objects.link(empty)

# Set the empty's display settings
empty.empty_display_size = 0.02
empty.empty_display_type = 'SPHERE'

# Set the location of the empty at the origin
empty.location = (0, 0, 0)

print("Empty named 'Wrist' created at the origin.")
