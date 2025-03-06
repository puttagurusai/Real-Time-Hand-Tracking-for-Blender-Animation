import bpy

# Define bone-to-target mapping (use the end point of each bone chain)
bone_to_target = {
    "wrist_to_thumb_cmc": "THUMB_CMC",  # Target the end of the thumb CMC joint
    "wrist_to_index_mcp": "INDEX_FINGER_MCP",  # Target the end of the index MCP joint
    "wrist_to_middle_mcp": "MIDDLE_FINGER_MCP",  # Target the end of the middle MCP joint
    "wrist_to_ring_mcp": "RING_FINGER_MCP",  # Target the end of the ring MCP joint
    "wrist_to_pinky_mcp": "PINKY_MCP",  # Target the end of the pinky MCP joint
    "thumb_cmc_to_thumb_mcp": "THUMB_MCP",  # Target the end of the thumb MCP joint
    "thumb_mcp_to_thumb_ip": "THUMB_IP",  # Target the end of the thumb IP joint
    "thumb_ip_to_thumb_tip": "THUMB_TIP",  # Target the end of the thumb tip joint
    "index_mcp_to_index_pip": "INDEX_FINGER_PIP",  # Target the end of the index PIP joint
    "index_pip_to_index_dip": "INDEX_FINGER_DIP",  # Target the end of the index DIP joint
    "index_dip_to_index_tip": "INDEX_FINGER_TIP",  # Target the end of the index tip joint
    "middle_mcp_to_middle_pip": "MIDDLE_FINGER_PIP",  # Target the end of the middle PIP joint
    "middle_pip_to_middle_dip": "MIDDLE_FINGER_DIP",  # Target the end of the middle DIP joint
    "middle_dip_to_middle_tip": "MIDDLE_FINGER_TIP",  # Target the end of the middle tip joint
    "ring_mcp_to_ring_pip": "RING_FINGER_PIP",  # Target the end of the ring PIP joint
    "ring_pip_to_ring_dip": "RING_FINGER_DIP",  # Target the end of the ring DIP joint
    "ring_dip_to_ring_tip": "RING_FINGER_TIP",  # Target the end of the ring tip joint
    "pinky_mcp_to_pinky_pip": "PINKY_PIP",  # Target the end of the pinky PIP joint
    "pinky_pip_to_pinky_dip": "PINKY_DIP",  # Target the end of the pinky DIP joint
    "pinky_dip_to_pinky_tip": "PINKY_TIP",  # Target the end of the pinky tip joint
}

# Armature name
ARMATURE_NAME = "Armature"
armature = bpy.data.objects.get(ARMATURE_NAME)

if not armature:
    raise ValueError(f"Armature '{ARMATURE_NAME}' not found. Check your Blender file.")

# Ensure armature is the active object
bpy.context.view_layer.objects.active = armature

# Switch to Pose Mode to apply the IK constraints
bpy.ops.object.mode_set(mode='POSE')

# Iterate over the bone-to-target mapping to apply IK constraints
for bone_name, target_name in bone_to_target.items():
    # Get the bone in the armature
    pose_bone = armature.pose.bones.get(bone_name)

    if not pose_bone:
        print(f"Bone '{bone_name}' not found. Skipping...")
        continue

    # Get the target object (empty)
    target = bpy.data.objects.get(target_name)
    if not target:
        print(f"Target empty '{target_name}' not found. Skipping...")
        continue

    # Remove existing IK constraints from the bone
    for constraint in pose_bone.constraints:
        if constraint.type == 'IK':
            pose_bone.constraints.remove(constraint)

    # Add IK constraint to the bone
    ik_constraint = pose_bone.constraints.new(type='IK')
    ik_constraint.target = target  # Set the target empty for the IK
    ik_constraint.subtarget = target.name  # The name of the target object (empty)

    # Set the chain length to include multiple bones in the chain
    ik_constraint.chain_count = 1  # Adjust chain length if necessary

    # Print confirmation
    print(f"IK constraint applied to bone '{bone_name}' with target '{target_name}' and .")

# Switch back to Object Mode
bpy.ops.object.mode_set(mode='OBJECT')

print("IK constraints applied to all bones.")
