"""Rig validation, IK cleanup and finger joint limits. Run inside Blender.

Paste into a Text block (or run with `blender file.blend --python v5/rig_tools.py`).
It prints a report and changes nothing unless APPLY = True. Everything is idempotent,
and clear_joint_limits() undoes the limits.

This is the single place that maps rig bones to landmark empties (BONE_TARGETS). The
receiver only needs the empties' names, which are the landmark names.
"""
import math

try:
    import bpy
    from mathutils import Vector
except ImportError:  # imported outside Blender
    bpy = None

# ---- configuration ---------------------------------------------------------
APPLY = False              # False = report only. True = write the changes.
WITH_JOINT_LIMITS = True   # only used when APPLY is True
ARMATURE_NAME = "Armature"
# Absolute flexion ranges in degrees (0 = finger straight).
FLEXION_RANGE = {"mcp": (-20.0, 90.0), "pip": (0.0, 110.0), "dip": (0.0, 90.0)}
MCP_ABDUCTION_DEG = 30.0   # side-to-side swing allowed at each knuckle
SIDE_MARGIN_DEG = 5.0      # extra side-axis slack at every finger joint
# The IK limits are a safety net, not the exact anatomy: blender_receiver.py clamps its targets
# to the ranges above, and Blender's per-axis limits cannot match a real hinge exactly, so they
# are widened by this much to never block a target the receiver considers legal.
# Measured on a real recording of curling fingers: with the side range only +25 degrees the rig
# stayed straight on 14% of curled joints; with +60 or more it matched the unlimited rig (0%).
IK_SLACK_DEG = 40.0        # extra range on the flexion axis
IK_SIDE_SLACK_DEG = 90.0   # extra range on the side axis (real fingers curl in tilted planes)
# ----------------------------------------------------------------------------

LANDMARK_EMPTIES = (
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
)

# bone -> empty its tail should reach
BONE_TARGETS = {
    "wrist_to_thumb_cmc": "THUMB_CMC",
    "wrist_to_index_mcp": "INDEX_FINGER_MCP",
    "wrist_to_middle_mcp": "MIDDLE_FINGER_MCP",
    "wrist_to_ring_mcp": "RING_FINGER_MCP",
    "wrist_to_pinky_mcp": "PINKY_MCP",
    "thumb_cmc_to_thumb_mcp": "THUMB_MCP",
    "thumb_mcp_to_thumb_ip": "THUMB_IP",
    "thumb_ip_to_thumb_tip": "THUMB_TIP",
    "index_mcp_to_index_pip": "INDEX_FINGER_PIP",
    "index_pip_to_index_dip": "INDEX_FINGER_DIP",
    "index_dip_to_index_tip": "INDEX_FINGER_TIP",
    "middle_mcp_to_middle_pip": "MIDDLE_FINGER_PIP",
    "middle_pip_to_middle_dip": "MIDDLE_FINGER_DIP",
    "middle_dip_to_middle_tip": "MIDDLE_FINGER_TIP",
    "ring_mcp_to_ring_pip": "RING_FINGER_PIP",
    "ring_pip_to_ring_dip": "RING_FINGER_DIP",
    "ring_dip_to_ring_tip": "RING_FINGER_TIP",
    "pinky_mcp_to_pinky_pip": "PINKY_PIP",
    "pinky_pip_to_pinky_dip": "PINKY_DIP",
    "pinky_dip_to_pinky_tip": "PINKY_TIP",
}
ROOT_BONES = tuple(b for b in BONE_TARGETS if b.startswith("wrist_to_"))
# palm bone, then proximal, middle, distal bone of each non-thumb finger
FINGERS = {
    "index": ("wrist_to_index_mcp", "index_mcp_to_index_pip", "index_pip_to_index_dip", "index_dip_to_index_tip"),
    "middle": ("wrist_to_middle_mcp", "middle_mcp_to_middle_pip", "middle_pip_to_middle_dip", "middle_dip_to_middle_tip"),
    "ring": ("wrist_to_ring_mcp", "ring_mcp_to_ring_pip", "ring_pip_to_ring_dip", "ring_dip_to_ring_tip"),
    "pinky": ("wrist_to_pinky_mcp", "pinky_mcp_to_pinky_pip", "pinky_pip_to_pinky_dip", "pinky_dip_to_pinky_tip"),
}
_JOINT_NAMES = ("mcp", "pip", "dip")
_AXES = "xyz"


def _armature():
    return bpy.data.objects.get(ARMATURE_NAME)


def validate():
    """Return a list of (level, message); level is 'OK', 'WARN' or 'ERROR'."""
    out = []
    add = lambda level, msg: out.append((level, msg))
    arm = _armature()
    if arm is None or arm.type != "ARMATURE":
        add("ERROR", f"armature '{ARMATURE_NAME}' not found")
        return out
    add("OK", f"armature '{ARMATURE_NAME}' found")

    scale = arm.matrix_world.to_scale()
    if any(abs(c - 1) > 1e-4 for c in scale):
        add("WARN", f"armature object scale is {tuple(round(c, 3) for c in scale)}; bone lengths in "
                    "the world differ from the edit-mode lengths (the measurement below is in world units)")
    if arm.parent:
        add("WARN", "armature has a parent, so its .location is parent-relative (affects TRANSLATE_HAND)")
    angle = math.degrees(arm.matrix_world.to_quaternion().angle)
    if angle > 0.1:
        add("OK", f"armature object is rotated {angle:.0f} deg; fine, because the empties and the IK "
                  "targets are both in world space")

    missing = [n for n in LANDMARK_EMPTIES if bpy.data.objects.get(n) is None]
    add("ERROR" if missing else "OK",
        f"missing empties: {', '.join(missing)}" if missing else "all 21 landmark empties exist")
    parented = [n for n in LANDMARK_EMPTIES
                if bpy.data.objects.get(n) is not None and bpy.data.objects[n].parent]
    if parented:
        add("WARN", f"empties with a parent (their .location is parent-relative): {', '.join(parented)}")

    for bone in BONE_TARGETS:
        if arm.pose.bones.get(bone) is None:
            add("ERROR", f"bone '{bone}' missing")
    for bone, target in BONE_TARGETS.items():
        pb = arm.pose.bones.get(bone)
        if pb is None:
            continue
        iks = [c for c in pb.constraints if c.type == "IK"]
        if len(iks) != 1:
            add("ERROR", f"{bone}: expected exactly 1 IK constraint, found {len(iks)}")
            continue
        ik = iks[0]
        if ik.target is None or ik.target.name != target:
            add("ERROR", f"{bone}: IK target is {ik.target.name if ik.target else None}, expected {target}")
        if ik.target is not None and ik.target.type != "ARMATURE" and ik.subtarget:
            add("WARN", f"{bone}: subtarget '{ik.subtarget}' is set but the target is an empty (fix_ik clears it)")
        expected_chain = 0 if bone in ROOT_BONES else 1
        if ik.chain_count not in ((0, 1) if bone in ROOT_BONES else (expected_chain,)):
            add("WARN", f"{bone}: chain_count={ik.chain_count}, expected {expected_chain}")
        if pb.ik_stretch > 0:
            add("WARN", f"{bone}: ik_stretch={pb.ik_stretch}; the bone may stretch away from its length")
    if not any(lv == "ERROR" for lv, _ in out):
        add("OK", "every bone has one IK constraint with the right empty target")

    wrist_heads = [arm.matrix_world @ arm.data.bones[b].head_local
                   for b in ROOT_BONES if arm.data.bones.get(b)]  # world space
    if wrist_heads:
        far = max(h.length for h in wrist_heads)
        add("OK" if far < 1e-3 else "WARN",
            "root bones start at the world origin (the receiver pins the WRIST empty there)"
            if far < 1e-3 else f"root bones start up to {far:.4f} from the world origin; "
                               "the receiver assumes the wrist is at (0,0,0)")

    mid = arm.data.bones.get("wrist_to_middle_mcp")
    if mid is not None:
        measured = (arm.matrix_world @ mid.tail_local - arm.matrix_world @ mid.head_local).length
        add("OK", f"rig WRIST->MIDDLE_FINGER_MCP = {measured:.4f} (the receiver scales the live hand to this length)")
    return out


def fix_ik(apply):
    """Clear the meaningless subtarget on constraints that target an empty."""
    changes = []
    arm = _armature()
    if arm is None:
        return changes
    for bone in BONE_TARGETS:
        pb = arm.pose.bones.get(bone)
        if pb is None:
            continue
        for ik in (c for c in pb.constraints if c.type == "IK"):
            if ik.target is not None and ik.target.type != "ARMATURE" and ik.subtarget:
                changes.append(f"{bone}: clear subtarget '{ik.subtarget}'")
                if apply:
                    ik.subtarget = ""
    return changes


def _dir(bone):
    return (bone.tail_local - bone.head_local).normalized()


def _signed_angle(axis, u, v):
    return math.atan2(axis.dot(u.cross(v)), u.dot(v))


def finger_hinge(bones, names):
    """Shared flexion axis of one finger (armature space, unit vector) from how its PIP
    and DIP joints are bent at rest. Positive rotation about it bends the finger more.
    None if the rest pose is straight. `names` is (palm, proximal, middle, distal)."""
    dirs = [_dir(bones[n]) for n in names]
    axis_sum = dirs[1].cross(dirs[2]) + dirs[2].cross(dirs[3])
    return axis_sum.normalized() if axis_sum.length >= 1e-3 else None


def align_finger_rolls(arm, apply):
    """Roll each finger bone so its local Z axis is the finger's hinge axis.

    Only the bone's local frame changes (head and tail stay put, so the rest pose and the
    skinning are identical). This lets one local axis represent the hinge exactly, which the
    joint limits need. Bones already aligned are left alone. Returns a list of changes."""
    bones = arm.data.bones
    todo = []  # (bone name, hinge axis, current alignment)
    for names in FINGERS.values():
        if any(bones.get(n) is None for n in names):
            continue
        h = finger_hinge(bones, names)
        if h is None:
            continue
        for n in names[1:]:
            # align_roll can only match the part of h that is perpendicular to the bone
            d = _dir(bones[n])
            target = (h - d * h.dot(d)).normalized()
            alignment = abs(bones[n].matrix_local.to_3x3().col[2].dot(target))
            if alignment < 0.999:
                todo.append((n, h, alignment))
    if apply and todo:
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="EDIT")
        for n, h, _ in todo:
            arm.data.edit_bones[n].align_roll(h)
        bpy.ops.object.mode_set(mode="OBJECT")
    return [f"{n}: roll so local Z follows the hinge axis (alignment was {a:.2f})" for n, _, a in todo]


def plan_joint_limits(arm):
    """Work out per-bone IK limits from the rig's own rest pose.

    Finger joints are hinges that share one axis. That axis is taken from how the
    PIP and DIP joints are bent at rest (their rest bend is the flexion direction), then
    expressed in each bone's local frame. Limits are stored relative to the rest pose.
    Returns (plan, notes); plan maps bone name -> dict of pose-bone IK settings.
    """
    plan, notes = {}, []
    bones = arm.data.bones
    for finger, names in FINGERS.items():
        if any(bones.get(n) is None for n in names):
            notes.append(f"{finger}: bones missing, skipped")
            continue
        h = finger_hinge(bones, names)
        if h is None:
            notes.append(f"{finger}: rest pose is straight, cannot infer the hinge axis; skipped")
            continue
        dirs = [_dir(bones[n]) for n in names]
        for joint, bone_name, parent_dir, child_dir in zip(
                _JOINT_NAMES, names[1:], dirs[:-1], dirs[1:]):
            bone = bones[bone_name]
            local = bone.matrix_local.to_3x3().inverted() @ h
            flex_idx = max((0, 2), key=lambda i: abs(local[i]))
            if abs(local[flex_idx]) < 0.85:
                notes.append(f"{bone_name}: hinge axis is not aligned with a local axis "
                             f"(components {tuple(round(c, 2) for c in local)}); skipped")
                continue
            side_idx = 2 if flex_idx == 0 else 0
            rest = math.degrees(_signed_angle(h, parent_dir, child_dir))
            lo, hi = FLEXION_RANGE[joint]
            # Allowed bend about the hinge axis h, relative to the rest pose. A rotation phi about
            # h is c_flex * phi about the local flex axis plus c_side * phi about the side axis,
            # so both local axes get the matching share (the side axis also gets a small margin).
            phi = (math.radians(lo - rest), math.radians(hi - rest))
            c_flex, c_side = local[flex_idx], local[side_idx]
            slack = math.radians(IK_SLACK_DEG)
            flex_rng = sorted((c_flex * phi[0], c_flex * phi[1]))
            flex_rng = [flex_rng[0] - slack, flex_rng[1] + slack]
            side_rng = sorted((c_side * phi[0], c_side * phi[1]))
            side_slack = math.radians(IK_SIDE_SLACK_DEG)
            margin = math.radians(SIDE_MARGIN_DEG) + side_slack
            side_rng = [side_rng[0] - margin, side_rng[1] + margin]
            if joint == "mcp":  # knuckles may also swing sideways
                swing = math.radians(MCP_ABDUCTION_DEG) + side_slack
                side_rng = [min(side_rng[0], -swing), max(side_rng[1], swing)]
            settings = {f"lock_ik_{a}": False for a in _AXES}
            settings.update({f"use_ik_limit_{a}": False for a in _AXES})
            settings["lock_ik_y"] = True                    # twist about the bone axis
            for idx, rng in ((flex_idx, flex_rng), (side_idx, side_rng)):
                a = _AXES[idx]
                settings[f"use_ik_limit_{a}"] = True
                settings[f"ik_min_{a}"], settings[f"ik_max_{a}"] = rng
            plan[bone_name] = settings
            notes.append(f"{bone_name}: flex axis local {_AXES[flex_idx].upper()} "
                         f"(alignment {abs(c_flex):.2f}), rest flexion {rest:.0f} deg, "
                         f"allowed {lo:.0f}..{hi:.0f} deg")
    return plan, notes


def apply_joint_limits(arm, plan):
    for bone_name, settings in plan.items():
        pb = arm.pose.bones[bone_name]
        for attr, value in settings.items():
            setattr(pb, attr, value)


def clear_joint_limits(arm=None):
    """Remove every IK lock and limit this module may have set."""
    arm = arm or _armature()
    for names in FINGERS.values():
        for bone_name in names[1:]:
            pb = arm.pose.bones.get(bone_name)
            if pb is None:
                continue
            for a in _AXES:
                setattr(pb, f"lock_ik_{a}", False)
                setattr(pb, f"use_ik_limit_{a}", False)
                setattr(pb, f"ik_min_{a}", -math.pi)
                setattr(pb, f"ik_max_{a}", math.pi)


def run():
    print("=" * 60)
    print(f"rig_tools: {'APPLYING changes' if APPLY else 'report only (set APPLY = True to write)'}")
    for level, msg in validate():
        print(f"  [{level}] {msg}")
    for change in fix_ik(APPLY):
        print(f"  {'fixed' if APPLY else 'would fix'}: {change}")
    arm = _armature()
    if arm is not None and WITH_JOINT_LIMITS:
        for change in align_finger_rolls(arm, APPLY):
            print(f"  {'fixed' if APPLY else 'would fix'}: {change}")
    if arm is not None:
        plan, notes = plan_joint_limits(arm)
        for n in notes:
            print(f"  limits: {n}")
        if APPLY and WITH_JOINT_LIMITS:
            apply_joint_limits(arm, plan)
            print(f"  applied joint limits to {len(plan)} bones "
                  "(clear_joint_limits() removes them)")
        else:
            print(f"  joint limits for {len(plan)} bones planned, not applied")
    print("=" * 60)


if bpy is not None and __name__ == "__main__":
    run()
