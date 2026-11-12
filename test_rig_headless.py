"""Rig behaviour test that runs inside headless Blender against your real .blend.

    blender -b --disable-autoexec untitled.blend --python v5/test_rig_headless.py -- v5

Nothing is saved: every change stays in memory and is discarded when Blender exits.
Prints a RESULT line per check and exits non-zero if any check fails.
"""
import json
import math
import socket
import sys
import time

V5_DIR = sys.argv[sys.argv.index("--") + 1]
sys.path.insert(0, V5_DIR)

import bpy
from mathutils import Matrix, Vector

import blender_receiver as br

br.HAND_YAW_DEG = 0.0  # this suite checks absolute positions and palm orientation of the unturned hand
br.UNMIRROR_HAND = False  # ... in the camera's own axes; test_features_headless covers undoing the mirror
import protocol
import rig_tools as rt

arm = bpy.data.objects["Armature"]
bones = arm.data.bones
failures = []


def check(name, ok, detail=""):
    print(f"RESULT {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


def evaluated():
    bpy.context.view_layer.update()
    return arm.evaluated_get(bpy.context.evaluated_depsgraph_get())


MW = arm.matrix_world.copy()  # the armature object is rotated, so empties must be in world space


def rest_targets():
    t = {"WRIST": MW @ Vector((0, 0, 0))}
    for bone, target in rt.BONE_TARGETS.items():
        t[target] = MW @ bones[bone].tail_local
    return t


def fk_targets(flex):
    """Rest landmarks (world space) with every non-thumb finger bent by
    flex = (mcp, pip, dip) extra degrees about its own hinge axis."""
    t = rest_targets()
    for names in rt.FINGERS.values():
        h = rt.finger_hinge(bones, names)
        dirs = [rt._dir(bones[n]) for n in names]
        pos = bones[names[0]].tail_local.copy()
        cum = 0.0
        for i in range(1, 4):
            cum += math.radians(flex[i - 1])
            pos = pos + (Matrix.Rotation(cum, 3, h) @ dirs[i]) * bones[names[i]].length
            t[rt.BONE_TARGETS[names[i]]] = MW @ pos
    return t


def set_empties(targets, offset=Vector((0, 0, 0))):
    for name, p in targets.items():
        bpy.data.objects[name].location = p + offset


def tail_errors(ev):
    errs = {}
    for bone, target in rt.BONE_TARGETS.items():
        tail = ev.matrix_world @ ev.pose.bones[bone].tail
        errs[bone] = (tail - bpy.data.objects[target].matrix_world.translation).length * 1000
    return errs


def finger_flexions(ev, names):
    """Total flexion (degrees) at MCP, PIP, DIP of one finger in the evaluated pose."""
    h = rt.finger_hinge(bones, names)
    d = []
    for n in names:
        pb = ev.pose.bones[n]
        d.append((pb.tail - pb.head).normalized())
    return [math.degrees(rt._signed_angle(h, d[i], d[i + 1])) for i in range(3)]


# ---- 1. static validation of the real file --------------------------------
report = rt.validate()
for level, msg in report:
    print(f"VALIDATE [{level}] {msg}")
check("validate_has_no_errors", not any(lv == "ERROR" for lv, _ in report))

_, before_notes = rt.plan_joint_limits(arm)
print("LIMITS before roll alignment:", sum("skipped" in n for n in before_notes), "bones skipped")
roll_changes = rt.align_finger_rolls(arm, apply=False)
check("roll_dry_run_changes_nothing", len(rt.align_finger_rolls(arm, apply=False)) == len(roll_changes))
print("ROLLS would change", len(roll_changes), "bones")
rt.align_finger_rolls(arm, apply=True)  # in memory only
check("rolls_aligned_after_apply", rt.align_finger_rolls(arm, apply=False) == [])
plan, notes = rt.plan_joint_limits(arm)
for n in notes:
    print("LIMITS", n)
check("limits_planned_for_all_12_finger_bones", len(plan) == 12, f"planned={len(plan)}/12")

subtargets = lambda: {b: arm.pose.bones[b].constraints[0].subtarget for b in rt.BONE_TARGETS}
snapshot = subtargets()
changes = rt.fix_ik(apply=False)
check("fix_ik_dry_run_changes_nothing", subtargets() == snapshot)
print("FIX_IK would change", len(changes), "constraints")

# ---- 2. IK reaches forward-kinematics poses, with and without limits ------
rest_flex = {f: [math.degrees(rt._signed_angle(rt.finger_hinge(bones, n), rt._dir(bones[n[i]]),
                                               rt._dir(bones[n[i + 1]]))) for i in range(3)]
             for f, n in rt.FINGERS.items()}
print("REST flexion (mcp, pip, dip) deg:", {f: [round(a) for a in v] for f, v in rest_flex.items()})

index_rest = rest_flex["index"]
scenarios = {
    "rest": (0, 0, 0),
    "moderate_curl": (10, 30, 20),
    "open_straight": (0, -index_rest[1], -index_rest[2]),
    "over_flexed_pip": (0, 120, 0),
}
results = {}
for limits in (False, True):
    rt.clear_joint_limits(arm)
    if limits:
        rt.apply_joint_limits(arm, plan)
    for name, flex in scenarios.items():
        set_empties(fk_targets(flex))
        ev = evaluated()
        errs = tail_errors(ev)
        worst = max(errs.values())
        pip = finger_flexions(ev, rt.FINGERS["index"])[1]
        results[(limits, name)] = (worst, pip)
        print(f"POSE limits={'on ' if limits else 'off'} {name:16s} worst tail error "
              f"{worst:7.2f} milli-units  index PIP flexion {pip:6.1f} deg")

for name in ("rest", "moderate_curl", "open_straight"):
    check(f"ik_reaches_{name}_without_limits", results[(False, name)][0] < 2.0,
          f"{results[(False, name)][0]:.2f}")
check("rest_unchanged_with_limits", results[(True, "rest")][0] < 2.0, f"{results[(True, 'rest')][0]:.2f}")
check("moderate_curl_allowed_by_limits", results[(True, "moderate_curl")][0] < 2.0,
      f"{results[(True, 'moderate_curl')][0]:.2f}")
check("over_flexed_reaches_beyond_110_without_limits", results[(False, "over_flexed_pip")][1] > 115,
      f"{results[(False, 'over_flexed_pip')][1]:.0f} deg")
# the IK limits are only a wider safety net (the receiver clamps exactly at 110 degrees)
check("ik_safety_net_bounds_pip", 110 < results[(True, "over_flexed_pip")][1] < 110 + rt.IK_SLACK_DEG + 8,
      f"{results[(True, 'over_flexed_pip')][1]:.0f} deg vs {results[(False, 'over_flexed_pip')][1]:.0f} unconstrained")

# ---- 3. real receiver code driving the real rig ---------------------------
rt.clear_joint_limits(arm)
K = 4.0  # Blender units per MediaPipe metre (arbitrary; the receiver must recover it)


def to_mp(targets, centre=Vector((0.01, 0.02, -0.03))):
    pts = []
    for name in br.LANDMARK_NAMES:
        b = targets[name]
        pts.append((b.x / K + centre.x, -b.z / K + centre.y, b.y / K + centre.z))
    return pts


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("127.0.0.1", 0))
s.setblocking(False)
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = s.getsockname()
state = br.State()
state.rig = br.Rig(arm)
rt.apply_joint_limits(arm, plan)  # the receiver is checked with the real-hand limits on
target_pose = fk_targets((10, 30, 20))
mp_pts = to_mp(target_pose)
for seq in range(1, 41):  # the rig is a right-hand shape, which MediaPipe labels "Left"
    tx.sendto(protocol.encode_frame("rig", seq, time.time(), "Left", mp_pts), addr)
time.sleep(0.05)
# This check is about scale and axis mapping, so the joint limiter is off for it: the rig's own
# fingers are not perfectly flat, and the limiter (correctly) flattens them a little.
br.CLAMP_TO_LIMITS = False
br.process(s, state)
br.CLAMP_TO_LIMITS = True
worst_empty = max((bpy.data.objects[n].location - target_pose[n]).length * 1000
                  for n in br.LANDMARK_NAMES if n != "WRIST")
check("receiver_places_empties_at_rig_scale", worst_empty < 0.5, f"{worst_empty:.3f} milli-units")
rig_span = (bones["wrist_to_middle_mcp"].tail_local - bones["wrist_to_middle_mcp"].head_local).length
expected_scale = K  # the receiver scales the live hand's palm span to the rig's own palm bone (rig_span)
check("receiver_calibrates_scale_from_hand_span", abs(state.scale / expected_scale - 1) < 1e-3,
      f"scale={state.scale:.4f} expected={expected_scale:.4f}")
ev = evaluated()
check("rig_follows_receiver_output", max(tail_errors(ev).values()) < 2.0,
      f"{max(tail_errors(ev).values()):.2f} milli-units")

# translation
br.TRANSLATE_HAND = True
br.restore_armature(state)
state2 = br.State()
state2.rig = br.Rig(arm)
rest_loc = tuple(arm.location)
for seq in range(1, 41):  # the first processed frame fixes the anchor position
    tx.sendto(protocol.encode_frame("rig2", seq, time.time(), "Left", mp_pts, (0.0, 0.0, 0.5)), addr)
time.sleep(0.05)
br.process(s, state2)
check("no_translation_at_anchor", (Vector(arm.location) - Vector(rest_loc)).length < 1e-6)
tx.sendto(protocol.encode_frame("rig2", 41, time.time(), "Left", mp_pts, (0.05, -0.02, 0.5)), addr)
time.sleep(0.05)
br.process(s, state2)
moved = Vector(arm.location) - Vector(rest_loc)
expected = Vector(br.to_blender_axes(0.05 * state2.scale, -0.02 * state2.scale, 0.0))
check("translation_moves_armature", (moved - expected).length < 1e-6, f"moved={tuple(round(v, 4) for v in moved)}")
ev = evaluated()
check("rig_follows_while_translated", max(tail_errors(ev).values()) < 2.0,
      f"{max(tail_errors(ev).values()):.2f} milli-units")
br.restore_armature(state2)
br.TRANSLATE_HAND = False
check("armature_restored", (Vector(arm.location) - Vector(rest_loc)).length < 1e-9)
check("armature_rotation_restored", all(abs(a - b) < 1e-6 for ra, rb in zip(arm.matrix_world, MW)
                                        for a, b in zip(ra, rb)))
s.close()
tx.close()


# ---- 3b. palm orientation and hand shape, with the finger limits ON -----------
def rotated(points, q):
    return {k: q @ v for k, v in points.items()}


def reflected(points):
    """The same hand with the opposite shape: reflect across its palm symmetry plane."""
    frame = br.palm_frame(tuple(points["WRIST"]), tuple(points["INDEX_FINGER_MCP"]),
                          tuple(points["MIDDLE_FINGER_MCP"]), tuple(points["PINKY_MCP"]))
    x = Vector(frame[0])
    return {k: v - 2 * v.dot(x) * x for k, v in points.items()}


def run_palm_case(points, label, track=True):
    br.TRACK_PALM = track
    br.restore_armature(st_palm)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    sender_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    st_palm.new_session(f"p{time.time_ns()}")
    for seq in range(1, 11):
        sender_sock.sendto(protocol.encode_frame(st_palm.sid, seq, time.time(), label, to_mp(points)),
                           sock.getsockname())
    time.sleep(0.03)
    st_palm.sid = None  # let drain() start a fresh session for these packets
    br.process(sock, st_palm)
    ev = evaluated()
    worst = max(tail_errors(ev).values())
    flex = [finger_flexions(ev, names)[1] for names in rt.FINGERS.values()]
    errors = []  # angle between each rig bone and the same segment of the live hand, in world space
    mw3 = ev.matrix_world.to_3x3()
    for finger, bone_names in br.BONE_CHAINS.items():
        idx = br.CHAIN_LANDMARKS[finger]
        for s, bone in enumerate(bone_names):
            live = (points[br.LANDMARK_NAMES[idx[s + 1]]] - points[br.LANDMARK_NAMES[idx[s]]]).normalized()
            pb = ev.pose.bones[bone]
            errors.append(math.degrees((mw3 @ (pb.tail - pb.head)).normalized().angle(live, 0.0)))
    sock.close()
    sender_sock.close()
    return worst, flex, sum(errors) / len(errors)


st_palm = br.State()
st_palm.rig = br.Rig(arm)
rt.apply_joint_limits(arm, plan)
base_curl = fk_targets((10, 30, 20))
over_flexed = fk_targets((0, 120, 0))
Rz = lambda deg: Matrix.Rotation(math.radians(deg), 3, "Z")
Rx = lambda deg: Matrix.Rotation(math.radians(deg), 3, "X")
Ry = lambda deg: Matrix.Rotation(math.radians(deg), 3, "Y")
orientations = {
    "as_rig": Matrix.Identity(3),
    "rolled_90_about_fingers": Rz(90),
    "back_of_hand_to_camera": Rz(180),
    "pitched_60": Rx(60),
    "yawed_45": Ry(45),
    "combined": Rz(120) @ Rx(-40) @ Ry(25),
}
palm_worst = {}
for shape in ("same_shape", "opposite_shape"):
    for oname, q in orientations.items():
        pts = rotated(base_curl if shape == "same_shape" else reflected(base_curl), q)
        label = "Left" if shape == "same_shape" else "Right"
        worst, _, _ = run_palm_case(pts, label)
        mirrored_ok = st_palm.mirror == (shape == "opposite_shape")
        palm_worst[(shape, oname)] = worst
        print(f"PALM {shape:14s} {oname:24s} worst tail error {worst:7.2f} milli-units"
              f"  rig mirrored={st_palm.mirror}")
        check(f"palm_{shape}_{oname}_chooses_right_mirror", mirrored_ok)
        check(f"palm_{shape}_{oname}_tracks_with_limits_on", worst < 2.0, f"{worst:.2f}")

# Without palm tracking the targets stay reachable (the empties stay attached) but a hand turned
# away cannot be mimicked: this is why TRACK_PALM exists.
back = rotated(base_curl, orientations["back_of_hand_to_camera"])
_, _, off_err = run_palm_case(back, "Left", track=False)
_, _, on_err = run_palm_case(back, "Left", track=True)
print(f"PALM comparison, back of hand to camera: mean bone direction error "
      f"TRACK_PALM off {off_err:.1f} deg, on {on_err:.2f} deg")
check("palm_tracking_is_needed_to_mimic_a_turned_hand", off_err > 20.0 and on_err < 2.0,
      f"off {off_err:.1f} on {on_err:.2f}")
br.TRACK_PALM = True

# limits still bound the fingers when the hand is turned or the other shape
for shape in ("same_shape", "opposite_shape"):
    pts = rotated(over_flexed if shape == "same_shape" else reflected(over_flexed), Rz(180))
    _, flex, _ = run_palm_case(pts, "Left" if shape == "same_shape" else "Right")
    check(f"limits_still_clamp_{shape}_when_turned", max(flex) < 118, f"max PIP {max(flex):.0f} deg")
br.restore_armature(st_palm)
rt.clear_joint_limits(arm)

# ---- 4. timing of the IK solve ---------------------------------------------
def bench(frames=200):
    t0 = time.perf_counter()
    for i in range(frames):
        a = 40 * (1 + math.sin(i / 8.0)) / 2
        set_empties(fk_targets((a * 0.3, a, a * 0.7)))
        evaluated()
    return (time.perf_counter() - t0) / frames * 1000

iterations_before = {b: arm.pose.bones[b].constraints[0].iterations for b in rt.BONE_TARGETS}
base = bench()
for b in rt.BONE_TARGETS:
    arm.pose.bones[b].constraints[0].iterations = 20
fast = bench()
for b, it in iterations_before.items():
    arm.pose.bones[b].constraints[0].iterations = it
print(f"BENCH per frame (Python FK + empties + depsgraph/IK, no viewport drawing): "
      f"iterations=500: {base:.2f} ms, iterations=20: {fast:.2f} ms")
check("ik_evaluation_fits_in_60hz_budget", base < 16.0, f"{base:.2f} ms")

print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
