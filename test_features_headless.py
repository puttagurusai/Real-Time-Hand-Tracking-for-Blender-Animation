"""Checks for the take-baking tool, the Blender panel and relaxing to rest, on the real rig.
Run headless on the fixed copy (nothing is saved, the .blend on disk is not changed):

    blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/test_features_headless.py -- v5 [landmarks.json]
"""
import hashlib
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types

V5_DIR = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
ARGS = sys.argv[sys.argv.index("--") + 2:]
TAKE = os.path.abspath(ARGS[0]) if ARGS else os.path.join(os.path.dirname(V5_DIR), "landmarks.json")
sys.path.insert(0, V5_DIR)

import bpy

import bake_recording as bake
import blender_receiver as br
import frames_io
import protocol
import rig_tools as rt

failures = []


def check(name, ok, detail=""):
    print(f"RESULT {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


arm = bpy.data.objects[rt.ARMATURE_NAME]
BLEND = bpy.data.filepath
BLEND_HASH = sha(BLEND)


def tails(ob=None):
    """World position of every bone tip, from the evaluated rig."""
    bpy.context.view_layer.update()
    ev = (ob or arm).evaluated_get(bpy.context.evaluated_depsgraph_get())
    return {n: tuple(ev.matrix_world @ pb.tail) for n, pb in ev.pose.bones.items()}


def worst(a, b):
    return max(math.dist(a[n], b[n]) for n in a)


# ---- frame plan ------------------------------------------------------------
check("frame_plan_follows_recorded_times",
      bake.frame_plan([0.0, 0.04, 0.08], 24.0) == [0, 1]
      and bake.frame_plan([0.0], 24.0) == [0]
      and bake.frame_plan([0.0, 1.0], 24.0)[0] == 0 and bake.frame_plan([0.0, 1.0], 24.0)[-1] == 1
      and bake.frame_plan([], 24.0) == [], "")

# ---- bake: the baked take equals the live result frame by frame -----------------------------------
frames = frames_io.load_frames(TAKE, br.LANDMARK_NAMES)
frames, _ = frames_io.emulate_sender(frames)
fps = bake.scene_fps(bpy.context.scene)
plan = bake.frame_plan([f[0] for f in frames], fps)
SAMPLE = sorted({1, 6, len(plan) // 3, len(plan) // 2, len(plan) - 1, len(plan)})  # 1-based take frames

live = {}
state = br.State()
state.rig = br.Rig(arm)
state.sid = "ref"
done = -1
for n, packet in enumerate(plan):
    for i in range(done + 1, packet + 1):
        _, label, score, lm = frames[i]
        state.last_seq = i
        br._apply_frame(state, {"lm": lm, "hand": label, "score": score})
    done = packet
    if n + 1 in SAMPLE:
        live[n + 1] = tails()
br.restore_armature(state)

START = 5
last = bake.bake(arm, frames, start=START, fps=fps, bake_bones=False)
check("bake_covers_the_whole_take", last == START + len(plan) - 1 and bpy.context.scene.frame_end == last,
      f"scene frames {START}..{last}")
check("bake_keys_empties_and_armature",
      all(bpy.data.objects[n].animation_data and bpy.data.objects[n].animation_data.action
          for n in br.LANDMARK_NAMES) and arm.animation_data.action is not None, "")
gap = []
for k in SAMPLE:
    bpy.context.scene.frame_set(START + k - 1)
    gap.append(worst(tails(), live[k]))
check("keyed_take_plays_back_like_live", max(gap) < 1e-4, f"worst bone-tip difference {max(gap) * 1000:.4f} milli-units")

bake.bake_pose(arm, START, last)
left = sum(len(pb.constraints) for pb in arm.pose.bones)
check("bones_baked_without_constraints", left == 0, f"{left} constraints left")
gap = []
for k in SAMPLE:
    bpy.context.scene.frame_set(START + k - 1)
    gap.append(worst(tails(), live[k]))
check("baked_bones_equal_live_result", max(gap) < 1e-4, f"worst bone-tip difference {max(gap) * 1000:.4f} milli-units")
check("bake_did_not_touch_the_file_on_disk", sha(BLEND) == BLEND_HASH)

# ---- bake command line: never replaces a file --------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    script = os.path.join(V5_DIR, "bake_recording.py")

    def run(out, *extra):
        return subprocess.run([bpy.app.binary_path, "-b", "--disable-autoexec", BLEND, "--python", script, "--",
                               V5_DIR, TAKE, out, *extra], capture_output=True, text=True, timeout=600)

    out = os.path.join(tmp, "take.blend")
    first = run(out)
    check("bake_cli_writes_a_new_file", first.returncode == 0 and os.path.exists(out), first.stdout[-200:])
    stamp = sha(out)
    second = run(out)
    check("bake_cli_refuses_to_replace_a_file",
          second.returncode != 0 and "already exists" in (second.stdout + second.stderr) and sha(out) == stamp)
    same = run(BLEND, "overwrite=1")
    check("bake_cli_never_replaces_its_own_source", same.returncode != 0 and sha(BLEND) == BLEND_HASH)

# ---- relax to rest on the real rig --------------------------------------------------------------
bpy.ops.wm.revert_mainfile()  # back to the file as it is on disk (the bake changed the open scene)
arm = bpy.data.objects[rt.ARMATURE_NAME]

# ---- HAND_TURN_ABOUT = "world": the hand turned about the vertical axis through its wrist ------------------
import mathutils


def pose_with_yaw(yaw, count=60):
    saved_yaw, saved_mode = br.HAND_YAW_DEG, br.HAND_TURN_ABOUT
    br.HAND_YAW_DEG, br.HAND_TURN_ABOUT = yaw, "world"
    st = br.make_states()[0]
    st.sid = f"yaw{yaw}"
    for i, (_, label, score, lm) in enumerate(frames2[:count]):
        st.last_seq = i
        br._apply_frame(st, {"lm": lm, "hand": label, "score": score})
    bpy.context.view_layer.update()
    ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    inv = ev.matrix_world.inverted()
    out = {"local": {n: tuple(inv @ (ev.matrix_world @ ev.pose.bones[n].tail)) for n in rt.BONE_TARGETS},
           "matrix": ev.matrix_world.to_3x3().copy(), "location": tuple(ev.matrix_world.translation),
           "empties": {n: tuple(bpy.data.objects[n].location) for n in br.LANDMARK_NAMES}, "mirror": st.mirror}
    br.restore_armature(st)
    br.HAND_YAW_DEG, br.HAND_TURN_ABOUT = saved_yaw, saved_mode
    return out


frames2 = frames_io.load_frames(TAKE, br.LANDMARK_NAMES)
frames2, _ = frames_io.emulate_sender(frames2)
plain, turned = pose_with_yaw(0.0), pose_with_yaw(180.0)
check("turn_keeps_the_finger_shapes", max(math.dist(plain["local"][n], turned["local"][n]) for n in plain["local"]) < 1e-4
      and plain["mirror"] == turned["mirror"], f"{max(math.dist(plain['local'][n], turned['local'][n]) for n in plain['local']) * 1000:.4f} milli-units")
rz = mathutils.Matrix.Rotation(math.pi, 3, "Z")
check("armature_is_turned_180_degrees_about_z", max(abs(a - b) for ra, rb in zip(rz @ plain["matrix"], turned["matrix"]) for a, b in zip(ra, rb)) < 1e-5)
check("turn_leaves_the_hand_where_it_is", math.dist(plain["location"], turned["location"]) < 1e-9)
turn_error = max(math.dist((-plain["empties"][n][0], -plain["empties"][n][1], plain["empties"][n][2]), turned["empties"][n]) for n in br.LANDMARK_NAMES)
check("empties_are_turned_about_the_wrist", turn_error < 1e-4 and turned["empties"]["WRIST"] == (0.0, 0.0, 0.0), f"{turn_error * 1000:.4f} milli-units")
quarter = pose_with_yaw(90.0)
check("any_angle_works", max(math.dist((-plain["empties"][n][1], plain["empties"][n][0], plain["empties"][n][2]), quarter["empties"][n])
                              for n in br.LANDMARK_NAMES) < 1e-4)


# ---- no mirror: what Blender shows is the hand you hold up -----------------------------------------------
# You and Blender's front view both look along +Y; X is to your right, Z up. A real hand held with the palm
# toward the webcam is facing forward (+Y) with the fingers up. The sender flips the picture like a mirror,
# so the receiver gets that hand as (x, -z, -y); a right hand must stay a right hand and its palm must face +Y.
def real_hand(right, tilt=None):
    """The 21 landmarks the sender would send for a real hand held palm toward the camera, fingers up (tilt:
    a rotation of the real hand in your own frame, applied after it is held up)."""
    rig = br.make_states()[0].rig
    palm = rig.collision.palm
    n, u = mathutils.Vector(palm["n"]), mathutils.Vector(palm["u"])
    lat = n.cross(u)
    basis = mathutils.Matrix((tuple(lat), tuple(u), tuple(n))).transposed()
    to_user = mathutils.Matrix(((1, 0, 0), (0, 0, 1), (0, 1, 0))) @ basis.transposed()  # n -> +Y, fingers -> +Z
    points = [mathutils.Vector((0, 0, 0))] * 21
    for finger, chain in rig.chains.items():
        pos = mathutils.Vector((0, 0, 0))
        for i, d, length in zip(br.CHAIN_LANDMARKS[finger][1:], chain.rest_dirs, chain.lengths):
            pos = pos + mathutils.Vector(d) * length
            points[i] = pos
    user = [to_user @ p for p in points]  # x right, y forward (toward the camera), z up, in the rig's own size
    if not right:
        user = [mathutils.Vector((-p.x, p.y, p.z)) for p in user]  # a left hand is the mirror image of a right one
    if tilt is not None:
        user = [tilt @ p for p in user]
    return [[p.x / 4.0, -p.z / 4.0, -p.y / 4.0] for p in user], user  # what the sender sends (mirror-flipped picture)


def show(right, unmirror, yaw=0.0, tilt=None, mode="hand"):
    saved_flags = (br.UNMIRROR_HAND, br.HAND_YAW_DEG, br.HAND_TURN_ABOUT)
    br.UNMIRROR_HAND, br.HAND_YAW_DEG, br.HAND_TURN_ABOUT = unmirror, yaw, mode
    lm, user = real_hand(right, tilt)
    st = br.make_states()[0]
    st.sid = "hand"
    for i in range(30):
        st.last_seq = i
        br._apply_frame(st, {"lm": lm, "hand": "Right" if right else "Left"})
    bpy.context.view_layer.update()
    ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    rig = st.rig
    m = ev.matrix_world.to_3x3()
    out = {"palm_normal": tuple(m @ mathutils.Vector(rig.collision.palm["n"])),
           "fingers": tuple(m @ mathutils.Vector(rig.collision.palm["u"])),
           "thumb_x": bpy.data.objects["THUMB_TIP"].location.x, "mirror": st.mirror,
           "user_thumb_x": user[4].x}
    br.restore_armature(st)
    br.UNMIRROR_HAND, br.HAND_YAW_DEG, br.HAND_TURN_ABOUT = saved_flags
    return out


right_fixed, left_fixed = show(True, True), show(False, True)
right_raw = show(True, False)
print("NOMIRROR right hand:", {k: (tuple(round(c, 2) for c in v) if isinstance(v, tuple) else v) for k, v in right_fixed.items()})
print("NOMIRROR left hand: ", {k: (tuple(round(c, 2) for c in v) if isinstance(v, tuple) else v) for k, v in left_fixed.items()})
check("premise_a_real_right_hand_has_its_thumb_on_your_left", right_fixed["user_thumb_x"] < 0 < show(False, True)["user_thumb_x"],
      f"thumb x {right_fixed['user_thumb_x']:.3f}")
for name, shown in (("right", right_fixed), ("left", left_fixed)):
    check(f"{name}_palm_toward_the_camera_faces_plus_y", shown["palm_normal"][1] > 0.95, f"palm normal {tuple(round(c, 2) for c in shown['palm_normal'])}")
    check(f"{name}_fingers_up_stay_up", shown["fingers"][2] > 0.95, f"fingers {tuple(round(c, 2) for c in shown['fingers'])}")
check("right_hand_keeps_its_thumb_on_the_left_of_the_front_view", right_fixed["thumb_x"] < -0.02, f"thumb x {right_fixed['thumb_x']:.3f}")
check("left_hand_keeps_its_thumb_on_the_right_of_the_front_view", left_fixed["thumb_x"] > 0.02, f"thumb x {left_fixed['thumb_x']:.3f}")
check("right_hand_needs_no_mirroring_and_left_hand_does", right_fixed["mirror"] is False and left_fixed["mirror"] is True,
      f"right mirrored={right_fixed['mirror']}, left mirrored={left_fixed['mirror']}")
check("without_the_fix_it_was_a_mirror", right_raw["palm_normal"][1] < -0.95 and right_raw["mirror"] is True,
      f"a right hand's palm faced the viewer ({tuple(round(c, 2) for c in right_raw['palm_normal'])}) and the rig had to be flipped to a left-hand shape (mirrored={right_raw['mirror']})")

# ---- as shipped: each hand turned 180 degrees about its own axis, and every movement keeps its direction ---------
def facing(right, **kw):
    return show(right, True, yaw=180.0, **kw)


right_t, left_t = facing(True), facing(False)
print("TURNED right hand:", {k: (tuple(round(c, 2) for c in v) if isinstance(v, tuple) else v) for k, v in right_t.items()})
print("TURNED left hand: ", {k: (tuple(round(c, 2) for c in v) if isinstance(v, tuple) else v) for k, v in left_t.items()})
for name, shown_hand in (("right", right_t), ("left", left_t)):
    check(f"{name}_palm_toward_the_camera_now_shows_its_palm_to_the_front_view", shown_hand["palm_normal"][1] < -0.95,
          f"palm normal {tuple(round(c, 2) for c in shown_hand['palm_normal'])}")
    check(f"{name}_fingers_up_stay_up_after_the_turn", shown_hand["fingers"][2] > 0.95)
check("turned_right_hand_has_its_thumb_on_the_right_of_the_front_view_and_is_not_mirrored",
      right_t["thumb_x"] > 0.02 and right_t["mirror"] is False, f"thumb x {right_t['thumb_x']:.3f}, mirrored={right_t['mirror']}")
check("turned_left_hand_has_its_thumb_on_the_left_of_the_front_view_and_is_mirrored",
      left_t["thumb_x"] < -0.02 and left_t["mirror"] is True, f"thumb x {left_t['thumb_x']:.3f}, mirrored={left_t['mirror']}")

# tilt the REAL hand and watch which way Blender's fingers go: roll the fingers to your right (+X), push them
# forward (+Y, toward the webcam), tilt them back. The sign of the movement must be the same in Blender.
tilts = {"to_your_right": (mathutils.Matrix.Rotation(math.radians(30), 3, "Y"), 0, 1.0),
         "forward": (mathutils.Matrix.Rotation(math.radians(-30), 3, "X"), 1, 1.0),
         "to_your_left": (mathutils.Matrix.Rotation(math.radians(-30), 3, "Y"), 0, -1.0),
         "back": (mathutils.Matrix.Rotation(math.radians(30), 3, "X"), 1, -1.0)}
for right in (True, False):
    side = "right" if right else "left"
    neutral = (right_t if right else left_t)["fingers"]  # the same hand held up straight
    neutral_world = show(right, True, yaw=180.0, mode="world")["fingers"]
    for name, (tilt, axis, sign) in tilts.items():
        change = facing(right, tilt=tilt)["fingers"][axis] - neutral[axis]  # how far the fingers moved on that axis
        check(f"{side}_hand_tilted_{name}_goes_the_same_way_in_blender", sign * change > 0.4, f"fingers moved {change:+.2f} on {'xyz'[axis]}")
    sideways = show(right, True, yaw=180.0, tilt=tilts["to_your_right"][0], mode="world")["fingers"][0] - neutral_world[0]
    check(f"premise_{side}_turning_about_the_fixed_vertical_axis_reverses_sideways_tilts", sideways < -0.4, f"fingers moved {sideways:+.2f} on x")

saved = (br.LOST_HOLD_S, br.LOST_EASE_S, br.PORT)
br.LOST_HOLD_S, br.LOST_EASE_S = 0.5, 1.0
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 0))
sock.setblocking(False)
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
st = br.State()
st.rig = br.Rig(arm)
rest_tails = tails()
frames2 = frames_io.load_frames(TAKE, br.LANDMARK_NAMES)
frames2, _ = frames_io.emulate_sender(frames2)
seq = 0
for dt, label, score, lm in frames2[:60]:
    seq += 1
    tx.sendto(protocol.encode_frame("relax", seq, time.time(), label or "Right", lm), sock.getsockname())
    time.sleep(0.001)
    br.process(sock, st)
bent = tails()
tx.sendto(protocol.encode_lost("relax", seq + 1), sock.getsockname())
time.sleep(0.01)
br.process(sock, st)
held = tails()
check("hand_lost_holds_the_last_pose_first", worst(held, bent) < 1e-6)
st.lost_since -= 10.0  # as if the hand had been gone for 10 s
br.process(sock, st)
relaxed = tails()
# Finger shape compared in the armature's own axes (it keeps its last orientation). The rest pose is
# the bones' own rest tails; the posed rig at load is not it (the empties saved in the file pull it).
from mathutils import Vector
now_inv = arm.matrix_world.inverted()
rest_local = {n: Vector(arm.data.bones[n].tail_local) for n in rt.BONE_TARGETS}
dist_from_rest = lambda tips: {n: (now_inv @ Vector(tips[n]) - rest_local[n]).length for n in rt.BONE_TARGETS}
bent_gap = max(dist_from_rest(held).values())
per_bone = dist_from_rest(relaxed)
err = max(per_bone.values())
check("test_pose_really_differs_from_rest", bent_gap > 0.05, f"{bent_gap * 1000:.0f} milli-units")
check("lost_hand_relaxes_to_the_rest_pose", st.relaxed and err < 1e-4,
      f"relaxed={st.relaxed}; worst bone tip {err * 1000:.4f} milli-units from rest")
seq += 2
tx.sendto(protocol.encode_frame("relax", seq, time.time(), frames2[0][1] or "Right", frames2[60][3]), sock.getsockname())
time.sleep(0.01)
br.process(sock, st)
check("hand_found_again_resumes_tracking", st.hand_present is True and not st.relaxed and worst(tails(), relaxed) > 1e-3)
br.restore_armature(st)
sock.close()
tx.close()
br.LOST_HOLD_S, br.LOST_EASE_S, br.PORT = saved

# ---- panel ---------------------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    for name in ("blender_receiver.py", "blender_panel.py"):
        shutil.copy(os.path.join(V5_DIR, name), os.path.join(tmp, name))
    port = free_port()
    path = os.path.join(tmp, "blender_receiver.py")
    text = open(path, encoding="utf-8").read().replace("PORT = 5004 ", f"PORT = {port} ", 1)
    assert f"PORT = {port} " in text
    open(path, "w", encoding="utf-8").write(text)

    ns = {"__name__": "__main__", "__file__": path}
    exec(compile(text, path, "exec"), ns)  # what v5_run_receiver does
    dn = bpy.app.driver_namespace
    check("panel_registered_by_the_receiver", hasattr(bpy.types, "V5HAND_PT_panel")
          and hasattr(bpy.types, "V5HAND_OT_start") and hasattr(bpy.types.WindowManager, "v5_hand_mirror"))
    check("status_and_control_published", dn[br.STATUS_KEY]["running"] is True and "stop" in dn[br.CONTROL_KEY])

    wm = bpy.context.window_manager
    hands = dn[br.CONTROL_KEY]["hands"]  # with two hands in the file the choice belongs to hand 1
    own = lambda: dn[br.CONTROL_KEY]["states"][0].mirror_override if hands > 1 else ns["MIRROR_OVERRIDE"]
    wm.v5_hand_mirror = "MIRRORED"
    check("mirror_choice_reaches_the_receiver", own() is True)
    wm.v5_hand_mirror = "AUTO"
    check("mirror_choice_auto_clears_the_override", own() is None)

    class Layout:
        def __init__(self):
            self.lines = []

        def operator(self, idname, **kw):
            self.lines.append(("op", idname))

        def label(self, text="", **kw):
            self.lines.append(("label", text))

        def prop(self, owner, name, **kw):
            self.lines.append(("prop", name))

        def box(self):
            return self

    lay = Layout()
    bpy.types.V5HAND_PT_panel.draw(types.SimpleNamespace(layout=lay), types.SimpleNamespace(window_manager=wm))
    check("panel_draws_stop_button_and_status", ("op", "v5hand.stop") in lay.lines
          and any(k == "label" and ("Status:" in t or "Left hand:" in t) for k, t in lay.lines) and ("prop", "v5_hand_mirror") in lay.lines,
          str(lay.lines))

    result = bpy.ops.v5hand.stop()
    check("stop_button_stops_the_receiver", result == {"FINISHED"} and br.CONTROL_KEY not in dn
          and dn[br.STATUS_KEY]["running"] is False)
    lay = Layout()
    bpy.types.V5HAND_PT_panel.draw(types.SimpleNamespace(layout=lay), types.SimpleNamespace(window_manager=wm))
    check("panel_offers_start_when_stopped", ("op", "v5hand.start") in lay.lines)
    result = bpy.ops.v5hand.start()
    check("start_button_restarts_the_receiver", result == {"FINISHED"} and br.CONTROL_KEY in dn
          and dn[br.STATUS_KEY]["running"] is True and hasattr(bpy.types, "V5HAND_PT_panel"))
    dn[br.CONTROL_KEY]["stop"]()

    sys.modules["blender_panel"] = None
    ns2 = {"__name__": "v5_hand_panel", "__file__": os.path.join(tmp, "blender_panel.py"), "RECEIVER_FILE": path}
    exec(compile(open(os.path.join(tmp, "blender_panel.py"), encoding="utf-8").read(), "panel", "exec"), ns2)
    ns2["unregister"]()  # unregistering leaves nothing behind
    check("panel_unregisters_cleanly", not hasattr(bpy.types.WindowManager, "v5_hand_mirror")
          and not dn.get("v5_hand_panel_classes"))

print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
