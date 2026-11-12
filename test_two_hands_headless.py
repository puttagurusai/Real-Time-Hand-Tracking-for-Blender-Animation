"""Two hands on the real rig. Builds the second hand in memory (build_two_hands.py), then drives both
hands through the real receiver with real UDP: hand 1 gets the recording, hand 2 gets its mirror
image (the other hand doing the same motion). Nothing is saved; the .blend on disk is not changed.

    blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/test_two_hands_headless.py -- v5 [landmarks.json]
"""
import hashlib
import math
import os
import socket
import subprocess
import sys
import tempfile
import time

V5_DIR = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
ARGS = sys.argv[sys.argv.index("--") + 2:]
TAKE = os.path.abspath(ARGS[0]) if ARGS else os.path.join(os.path.dirname(V5_DIR), "landmarks.json")
sys.path.insert(0, V5_DIR)

import bpy
from mathutils import Vector

import blender_receiver as br
import build_two_hands as two
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


BLEND = bpy.data.filepath
BLEND_HASH = sha(BLEND)
arm0 = bpy.data.objects[rt.ARMATURE_NAME]
frames = frames_io.load_frames(TAKE, br.LANDMARK_NAMES)
frames, _ = frames_io.emulate_sender(frames)
# the wire format rounds landmarks to 5 decimals; the one-hand reference gets the same numbers
frames = [(dt, label, score, [[round(c, 5) for c in p] for p in lm]) for dt, label, score, lm in frames[:75]]
SAMPLE = {20, 40, 60, 74}
BONES = list(rt.BONE_TARGETS)


def local_tails(arm):
    """Bone tips in the armature's own axes, from the evaluated rig."""
    bpy.context.view_layer.update()
    ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    inv = ev.matrix_world.inverted()
    return {n: inv @ (ev.matrix_world @ ev.pose.bones[n].tail) for n in BONES}


def spread(a, b):
    return max((a[n] - b[n]).length for n in BONES)


def reach(arm, suffix):
    """Worst distance from a bone tip to the empty it is meant to reach, in milli-units."""
    bpy.context.view_layer.update()
    ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return max((ev.matrix_world @ ev.pose.bones[b].tail - bpy.data.objects[t + suffix].matrix_world.translation).length
               for b, t in rt.BONE_TARGETS.items()) * 1000


# ---- one hand alone, as the reference ---------------------------------------------------------------
ref_state = br.State()
ref_state.rig = br.Rig(arm0)
ref_state.sid = "ref"
ref = {}
for i, (_, label, score, lm) in enumerate(frames):
    ref_state.last_seq = i
    br._apply_frame(ref_state, {"lm": lm, "hand": label, "score": score})
    if i in SAMPLE:
        ref[i] = local_tails(arm0)
br.restore_armature(ref_state)

# ---- build the second hand ---------------------------------------------------------------------------
SPACING = 0.8  # the left-hand copy sits on the +X side of the right hand
arm1 = two.add_second_hand(SPACING)
check("second_armature_has_its_own_data", arm1.name == "Armature.001" and arm1.data is not arm0.data
      and len(arm1.data.bones) == len(arm0.data.bones))
targets1 = {c.target.name for pb in arm1.pose.bones for c in pb.constraints if getattr(c, "target", None)}
targets0 = {c.target.name for pb in arm0.pose.bones for c in pb.constraints if getattr(c, "target", None)}
check("second_hand_ik_points_at_its_own_empties",
      len(targets1) == 20 and all(n.endswith(".001") for n in targets1)
      and not any(n.endswith(".001") for n in targets0), f"{len(targets1)} targets")
kids1 = [o for o in arm1.children if o.type == "MESH"]
check("second_hand_has_its_own_skinned_mesh",
      len(kids1) == 1 and all(m.object in (arm1, None) for m in kids1[0].modifiers if m.type == "ARMATURE")
      and any(m.object is arm1 for m in kids1[0].modifiers if m.type == "ARMATURE")
      and kids1[0].data is not arm0.children[0].data, str([o.name for o in kids1]))
mesh0 = arm0.children[0]
check("second_mesh_is_shown_or_hidden_like_the_first",
      kids1[0].hide_get() == mesh0.hide_get() and kids1[0].hide_viewport == mesh0.hide_viewport
      and kids1[0].hide_render == mesh0.hide_render, f"first hidden={mesh0.hide_get()}, copy hidden={kids1[0].hide_get()}")
check("second_hand_sits_beside_the_first", (arm1.location - arm0.location - Vector((SPACING, 0, 0))).length < 1e-9
      and (bpy.data.objects["WRIST.001"].location - Vector((SPACING, 0, 0))).length < 1e-9)
refuse = False
try:
    two.add_second_hand(SPACING)
except ValueError:
    refuse = True
check("adding_a_third_hand_is_refused", refuse)

# ---- both hands through the receiver ---------------------------------------------------------------
states = br.make_states()
check("receiver_finds_both_rigs", sorted(states) == [0, 1] and states[1].armature_name == "Armature.001"
      and states[1].suffix == ".001" and states[0].tag == "[left hand]" and states[1].tag == "[right hand]")
check("second_origin_is_the_spacing", all(abs(a - b) < 1e-6 for a, b in zip(states[1].origin, (SPACING, 0.0, 0.0)))
      and states[0].origin == (0.0, 0.0, 0.0), str(states[1].origin))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 0))
sock.setblocking(False)
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = sock.getsockname()
gap0, gap1, reach0, reach1 = 0.0, 0.0, 0.0, 0.0
for i, (_, label, score, lm) in enumerate(frames):
    mirror_image = [[-x, y, z] for x, y, z in lm]
    # the same label on both hands: the rigs must decide their mirroring from the finger bends
    tx.sendto(protocol.encode_frame("two", i + 1, time.time(), "Right", lm, score=score), addr)
    tx.sendto(protocol.encode_frame("two", i + 1, time.time(), "Right", mirror_image, score=score, slot=1), addr)
    time.sleep(0.001)
    br.process(sock, states)
    if i in SAMPLE:
        gap0 = max(gap0, spread(local_tails(arm0), ref[i]))
        gap1 = max(gap1, spread(local_tails(arm1), ref[i]))
        reach0 = max(reach0, reach(arm0, ""))
        reach1 = max(reach1, reach(arm1, ".001"))
check("hand_1_unaffected_by_hand_2", gap0 < 1e-5, f"{gap0 * 1000:.4f} milli-units from the one-hand run")
check("hand_2_makes_the_same_motion_mirrored", gap1 < 1e-5, f"{gap1 * 1000:.4f} milli-units from the one-hand run")
print(f"DEBUG scale ref {ref_state.scale:.6f} hand1 {states[0].scale:.6f} hand2 {states[1].scale:.6f}; applied "
      f"{states[0].stats['applied']} {states[1].stats['applied']} of {len(frames)}; dropped "
      f"{states[0].stats['dropped']} {states[1].stats['dropped']}")
check("each_hand_reaches_its_own_empties", reach0 < 0.01 and reach1 < 0.01, f"{reach0:.5f} / {reach1:.5f} milli-units")
check("mirror_decided_separately_from_the_bends",
      states[0].shape_confirmed and states[1].shape_confirmed and states[0].mirror != states[1].mirror,
      f"hand 1 mirrored={states[0].mirror}, hand 2 mirrored={states[1].mirror}")
check("hand_2_wrist_empty_is_beside_hand_1", (bpy.data.objects["WRIST.001"].location
                                              - bpy.data.objects["WRIST"].location - Vector((SPACING, 0, 0))).length < 1e-6)

# one hand leaves: only that rig relaxes
tx.sendto(protocol.encode_lost("two", 200, slot=1), addr)
time.sleep(0.01)
saved = (br.LOST_HOLD_S, br.LOST_EASE_S)
br.LOST_HOLD_S, br.LOST_EASE_S = 0.5, 1.0
br.process(sock, states)
held0 = local_tails(arm0)
states[1].lost_since -= 10.0
br.process(sock, states)
rest = {n: Vector(arm1.data.bones[n].tail_local) for n in BONES}
now1 = local_tails(arm1)
relaxed_gap = max((now1[n] - rest[n]).length for n in BONES)
check("hand_2_lost_relaxes_only_hand_2", states[1].relaxed and not states[0].relaxed and states[0].hand_present
      and relaxed_gap < 1e-4 and spread(local_tails(arm0), held0) < 1e-9, f"{relaxed_gap * 1000:.4f} milli-units from rest")
br.LOST_HOLD_S, br.LOST_EASE_S = saved
for st in states.values():
    br.restore_armature(st)
sock.close()
tx.close()

# ---- recordings and baking keep the hands apart -------------------------------------------------------
import bake_recording as bake
import recording

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "two.jsonl")
    rec = recording.Recorder(path)
    for i, (_, label, score, lm) in enumerate(frames[:30]):
        rec.write(protocol.encode_frame("rec", i + 1, 1.0, "Right", lm))
        rec.write(protocol.encode_frame("rec", i + 1, 1.0, "Left", [[-x, y, z] for x, y, z in lm], slot=1))
    rec.close()
    check("recording_has_both_slots", frames_io.slots_in(path) == [0, 1])
    h0 = frames_io.load_frames(path, br.LANDMARK_NAMES, 0)
    h1 = frames_io.load_frames(path, br.LANDMARK_NAMES, 1)
    check("loader_returns_one_hand_at_a_time", len(h0) == len(h1) == 30 and h0[0][1] == "Right" and h1[0][1] == "Left"
          and h1[0][3][8][0] == -h0[0][3][8][0])

takes = {0: [(i / 30.0, label, score, lm) for i, (_, label, score, lm) in enumerate(frames)],
         1: [(i / 30.0, label, score, [[-x, y, z] for x, y, z in lm]) for i, (_, label, score, lm) in enumerate(frames)]}
last = bake.bake_hands(takes, start=1, fps=24.0)
check("both_hands_keyed", all(bpy.data.objects[n].animation_data and bpy.data.objects[n].animation_data.action
                              for n in ("WRIST", "WRIST.001", "INDEX_FINGER_TIP", "INDEX_FINGER_TIP.001"))
      and arm0.animation_data.action and arm1.animation_data.action, f"scene frames 1..{last}")
worst_gap, worst_reach = 0.0, 0.0
for frame in (10, 30, 50, last):
    bpy.context.scene.frame_set(frame)
    worst_gap = max(worst_gap, spread(local_tails(arm0), local_tails(arm1)))
    worst_reach = max(worst_reach, reach(arm0, ""), reach(arm1, ".001"))
check("baked_hands_are_mirror_images", worst_gap < 1e-4, f"{worst_gap * 1000:.4f} milli-units")
check("baked_hands_reach_their_empties", worst_reach < 0.01, f"{worst_reach:.5f} milli-units")
refused = False
try:
    bake.bake_hands({0: takes[0], 2: takes[1]})
except ValueError:
    refused = True
check("bake_refuses_a_hand_without_a_rig", refused)

# ---- control and status through the panel's entry points ------------------------------------------------
port = br.PORT
br.PORT = free_port()
br.start()
control = bpy.app.driver_namespace[br.CONTROL_KEY]
status = bpy.app.driver_namespace[br.STATUS_KEY]
check("status_lists_both_hands", control["hands"] == 2 and [h["slot"] for h in status["hands"]] == [0, 1])
live = control["states"]
control["set_mirror"](True, 1)
check("mirror_choice_per_hand", live[1].mirror_override is True and live[0].mirror_override is None
      and br.MIRROR_OVERRIDE is None)
control["set_mirror"](False)  # the default for every hand without a choice of its own
check("mirror_default_leaves_a_hands_own_choice", br.MIRROR_OVERRIDE is False and live[1].mirror_override is True
      and br.status_of(live[0])["override"] is False and br.status_of(live[1])["override"] is True)
control["set_mirror"](None)
control["set_mirror"](None, 1)
check("mirror_choices_clear", br.MIRROR_OVERRIDE is None and live[1].mirror_override is None)
import types


class Layout:
    def __init__(self):
        self.lines = []

    def operator(self, idname, **kw):
        self.lines.append(("op", idname))

    def label(self, text="", **kw):
        self.lines.append(("label", text))

    def prop(self, owner, name, **kw):
        self.lines.append(("prop", name, kw.get("text")))

    def box(self):
        return self


wm = bpy.context.window_manager
lay = Layout()
bpy.types.V5HAND_PT_panel.draw(types.SimpleNamespace(layout=lay), types.SimpleNamespace(window_manager=wm))
texts = [l[1] for l in lay.lines if l[0] == "label"]
check("panel_shows_each_hand", any(t.startswith("Left hand:") for t in texts) and any(t.startswith("Right hand:") for t in texts)
      and ("prop", "v5_hand_mirror", "Left hand mirror") in lay.lines and ("prop", "v5_hand_mirror2", "Right hand mirror") in lay.lines,
      str(texts))
wm.v5_hand_mirror2 = "MIRRORED"
check("panel_hand_2_choice_reaches_hand_2", live[1].mirror_override is True and live[0].mirror_override is None)
wm.v5_hand_mirror = "NOT_MIRRORED"
check("panel_hand_1_choice_reaches_hand_1_only", live[0].mirror_override is False and live[1].mirror_override is True
      and br.MIRROR_OVERRIDE is None)
wm.v5_hand_mirror = "AUTO"
wm.v5_hand_mirror2 = "AUTO"
check("panel_auto_clears_each_hand", live[0].mirror_override is None and live[1].mirror_override is None)
control["stop"]()
br.PORT = port

# ---- the build command never replaces a file ------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    script = os.path.join(V5_DIR, "build_two_hands.py")

    def run(out, *extra, source=BLEND):
        return subprocess.run([bpy.app.binary_path, "-b", "--disable-autoexec", source, "--python", script, "--",
                               V5_DIR, out, *extra], capture_output=True, text=True, timeout=600)

    out = os.path.join(tmp, "two.blend")
    first = run(out)
    check("build_cli_writes_a_new_file", first.returncode == 0 and os.path.exists(out), first.stdout[-160:])
    stamp = sha(out)
    second = run(out)
    check("build_cli_refuses_to_replace_a_file", second.returncode != 0 and "already exists" in second.stdout + second.stderr
          and sha(out) == stamp)
    same = run(BLEND, "overwrite=1")
    check("build_cli_never_replaces_its_own_source", same.returncode != 0 and sha(BLEND) == BLEND_HASH)
    again = run(os.path.join(tmp, "three.blend"), source=out)
    check("build_cli_refuses_a_file_that_already_has_two_hands", again.returncode != 0
          and not os.path.exists(os.path.join(tmp, "three.blend")))
    probe = subprocess.run([bpy.app.binary_path, "-b", "--disable-autoexec", out, "--python-expr",
                            "import bpy; a=bpy.data.objects; m=[o for o in a if o.type=='MESH']; print('PROBE', 'Armature' in a, 'Armature.001' in a, "
                            "'WRIST.001' in a, sum(len(pb.constraints) for pb in a['Armature.001'].pose.bones)); "
                            "print('HIDDEN', a['Armature'].hide_get(), a['Armature.001'].hide_get(), "
                            "sorted(o.hide_get() for o in m))"],
                           capture_output=True, text=True, timeout=600)
    check("saved_file_reopens_with_two_hands", "PROBE True True True 20" in probe.stdout, probe.stdout[-120:])
    expect = f"HIDDEN {arm0.hide_get()} {arm0.hide_get()} {sorted([mesh0.hide_get()] * 2)}"
    check("saved_file_keeps_what_is_shown_and_hidden", expect in probe.stdout,
          f"wanted '{expect}', got '{probe.stdout[probe.stdout.find('HIDDEN'):][:60].strip()}'")

check("source_file_unchanged", sha(BLEND) == BLEND_HASH)
print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
