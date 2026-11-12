"""Feed a recording through the real receiver and the real rig, and report how well the rig's
fingers follow the live finger bends. Run headless on the fixed copy:

    blender -b --disable-autoexec v5/untitled_rig_fixed.blend --python v5/analyze_recording_headless.py -- v5 FILE [options]

FILE is either a recording made with `sender.py --record` (.jsonl) or the old `landmarks.json`
(a list of per-frame normalised landmarks, assumed 640x480 and captured from a flipped image).
Options (key=value): label=Left|Right|none  hand label attached to every frame (jsonl uses its own)
                     mirror=auto|true|false  receiver MIRROR_OVERRIDE
                     track=1|0  TRACK_PALM    retarget=1|0  RETARGET    clamp=1|0  CLAMP_TO_LIMITS
                     collision=0|1  COLLISION (default 0: this tool scores how well the rig follows
                                    your fingers, and collision moves fingers on purpose where they
                                    would overlap; test_collision_headless.py scores collision)
                     slot=0|1  which hand of a two-hand recording (rig: Armature / Armature.001)
                     sender=1|0  emulate the sender's gate and filter (default: on for landmarks.json,
                                 off for a .jsonl recording, which the sender already filtered)
A frame is "stuck" when a live finger joint is bent more than 40 degrees but the same rig joint
is bent less than 15 degrees: the rig finger is not following the curl. Bend error is measured
against the live bend after the receiver's straight dead zone (STRAIGHT_ZONE_DEG). A joint whose
live bend is under 10 degrees is "straight": the rig's bend there must stay small.
Nothing is saved.
"""
import json
import math
import sys
import time

V5_DIR = sys.argv[sys.argv.index("--") + 1]
sys.path.insert(0, V5_DIR)

import bpy

import blender_receiver as br
import frames_io
import rig_tools as rt

args = sys.argv[sys.argv.index("--") + 2:]
path = args[0]
opts = dict(a.split("=", 1) for a in args[1:])


def unit_angle(a, b):
    d = br._dot(a, b) / (math.sqrt(br._dot(a, a)) * math.sqrt(br._dot(b, b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def bends(chain_points):
    """MCP, PIP, DIP bend (degrees between consecutive segments) of one chain of 5 points."""
    d = [br._sub(chain_points[s + 1], chain_points[s]) for s in range(4)]
    return [unit_angle(d[s], d[s + 1]) for s in range(3)]


SLOT = int(opts.get("slot", 0))
state = br.make_states()[SLOT]  # the rig of the chosen hand: Armature, or Armature.001 for hand 2
arm = bpy.data.objects[state.armature_name]
if opts.get("track"):
    br.TRACK_PALM = opts["track"] == "1"
if opts.get("retarget"):
    br.RETARGET = opts["retarget"] == "1"
br.COLLISION = opts.get("collision", "0") == "1"
if opts.get("clamp"):
    br.CLAMP_TO_LIMITS = opts["clamp"] == "1"
if opts.get("mirror") in ("true", "false"):
    br.MIRROR_OVERRIDE = opts["mirror"] == "true"

if "margin" in opts:  # diagnostic: out-of-plane swing allowed at PIP/DIP, in degrees
    br.SIDE_MARGIN_DEG = float(opts["margin"])
if opts.get("iklimits") == "0":  # diagnostic: take the rig's IK limits away (in memory)
    rt.clear_joint_limits(arm)
if "slack" in opts or "sideslack" in opts:  # diagnostic: re-plan the IK limits with other slack
    rt.IK_SLACK_DEG = float(opts.get("slack", rt.IK_SLACK_DEG))
    rt.IK_SIDE_SLACK_DEG = float(opts.get("sideslack", rt.IK_SIDE_SLACK_DEG))
    rt.clear_joint_limits(arm)
    rt.apply_joint_limits(arm, rt.plan_joint_limits(arm)[0])

loaded = frames_io.load_frames(path, br.LANDMARK_NAMES, SLOT)
if opts.get("reflect") == "1":  # the mirror image of the recording: the other hand, same motion
    loaded = [(dt, label, score, [[-p[0], p[1], p[2]] for p in lm]) for dt, label, score, lm in loaded]
forced_label = opts.get("label", "none")
# A .jsonl recording was gated and filtered by the sender already; the old landmarks.json is raw
emulate = opts.get("sender", "0" if frames_io.is_recording(path) else "1") == "1"
filtered, rejected = frames_io.emulate_sender(loaded) if emulate else (loaded, 0)
sender_gate = emulate or None
raw_frames = [(label, score, lm) for _, label, score, lm in loaded]
frames = [(label, score, lm) for _, label, score, lm in filtered]
state.sid = "analyze"
fingers = ("index", "middle", "ring", "pinky")
errors = {j: [] for j in ("mcp", "pip", "dip")}
straight = {"pip": [], "dip": []}  # (live bend, rig bend) where the live finger is nearly straight
STRAIGHT_LIVE_DEG = 10.0
stuck = total_curled = 0
mirror_log = []
for i, ((label, score, lm), (_, _, raw_lm)) in enumerate(zip(frames, raw_frames)):
    if label is None and forced_label != "none":
        label = forced_label
    state.last_seq = i
    br._apply_frame(state, {"lm": lm, "hand": label, "score": score, "t": time.time()})
    if state.scale is None:
        continue
    bpy.context.view_layer.update()
    ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mirror_log.append(state.mirror)
    if i < br.CALIBRATION_FRAMES:
        continue
    for finger in fingers:
        live = bends([tuple(raw_lm[k]) for k in br.CHAIN_LANDMARKS[finger]])  # the real finger, unfiltered
        names = br.BONE_CHAINS[finger]
        d = [tuple(ev.pose.bones[n].tail - ev.pose.bones[n].head) for n in names]
        rig = [unit_angle(d[s], d[s + 1]) for s in range(3)]
        for j, name in enumerate(("mcp", "pip", "dip")):
            # what the rig should show: the live bend with the receiver's straight dead zone applied
            expected = live[j]
            if name in br.STRAIGHT_ZONE_DEG:
                expected = math.degrees(br._straighten(math.radians(live[j]), br.STRAIGHT_ZONE_DEG[name]))
            errors[name].append(abs(expected - rig[j]))
            if name != "mcp" and live[j] > 40:
                total_curled += 1
                stuck += rig[j] < 15
            if name != "mcp" and live[j] < STRAIGHT_LIVE_DEG:
                straight[name].append((live[j], rig[j]))

def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")

print(f"ANALYZE file={path} frames={len(frames)} options={opts or 'defaults'}"
      f"{f' (sender gate+filter emulated, gate rejected {rejected})' if sender_gate is not None else ''}")
print(f"ANALYZE rig mirrored over time: {sorted({str(m) for m in mirror_log})}"
      f" (final {state.mirror})")
for j in ("mcp", "pip", "dip"):
    v = errors[j]
    print(f"ANALYZE {j}: bend error mean {sum(v) / len(v):5.1f} deg  p90 {pct(v, 0.9):5.1f}  max {max(v):5.1f}")
print(f"ANALYZE stuck joints (live bent >40 deg, rig <15 deg): {stuck} of {total_curled} curled joints "
      f"= {100.0 * stuck / max(total_curled, 1):.1f}%")
for j, pairs in straight.items():
    if pairs:
        rig_bends = [r for _, r in pairs]
        print(f"ANALYZE straight {j}: {len(pairs)} joints with live bend < {STRAIGHT_LIVE_DEG:.0f} deg: "
              f"live mean {sum(l for l, _ in pairs) / len(pairs):4.1f}, rig mean {sum(rig_bends) / len(pairs):4.1f} "
              f"p90 {pct(rig_bends, 0.9):4.1f} max {max(rig_bends):4.1f} deg")
if opts.get("check") == "1":
    bad = []
    for name, ok, detail in (
            ("curled_joints_never_stuck", stuck == 0, f"{stuck} stuck"),
            ("mcp_bend_error_under_4_deg", sum(errors["mcp"]) / len(errors["mcp"]) < 4.0, ""),
            ("pip_bend_error_under_4_deg", sum(errors["pip"]) / len(errors["pip"]) < 4.0, ""),
            ("dip_bend_error_under_4_deg", sum(errors["dip"]) / len(errors["dip"]) < 4.0, ""),
            ("pip_p90_under_12_deg", pct(errors["pip"], 0.9) < 12.0, f"{pct(errors['pip'], 0.9):.1f}"),
            ("straight_pip_and_dip_stay_under_8_deg",
             all(not v or max(r for _, r in v) < 8.0 for v in straight.values()),
             ", ".join(f"{j} max {max((r for _, r in v), default=0):.1f}" for j, v in straight.items())),
            ("straight_joints_seen", all(len(v) >= 10 for v in straight.values()),
             ", ".join(f"{j} {len(v)}" for j, v in straight.items())),
            ("shape_decided_from_bends", state.shape_confirmed, "")):
        print(f"RESULT {'PASS' if ok else 'FAIL'} {name} {detail}")
        if not ok:
            bad.append(name)
    print("FAILURES:", bad or "none")
    sys.exit(1 if bad else 0)
