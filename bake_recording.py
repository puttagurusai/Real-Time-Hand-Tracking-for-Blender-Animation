"""Bake a recorded hand take into keyframes on the rig, so it plays back in Blender without the
sender, the webcam or the receiver. Run headless; the result goes to a NEW .blend file:

    blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/bake_recording.py -- v5 TAKE.jsonl OUT.blend [options]

TAKE is a recording made with `sender.py --record` (.jsonl) or the old landmarks.json. Every
recorded frame goes through the real receiver (palm tracking, mirror decision, retargeting, the
joint limits), exactly as in a live session, and the result is keyed at the scene's frame rate.
Options (key=value):
    start=1        scene frame of the first recorded frame
    bones=0|1      1: also bake the bones themselves and remove the IK constraints in OUT, so the
                   take is plain bone animation (to export it, or move it to another file); 0 keeps
                   the live setup: animated empties + IK, which plays like a live session
    hands=all|1|2  which hands of the take to bake (default: every hand it has; hand 2 needs a
                   file made with build_two_hands.py)
    sender=1|0     emulate the sender's gate and filter (default: only for the old landmarks.json)
    overwrite=1    allow replacing an existing OUT (the file you opened is never replaced)
The file you ran it on is not changed.
"""
import os
import sys

V5_DIR = sys.argv[sys.argv.index("--") + 1]
sys.path.insert(0, V5_DIR)

import bpy

import blender_receiver as br
import frames_io
import rig_tools as rt


def scene_fps(scene):
    return scene.render.fps / scene.render.fps_base


def frame_plan(times, fps):
    """For frame n of the take (time n / fps) the index of the last recorded packet at or before
    that time. A take shorter than one frame still gets one."""
    if not times:
        return []
    count = int(times[-1] * fps) + 1
    plan, i = [], 0
    for n in range(count):
        t = n / fps
        while i + 1 < len(times) and times[i + 1] <= t + 1e-9:
            i += 1
        plan.append(i)
    return plan


def bake(arm, frames, start=1, fps=24.0, bake_bones=False):
    """One hand (slot 0): key its empties and armature object for every frame of the take (and
    the bones when bake_bones). frames: [(dt, label, score, landmarks)]. Returns the last frame."""
    return bake_hands({0: frames}, start, fps, bake_bones)


def bake_hands(takes, start=1, fps=24.0, bake_bones=False):
    """takes: {slot: frames}. Each hand goes through its own receiver state and is keyed on its own
    armature and empties (slot 1 is "Armature.001" and the ".001" empties). Returns the last frame."""
    states = br.make_states()
    for slot in takes:
        if slot not in states:
            raise ValueError(f"the recording has hand {slot + 1}, but the file has no rig for it "
                             "(build one with build_two_hands.py)")
    states = {slot: states[slot] for slot in takes}
    for slot, state in states.items():
        state.sid = "bake"
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    empties = {}
    for slot, state in states.items():
        arm = bpy.data.objects[state.armature_name]
        empties[slot] = [bpy.data.objects[n + state.suffix] for n in br.LANDMARK_NAMES
                         if n + state.suffix in bpy.data.objects]
        for ob in empties[slot] + [arm]:
            ob.animation_data_clear()
        arm.rotation_mode = "QUATERNION"  # no Euler flips between keys

    plans = {slot: frame_plan([f[0] for f in takes[slot]], fps) for slot in takes}
    count = max(len(p) for p in plans.values())
    applied = {slot: -1 for slot in takes}
    for n in range(count):
        frame = start + n
        for slot, state in states.items():
            plan = plans[slot]
            packet = plan[min(n, len(plan) - 1)]  # a shorter take holds its last pose
            for i in range(applied[slot] + 1, packet + 1):  # every packet up to now goes through the receiver
                _, label, score, lm = takes[slot][i]
                state.last_seq = i
                br._apply_frame(state, {"lm": lm, "hand": label, "score": score})
            applied[slot] = max(applied[slot], packet)
            arm = bpy.data.objects[state.armature_name]
            for ob in empties[slot]:
                ob.keyframe_insert("location", frame=frame)
            arm.keyframe_insert("location", frame=frame)
            arm.keyframe_insert("rotation_quaternion", frame=frame)
            arm.keyframe_insert("scale", frame=frame)
    last = start + count - 1
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = start, last
    if bake_bones:
        for state in states.values():
            bake_pose(bpy.data.objects[state.armature_name], start, last)
    return last


def bake_pose(arm, first, last):
    """Bake the evaluated pose (IK included) into bone keyframes and drop the constraints."""
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.nla.bake(frame_start=first, frame_end=last, only_selected=False, visual_keying=True,
                     clear_constraints=True, use_current_action=False, bake_types={"POSE", "OBJECT"})
    bpy.ops.object.mode_set(mode="OBJECT")


def main():
    args = sys.argv[sys.argv.index("--") + 2:]
    if len(args) < 2:
        sys.exit("usage: ... -- V5_DIR TAKE OUT.blend [start=1] [bones=0|1] [sender=1|0] [overwrite=1]")
    take, out = args[0], os.path.abspath(args[1])
    opts = dict(a.split("=", 1) for a in args[2:])
    if bpy.data.filepath and os.path.samefile(os.path.dirname(out), os.path.dirname(bpy.data.filepath)) \
            and os.path.basename(out) == os.path.basename(bpy.data.filepath):
        sys.exit("refusing to replace the file this run started from; choose another OUT name")
    if os.path.exists(out) and opts.get("overwrite") != "1":
        sys.exit(f"{out} already exists; choose another name or pass overwrite=1")

    wanted = frames_io.slots_in(take) if opts.get("hands", "all") == "all" else [int(opts["hands"]) - 1]
    takes = {}
    for slot in wanted:
        frames = frames_io.load_frames(take, br.LANDMARK_NAMES, slot)
        if frames and opts.get("sender", "0" if frames_io.is_recording(take) else "1") == "1":
            frames, rejected = frames_io.emulate_sender(frames)
        if frames:
            takes[slot] = frames
    if not takes:
        sys.exit(f"no hand frames in {take}")
    fps = scene_fps(bpy.context.scene)
    try:
        last = bake_hands(takes, start=int(opts.get("start", 1)), fps=fps, bake_bones=opts.get("bones") == "1")
    except ValueError as exc:
        sys.exit(str(exc))
    frames = [f for fr in takes.values() for f in fr]
    bpy.ops.wm.save_as_mainfile(filepath=out, copy=True)
    print(f"BAKE {len(takes)} hand(s), {len(frames)} recorded frames -> scene frames {opts.get('start', 1)}..{last} "
          f"at {fps:g} fps; bones baked: {opts.get('bones') == '1'}; saved a copy to {out}")


if __name__ == "__main__":
    main()
