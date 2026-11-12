"""Check the saved rig copy as a file. Run headless on the COPY:

    blender -b --disable-autoexec v5/untitled_rig_fixed.blend --python v5/verify_copy_headless.py -- v5
"""
import math
import sys

V5_DIR = sys.argv[sys.argv.index("--") + 1]
sys.path.insert(0, V5_DIR)

import bpy

import rig_tools as rt

failures = []


def check(name, ok, detail=""):
    print(f"RESULT {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


arm = bpy.data.objects[rt.ARMATURE_NAME]
report = rt.validate()
for level, msg in report:
    print(f"VALIDATE [{level}] {msg}")
check("validate_all_ok", all(lv == "OK" for lv, _ in report))
check("no_subtargets", all(not c.subtarget for b in rt.BONE_TARGETS
                           for c in arm.pose.bones[b].constraints if c.type == "IK"))
check("rolls_already_aligned", rt.align_finger_rolls(arm, apply=False) == [])

limited = []
for names in rt.FINGERS.values():
    for b in names[1:]:
        pb = arm.pose.bones[b]
        if pb.lock_ik_y and sum((pb.use_ik_limit_x, pb.use_ik_limit_y, pb.use_ik_limit_z)) == 2:
            limited.append(b)
check("12_finger_bones_have_limits", len(limited) == 12, f"{len(limited)}/12")
check("root_and_thumb_bones_unlimited", not any(
    arm.pose.bones[b].lock_ik_y or arm.pose.bones[b].use_ik_limit_x
    for b in rt.BONE_TARGETS if b.startswith("wrist_to_") or b.startswith("thumb_")))

texts = {t.name for t in bpy.data.texts}
print("TEXTS", sorted(texts))
check("old_conflicting_texts_removed",
      not texts & {"IK TESTING", "edit mode", "scale", "small", "Text", "Text.001"})
check("loaders_present", {"v5_run_receiver", "v5_rig_tools"} <= texts)
check("setup_scripts_kept", {"emptyness creation", "IK CONSTRAIN"} <= texts)

# the loader finds blender_receiver.py next to the .blend and starts and stops cleanly
import io
import contextlib
# Run the loader with a free port, so a receiver already running in another Blender (which
# holds 5004) cannot make this check fail. The file the loader reads is patched in memory only.
import socket
probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
probe.bind(("127.0.0.1", 0))
free_port = probe.getsockname()[1]
probe.close()
real_open = open
receiver_path = bpy.path.abspath("//blender_receiver.py")


def patched_open(path, *a, **kw):
    if str(path) == receiver_path:
        return io.StringIO(real_open(path, encoding="utf-8").read().replace("PORT = 5004", f"PORT = {free_port}", 1))
    return real_open(path, *a, **kw)


import builtins
builtins.open = patched_open
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(bpy.data.texts["v5_run_receiver"].as_string(), "<loader>", "exec"), {})
finally:
    builtins.open = real_open
out = buf.getvalue()
print("LOADER OUTPUT:", out.strip())
check("receiver_loader_starts", "listening on" in out and "palm tracking ON" in out, out.strip()[:120])
stop = bpy.app.driver_namespace.get("hand_receiver_v5_stop")
check("receiver_registered_stop", callable(stop))
if stop:
    stop()
check("receiver_stop_cleans_up", "hand_receiver_v5_stop" not in bpy.app.driver_namespace)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(bpy.data.texts["v5_rig_tools"].as_string(), "<loader>", "exec"), {})
check("rig_tools_loader_reports_no_changes_needed", "[ERROR]" not in buf.getvalue()
      and "[WARN]" not in buf.getvalue() and "would fix" not in buf.getvalue(), "")

print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
