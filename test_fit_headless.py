"""Does the fitted skeleton fit the mesh? Runs the whole fit (fit_skeleton.py) in memory on the file it is
given and checks the result. Nothing is saved; the .blend on disk is not changed.

    blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/test_fit_headless.py -- v5 [python=python]

It needs MediaPipe in the interpreter named by python= (the one you run sender.py with), because the
fit uses the same hand model as the tracker. Run it on the file with the ORIGINAL skeleton: the first
checks show that skeleton does not fit, so on an already fitted file they would (rightly) fail.
"""
import hashlib
import math
import os
import subprocess
import sys

import numpy as np

V5_DIR = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
OPTS = dict(a.split("=", 1) for a in sys.argv[sys.argv.index("--") + 2:] if "=" in a)
sys.path.insert(0, V5_DIR)

import bpy

import blender_receiver as br
import fit_skeleton as fit
import rig_tools as rt

failures = []


def check(name, ok, detail=""):
    print(f"RESULT {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


BLEND = bpy.data.filepath
BLEND_HASH = sha(BLEND)
arm = bpy.data.objects[rt.ARMATURE_NAME]
mesh = fit.hand_mesh(arm)
scene = bpy.context.scene
settings_before = (scene.render.engine, scene.render.resolution_x, scene.world, scene.camera,
                   mesh.hide_get(), len(bpy.data.objects), len(bpy.data.cameras), len(bpy.data.worlds))


def joints_of(armature):
    """Landmark -> position from the bones' own heads and tails (armature axes)."""
    joints = {}
    for name, (head, tail) in fit.BONES.items():
        bone = armature.data.bones[name]
        joints[head] = tuple(bone.head_local)
        joints[tail] = tuple(bone.tail_local)
    return joints


FINGERS = ("index", "middle", "ring", "pinky")

# ---- the skeleton as it comes ---------------------------------------------------------------------------
before = joints_of(arm)
shares_before = fit.phalanx_ratios(before)
inside_before = fit.inside_mesh(mesh, [before[k] for k in range(21)])
print("BEFORE bone shares " + ", ".join(f"{f} {tuple(round(v, 2) for v in shares_before[f])}" for f in FINGERS)
      + f"; joints outside the mesh {[k for k, ok in enumerate(inside_before) if not ok]}")

# ---- fit it ---------------------------------------------------------------------------------------------
import tempfile
folder = tempfile.mkdtemp(prefix="v5_fit_test_")
normal, up, lateral = fit.palm_axes(arm)
views = fit.render_views(arm, mesh, fit.make_views(normal, up, lateral), folder)
check("renders_leave_the_scene_as_it_was", (scene.render.engine, scene.render.resolution_x, scene.world, scene.camera,
                                            mesh.hide_get(), len(bpy.data.objects), len(bpy.data.cameras),
                                            len(bpy.data.worlds)) == settings_before)
joints, report, used = fit.fit_joints(arm, mesh, views, OPTS.get("python", "python"))
check("hand_found_in_most_views", len(used) >= 6, f"{len(used)} of {len(views)}: {used}")
check("views_agree_with_each_other", max(r[1] for r in report) < 0.015 and min(r[2] for r in report) >= 4,
      f"worst {max(r[1] for r in report):.4f} of the picture width, fewest views used {min(r[2] for r in report)}")
fit.centre_joints(joints, fit.rest_vertices(mesh, arm), normal)
shift = [math.dist(before[k], joints[k]) for k in range(21)]
print(f"SHIFT the original joints are {sum(shift) / 21:.3f} from the fitted ones on average, {max(shift):.3f} at most "
      f"(a finger is about 0.24 long and 0.04 thick)")
check("premise_the_original_joints_are_misplaced", sum(shift) / 21 > 0.02 and max(shift) > 0.1 and not inside_before[0],
      f"mean {sum(shift) / 21:.3f}, max {max(shift):.3f}, original wrist inside the mesh: {inside_before[0]}")
inside = fit.inside_mesh(mesh, [joints[k] for k in range(21)])
check("every_joint_is_inside_the_mesh", all(inside), f"outside: {[k for k, ok in enumerate(inside) if not ok]}")
vertices = fit.rest_vertices(mesh, arm)
off_centre = fit.centre_joints(dict(joints), vertices, normal, iterations=1)  # how far another pass would move each joint
print("CENTRE another pass would move:", {k: round(v, 4) for k, v in off_centre.items()})
check("joints_are_centred_in_their_fingers", max(off_centre.values()) < 0.01,
      f"another centring pass would move a joint at most {max(off_centre.values()):.4f}")
shares = fit.phalanx_ratios(joints)
print("AFTER bone shares " + ", ".join(f"{f} {tuple(round(v, 2) for v in shares[f])}" for f in (*FINGERS, "thumb")))
good = all(0.36 <= shares[f][0] <= 0.52 and 0.22 <= shares[f][1] <= 0.34 and 0.2 <= shares[f][2] <= 0.36 for f in FINGERS)
check("finger_bone_lengths_are_anatomical", good, "proximal 0.36-0.52, middle 0.22-0.34, distal 0.20-0.36 of the finger")
check("thumb_bone_lengths_are_anatomical", all(0.25 <= v <= 0.42 for v in shares["thumb"]), str(tuple(round(v, 2) for v in shares["thumb"])))
lengths = {f: math.dist(joints[m], joints[t]) for f, (m, t) in {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}.items()}
check("finger_lengths_are_in_the_usual_order", lengths["middle"] > lengths["index"] > lengths["pinky"] and lengths["middle"] > lengths["ring"] > lengths["pinky"],
      str({f: round(v, 3) for f, v in lengths.items()}))

joints = fit.move_origin_to_wrist(arm, mesh, joints)
fit.apply_joints(arm, joints)
fit.finish_rig(arm, mesh, joints)

# ---- the rig that came out -------------------------------------------------------------------------------
bones = arm.data.bones
palm_heads = [tuple(bones[n].head_local) for n in rt.ROOT_BONES]
check("wrist_is_the_armature_origin", max(math.sqrt(sum(c * c for c in h)) for h in palm_heads) < 1e-6,
      f"{max(math.sqrt(sum(c * c for c in h)) for h in palm_heads):.2e}")
joined = all(math.dist(bones[name].head_local, bones[bones[name].parent.name].tail_local) < 1e-6
             for name in fit.BONES if bones[name].parent is not None)
check("bones_start_where_their_parents_end", joined)
check("every_bone_keeps_its_ik_constraint_and_target", all(
    [c.target.name for c in arm.pose.bones[name].constraints if c.type == "IK"] == [rt.BONE_TARGETS[name]]
    for name in fit.BONES))
errors = [msg for level, msg in rt.validate() if level == "ERROR"]
check("rig_tools_finds_no_errors", not errors, str(errors))
plan, notes = rt.plan_joint_limits(arm)
check("ik_limits_planned_for_all_twelve_finger_bones", len(plan) == 12, f"{len(plan)}")
rig = br.Rig(arm)
check("receiver_reads_the_new_palm_length", abs(rig.span - math.dist(joints[0], joints[9])) < 1e-6 and rig.collision.palm is not None
      and rig.bend_sign in (1.0, -1.0), f"span {rig.span:.4f}, bend sign {rig.bend_sign}")
bpy.context.view_layer.update()
ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
worst = max((ev.matrix_world @ ev.pose.bones[n].tail - bpy.data.objects[t].matrix_world.translation).length
            for n, t in rt.BONE_TARGETS.items()) * 1000
check("ik_reaches_the_empties_at_rest", worst < 0.5, f"{worst:.3f} milli-units")

# the skin follows the bones and the mesh has not moved relative to them
mods = [m for m in mesh.modifiers if m.type == "ARMATURE"]
check("mesh_is_skinned_to_the_new_bones", len(mods) == 1 and mods[0].object is arm
      and sorted(g.name for g in mesh.vertex_groups) == sorted(fit.BONES)
      and all(bones[n].get("v5_radius") for n in fit.BONES if not n.startswith("wrist_to_")))

# ---- the command never replaces a file ------------------------------------------------------------------------
import tempfile as _t
script = os.path.join(V5_DIR, "fit_skeleton.py")


def run(out, *extra):
    return subprocess.run([bpy.app.binary_path, "-b", "--disable-autoexec", BLEND, "--python", script, "--", V5_DIR, out, *extra],
                          capture_output=True, text=True, timeout=900)


with _t.TemporaryDirectory() as tmp:
    existing = os.path.join(tmp, "existing.blend")
    open(existing, "wb").write(b"not a blend")
    refused = run(existing)
    check("fit_cli_refuses_to_replace_a_file", refused.returncode != 0 and "already exists" in refused.stdout + refused.stderr
          and open(existing, "rb").read() == b"not a blend")
    own = run(BLEND, "overwrite=1")
    check("fit_cli_never_replaces_its_own_source", own.returncode != 0 and sha(BLEND) == BLEND_HASH)

check("source_file_unchanged", sha(BLEND) == BLEND_HASH)
print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
