"""Does the hand mesh follow the skeleton? Run headless on a fixed copy (nothing is saved; the .blend
on disk is not changed):

    blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/test_skin_headless.py -- v5 [landmarks.json]

The mesh is posed by the real receiver from the real recording and measured three ways:
  tip follow   how far each fingertip's skin ends up from its bone tip (in armature units)
  edge growth  how much longer any mesh edge gets than at rest: a tear or a stretched ribbon
  volume       the mesh volume against rest: pinching and shrinking at bent joints
The weights in the file as it comes are measured first: they must fail (so this test shows the
problem), then the same measurements run after skin_mesh re-skins the mesh.
"""
import hashlib
import os
import subprocess
import sys
import tempfile

V5_DIR = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
ARGS = sys.argv[sys.argv.index("--") + 2:]
TAKE = os.path.abspath(ARGS[0]) if ARGS else os.path.join(os.path.dirname(V5_DIR), "landmarks.json")
sys.path.insert(0, V5_DIR)

import bpy
import numpy as np

import blender_receiver as br
import build_two_hands as two
import frames_io
import rig_tools as rt
import skin_mesh

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
mesh = [c for c in arm.children if c.type == "MESH"][0]
TIPS = {"thumb": "thumb_ip_to_thumb_tip", "index": "index_dip_to_index_tip", "middle": "middle_dip_to_middle_tip",
        "ring": "ring_dip_to_ring_tip", "pinky": "pinky_dip_to_pinky_tip"}

frames = frames_io.load_frames(TAKE, br.LANDMARK_NAMES)
frames, _ = frames_io.emulate_sender(frames)
SAMPLE = list(range(34, len(frames), 6))

n_v = len(mesh.data.vertices)
co = np.empty(n_v * 3, np.float32)
mesh.data.vertices.foreach_get("co", co)
REST = co.reshape(-1, 3)
edge_ids = np.empty(len(mesh.data.edges) * 2, np.int32)
mesh.data.edges.foreach_get("vertices", edge_ids)
EDGES = edge_ids.reshape(-1, 2)
REST_LEN = np.linalg.norm(REST[EDGES[:, 0]] - REST[EDGES[:, 1]], axis=1)
mesh.data.calc_loop_triangles()
tri = np.empty(len(mesh.data.loop_triangles) * 3, np.int32)
mesh.data.loop_triangles.foreach_get("vertices", tri)
TRI = tri.reshape(-1, 3)
SCALE = abs(mesh.scale[0])


def volume(p):
    a, b, c = (p[TRI[:, k]].astype(np.float64) for k in range(3))
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


REST_VOL = volume(REST)


def evaluated(ob):
    bpy.context.view_layer.update()
    return ob.evaluated_get(bpy.context.evaluated_depsgraph_get())


def posed_points(ob):
    ev = evaluated(ob)
    me = ev.to_mesh()
    out = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", out)
    ev.to_mesh_clear()
    return out.reshape(-1, 3)


def to_armature_axes(points, mesh_ob, arm_ob):
    ev_mesh, ev_arm = evaluated(mesh_ob), evaluated(arm_ob)
    m = np.array(ev_arm.matrix_world.inverted() @ ev_mesh.matrix_world)
    return points.astype(np.float64) @ m[:3, :3].T + m[:3, 3]


def tip_vertices(arm_ob, mesh_ob):
    """For each fingertip, the mesh vertex closest to its bone's tail at rest (armature units)."""
    rest_arm = to_armature_axes(REST, mesh_ob, arm_ob)
    return {f: int(np.argmin(np.linalg.norm(rest_arm - np.array(arm_ob.data.bones[b].tail_local), axis=1)))
            for f, b in TIPS.items()}, rest_arm


def measure(arm_ob, mesh_ob, state_factory=None):
    """Pose the hand through the recording; returns the worst fingertip-follow error per finger,
    the worst edge growth, edges grown more than 0.01, and the lowest volume ratio."""
    tips, rest_arm = tip_vertices(arm_ob, mesh_ob)
    state = br.make_states()[0]
    state.sid = "skin"
    follow = {f: 0.0 for f in TIPS}
    worst, over, vmin = 0.0, 0, 9.0
    for i, (_, label, score, lm) in enumerate(frames):
        state.last_seq = i
        br._apply_frame(state, {"lm": lm, "hand": label, "score": score})
        if i not in SAMPLE:
            continue
        p = posed_points(mesh_ob)
        posed_arm = to_armature_axes(p, mesh_ob, arm_ob)
        ev_arm = evaluated(arm_ob)
        for f, b in TIPS.items():
            bone = ev_arm.pose.bones[b]
            rest_tail = np.array(arm_ob.data.bones[b].tail_local)
            moved_bone = np.array(bone.tail) - rest_tail
            moved_skin = posed_arm[tips[f]] - rest_arm[tips[f]]
            follow[f] = max(follow[f], float(np.linalg.norm(moved_skin - moved_bone)))
        grow = (np.linalg.norm(p[EDGES[:, 0]] - p[EDGES[:, 1]], axis=1) - REST_LEN) * SCALE
        worst = max(worst, float(grow.max()))
        over += int((grow > 0.01).sum())
        vmin = min(vmin, volume(p) / REST_VOL)
    br.restore_armature(state)
    return follow, worst, over, vmin


# ---- the file as it comes ---------------------------------------------------------------------------
bones = {b.name for b in arm.data.bones}
stale = [g.name for g in mesh.vertex_groups if g.name not in bones]
sample = range(0, n_v, 7)
unweighted = sum(1 for i in sample if not any(mesh.vertex_groups[g.group].name in bones for g in mesh.data.vertices[i].groups))
before = measure(arm, mesh)
print(f"BEFORE stale groups {stale}; unweighted {unweighted * 7} of {n_v}; tip follow error "
      + ", ".join(f"{f} {e:.3f}" for f, e in before[0].items()) + f"; worst edge growth {before[1]:.3f}")
check("premise_original_weights_are_broken", bool(stale) and unweighted > 0 and before[0]["ring"] > 0.05 and before[1] > 0.2,
      "(this test would pass on any mesh if this fails)")

# ---- re-skinned ---------------------------------------------------------------------------------------
info = skin_mesh.skin_object(mesh, arm)
names = [g.name for g in mesh.vertex_groups]
check("one_group_per_deform_bone_and_nothing_else", sorted(names) == sorted(b.name for b in arm.data.bones if b.use_deform)
      and len(names) == 20, f"{len(names)} groups")
unweighted = sum(1 for i in sample if not mesh.data.vertices[i].groups)
check("every_vertex_has_a_bone", unweighted == 0, f"{unweighted * 7} unweighted")
mods = [m for m in mesh.modifiers if m.type == "ARMATURE"]
check("one_armature_modifier_on_the_rig", len(mods) == 1 and mods[0].object is arm and mods[0].use_vertex_groups
      and not mods[0].use_bone_envelopes and mods[0].use_deform_preserve_volume)
check("at_most_four_bones_per_vertex", info["mean_bones_per_vertex"] <= 4.0 and info["mean_bones_per_vertex"] > 1.2,
      f"{info['mean_bones_per_vertex']:.2f}")
weights_ok = True
for i in range(0, n_v, 97):
    gs = mesh.data.vertices[i].groups
    if len(gs) > 4 or abs(sum(g.weight for g in gs) - 1.0) > 0.03:
        weights_ok = False
check("weights_sum_to_one", weights_ok)
for finger, bone in TIPS.items():
    tips, _ = tip_vertices(arm, mesh)
    gs = mesh.data.vertices[tips[finger]].groups
    top = max(gs, key=lambda g: g.weight)
    check(f"{finger}_fingertip_belongs_to_its_tip_bone", mesh.vertex_groups[top.group].name == bone,
          mesh.vertex_groups[top.group].name)

after = measure(arm, mesh)
print("AFTER tip follow error " + ", ".join(f"{f} {e:.3f}" for f, e in after[0].items())
      + f"; worst edge growth {after[1]:.3f}; edges grown >0.01: {after[2]}; lowest volume {after[3]:.3f}")
for finger, err in after[0].items():
    check(f"{finger}_fingertip_skin_follows_its_bone", err < 0.05, f"{err:.3f} armature units")
check("no_torn_or_stretched_skin", after[1] < 0.06, f"worst edge growth {after[1]:.3f} (was {before[1]:.3f})")
check("few_edges_grow_visibly", after[2] < 8000 and after[2] < before[2] * 1.5, f"{after[2]} edges (was {before[2]})")
check("volume_stays_close_to_rest", after[3] > 0.94, f"lowest {after[3]:.3f} of rest")

# ---- two hands: both meshes get the same skin --------------------------------------------------------------
arm2 = two.add_second_hand()
pairs = skin_mesh.hand_meshes()
check("skin_tool_finds_both_hand_meshes", len(pairs) == 2 and {a.name for a, _ in pairs} == {"Armature", "Armature.001"})
for a, m in pairs:
    skin_mesh.skin_object(m, a)
m2 = [c for c in arm2.children if c.type == "MESH"][0]
same = all(sorted((mesh.vertex_groups[g.group].name, round(g.weight, 3)) for g in mesh.data.vertices[i].groups)
           == sorted((m2.vertex_groups[g.group].name, round(g.weight, 3)) for g in m2.data.vertices[i].groups)
           for i in range(0, n_v, 1009))
check("second_hand_mesh_has_the_same_skin", same and any(mod.object is arm2 for mod in m2.modifiers if mod.type == "ARMATURE"))

# ---- the command never replaces a file -----------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    script = os.path.join(V5_DIR, "skin_mesh.py")

    def run(out, *extra):
        return subprocess.run([bpy.app.binary_path, "-b", "--disable-autoexec", BLEND, "--python", script, "--",
                               V5_DIR, out, *extra], capture_output=True, text=True, timeout=900)

    out = os.path.join(tmp, "skinned.blend")
    first = run(out)
    check("skin_cli_writes_a_new_file", first.returncode == 0 and os.path.exists(out), first.stdout[-160:])
    stamp = sha(out)
    second = run(out)
    check("skin_cli_refuses_to_replace_a_file", second.returncode != 0 and "already exists" in second.stdout + second.stderr
          and sha(out) == stamp)
    same_file = run(BLEND, "overwrite=1")
    check("skin_cli_never_replaces_its_own_source", same_file.returncode != 0 and sha(BLEND) == BLEND_HASH)

check("source_file_unchanged", sha(BLEND) == BLEND_HASH)
print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
