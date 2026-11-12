"""Do the fingers stay out of each other? Run headless on a fixed copy (nothing is saved; the .blend on
disk is not changed):

    blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/test_collision_headless.py -- v5 [landmarks.json]

The mesh is skinned in memory first (skin_mesh.py), which also gives every bone its skin thickness.
Then the real receiver poses the hand from the real recording with collision off and on, and three
things are measured:
  capsules   how often neighbouring finger segments overlap more than they do at rest, and how
             often a fingertip sinks into the palm
  mesh       how many vertices of one finger end up deep inside the neighbouring finger's core
  motion     how far collision moves the pose, and whether it makes the motion jumpy
"""
import math
import os
import sys

V5_DIR = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
ARGS = sys.argv[sys.argv.index("--") + 2:]
OPTS = dict(a.split("=", 1) for a in ARGS if "=" in a)  # iters=N, margin=X: try other solver settings
ARGS = [a for a in ARGS if "=" not in a]
TAKE = os.path.abspath(ARGS[0]) if ARGS else os.path.join(os.path.dirname(V5_DIR), "landmarks.json")
sys.path.insert(0, V5_DIR)

import bpy
import numpy as np

import blender_receiver as br
import frames_io
import rig_tools as rt
import skin_mesh

failures = []


def check(name, ok, detail=""):
    print(f"RESULT {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


if "iters" in OPTS:
    br.COLLISION_ITERATIONS = int(OPTS["iters"])
if "margin" in OPTS:
    br.COLLISION_MARGIN = float(OPTS["margin"])
if "percentile" in OPTS:
    skin_mesh.RADIUS_PERCENTILE = float(OPTS["percentile"])
arm = bpy.data.objects[rt.ARMATURE_NAME]
mesh = [c for c in arm.children if c.type == "MESH"][0]
skin_mesh.skin_object(mesh, arm)  # good weights and a radius on every bone, whatever the file had

frames = frames_io.load_frames(TAKE, br.LANDMARK_NAMES)
frames, _ = frames_io.emulate_sender(frames)
state = br.make_states()[0]
state.sid = "collision"
rig, col = state.rig, state.rig.collision
check("radii_come_from_the_mesh", all(abs(rig.chains[f].radii[1] - arm.data.bones[names[1]].get("v5_radius")) < 1e-9
                                       for f, names in br.BONE_CHAINS.items()), str({f: round(rig.chains[f].radii[1], 4) for f in rig.chains}))


def live_points(i, lm, label, score):
    """What _apply_frame hands to retarget(): the live landmarks in the armature's own axes."""
    state.last_seq = i
    if state.scale is None:
        br.update_scale(state, lm)
    rel = br.world_to_blender(lm, state.scale)
    br.update_mirror(state, {"hand": label}, rel)
    matrix = br.drive_armature(state, rel, br._ZERO)
    return [br._matvec(br._inv3(matrix), p) for p in rel]


def depths(pose):
    out = []
    for (fa, fb), entries in col.pairs.items():
        pa, pb = [pose[k] for k in br.CHAIN_LANDMARKS[fa]], [pose[k] for k in br.CHAIN_LANDMARKS[fb]]
        for sa, sb, radii_sum, gap in entries:
            need = min(radii_sum * br.COLLISION_MARGIN, gap * 0.98)
            _, _, c1, c2 = br._closest_segments(pa[sa], pa[sa + 1], pb[sb], pb[sb + 1])
            out.append(max(0.0, need - math.dist(c1, c2)))
    return out


def palm_sinks(pose):
    palm, count = col.palm, 0
    for f in br.FINGERS4:
        for k in (2, 3, 4):
            q = pose[br.CHAIN_LANDMARKS[f][k]]
            a, l, h = br._dot(q, palm["u"]), br._dot(q, palm["lat"]), br._dot(q, palm["n"])
            if 0 <= a < palm["a_mcp"][f] and palm["l_range"][0] <= l <= palm["l_range"][1]:
                need = palm["offset"] + rig.chains[f].radii[k - 1] * br.COLLISION_MARGIN
                count += (-0.5 * need < h < need - 0.003)
    return count


# ---- capsules, motion: the whole recording ---------------------------------------------------------------
free_poses, solved_poses = [], []
for i, (_, label, score, lm) in enumerate(frames):
    local = live_points(i, lm, label, score)
    br.COLLISION = False
    free_poses.append(br.retarget(rig, local, True))
    br.COLLISION = True
    solved_poses.append(br.retarget(rig, local, True))
br.restore_armature(state)

before = sum(d > 0.003 for p in free_poses for d in depths(p))
after = sum(d > 0.003 for p in solved_poses for d in depths(p))
worst_after = max(d for p in solved_poses for d in depths(p))
sinks_before = sum(palm_sinks(p) for p in free_poses)
sinks_after = sum(palm_sinks(p) for p in solved_poses)
print(f"CAPSULES over {len(frames)} frames: overlapping segment pairs {before} -> {after} (worst depth {worst_after:.3f}); "
      f"fingertips in the palm {sinks_before} -> {sinks_after}")
check("premise_the_live_pose_overlaps", before > 30 and sinks_before > 5, f"{before} overlaps, {sinks_before} palm sinks")
check("fingers_no_longer_overlap", after <= 0.3 * before and worst_after < 0.04, f"{after} left of {before}, worst {worst_after:.3f}")
check("fingertips_stay_out_of_the_palm", sinks_after <= 0.25 * sinks_before, f"{sinks_after} left of {sinks_before}")

angles = []
for free, solved in zip(free_poses, solved_poses):
    for f in rig.chains:
        idx = br.CHAIN_LANDMARKS[f]
        for s in range(4):
            a = br._unit(br._sub(free[idx[s + 1]], free[idx[s]]))
            b = br._unit(br._sub(solved[idx[s + 1]], solved[idx[s]]))
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, br._dot(a, b))))))
angles.sort()
mean_move, p90_move = sum(angles) / len(angles), angles[int(0.9 * len(angles))]
print(f"MOTION collision changes a bone direction by {mean_move:.2f} deg on average, {p90_move:.2f} at the 90th percentile")
check("collision_moves_the_pose_only_a_little", mean_move < 2.0 and p90_move < 4.0, f"mean {mean_move:.2f}, p90 {p90_move:.2f}")


def worst_jump(poses):
    return [max(math.dist(p, q) for p, q in zip(a, b)) for a, b in zip(poses, poses[1:])]


jf, js = sorted(worst_jump(free_poses)), sorted(worst_jump(solved_poses))
p95f, p95s = jf[int(0.95 * len(jf))], js[int(0.95 * len(js))]
print(f"MOTION frame-to-frame travel of the fastest point, 95th percentile: {p95f:.3f} without, {p95s:.3f} with collision")
print(f"MOTION the single biggest frame-to-frame jump: {jf[-1]:.3f} without, {js[-1]:.3f} with collision")
check("collision_does_not_make_motion_jumpy", p95s < p95f * 1.6 + 0.01 and js[-1] < jf[-1] + 0.06,
      f"95th percentile {p95s:.3f} against {p95f:.3f}; biggest {js[-1]:.3f} against {jf[-1]:.3f}")

# the rest pose is the contact the mesh was made with
rest = [(0.0, 0.0, 0.0)] * 21
for f, chain in rig.chains.items():
    pos = (0.0, 0.0, 0.0)
    for i, d, length in zip(br.CHAIN_LANDMARKS[f][1:], chain.rest_dirs, chain.lengths):
        pos = tuple(p + length * c for p, c in zip(pos, d))
        rest[i] = pos
br.COLLISION = False
rest_limited = br.retarget(rig, rest, True)  # the limits and the straight dead zone may adjust a rest pose
br.COLLISION = True
moved = max(math.dist(a, b) for a, b in zip(br.retarget(rig, rest, True), rest_limited))
check("rest_pose_is_not_pushed_by_collision", moved < 1e-9, f"{moved:.2e}")

# The index finger given the middle finger's own shape and turned toward it: the two lie across each
# other at the same height. The pose is built from the rig's own bone lengths, so the solver gets
# exactly what retarget() would hand it and the joint limits cannot hide the overlap first.
n = col.palm["n"]
crossed_depths = []
for sign in (1.0, -1.0):
    pts = list(rest)
    idx_i, idx_m = br.CHAIN_LANDMARKS["index"], br.CHAIN_LANDMARKS["middle"]
    c, s_ = math.cos(math.radians(18.0 * sign)), math.sin(math.radians(18.0 * sign))
    pos = pts[idx_i[1]]
    for s in (1, 2, 3):
        d = br._unit(br._sub(rest[idx_m[s + 1]], rest[idx_m[s]]))
        k, dn = br._cross(n, d), br._dot(n, d)
        d = tuple(d[j] * c + k[j] * s_ + n[j] * dn * (1 - c) for j in range(3))
        pos = br._add(pos, br._scale(d, rig.chains["index"].lengths[s]))
        pts[idx_i[s + 1]] = pos
    solved_pose = list(pts)
    br.resolve_collisions(rig, solved_pose, True)
    crossed_depths.append((max(depths(pts)), max(depths(solved_pose)),
                           max(math.dist(solved_pose[i], pts[i]) for i in range(21))))
worst_free, worst_solved, travel = max(crossed_depths, key=lambda t: t[0])
check("crossed_fingers_are_pushed_apart", worst_free > 0.008 and worst_solved < 0.004,
      f"depth {worst_free:.3f} -> {worst_solved:.3f}, the most any point moved {travel:.3f}")

# ---- the mesh ----------------------------------------------------------------------------------------
n_v = len(mesh.data.vertices)
dominant = np.full(n_v, -1, dtype=np.int32)
group_bone = {g.index: g.name for g in mesh.vertex_groups}
bone_ids = {name: i for i, name in enumerate(group_bone.values())}
for v in mesh.data.vertices:
    if v.groups:
        dominant[v.index] = max(v.groups, key=lambda g: g.weight).group
owned = {}  # finger -> vertices whose strongest bone is one of its three finger bones
for f, names in br.BONE_CHAINS.items():
    ids = [g.index for g in mesh.vertex_groups if g.name in names[1:]]
    owned[f] = np.nonzero(np.isin(dominant, ids))[0]
SAMPLE = list(range(34, len(frames), 6))


def evaluated(ob):
    bpy.context.view_layer.update()
    return ob.evaluated_get(bpy.context.evaluated_depsgraph_get())


DEPTHS = (0.5, 0.35, 0.2)  # how deep: closer to the neighbour's bone than this fraction of its radius


def deep_vertices(collision):
    """Vertices of one finger that sit deep inside the neighbouring finger, summed over the sampled
    poses, for each depth in DEPTHS: {depth: count}, and the counts per finger pair."""
    br.COLLISION = collision
    st = br.make_states()[0]
    st.sid = "mesh"
    total, per_pair = {d: 0 for d in DEPTHS}, {}
    for i, (_, label, score, lm) in enumerate(frames):
        st.last_seq = i
        br._apply_frame(st, {"lm": lm, "hand": label, "score": score})
        if i not in SAMPLE:
            continue
        ev_arm, ev_mesh = evaluated(arm), evaluated(mesh)
        me = ev_mesh.to_mesh()
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        ev_mesh.to_mesh_clear()
        m = np.array(ev_arm.matrix_world.inverted() @ ev_mesh.matrix_world)
        pts = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
        for fa, fb in br.COLLISION_PAIRS:
            for a, b in ((fa, fb), (fb, fa)):
                seg_points = pts[owned[a]]
                for s in (1, 2, 3):
                    bone = ev_arm.pose.bones[br.BONE_CHAINS[b][s]]
                    radius = rig.chains[b].radii[s]
                    d = skin_mesh.segment_distances(seg_points, np.array([tuple(bone.head)]), np.array([tuple(bone.tail)]))[:, 0]
                    for depth in DEPTHS:
                        n_deep = int((d < depth * radius).sum())
                        total[depth] += n_deep
                        per_pair[(a, b, depth)] = per_pair.get((a, b, depth), 0) + n_deep
    br.restore_armature(st)
    br.COLLISION = True
    return total, per_pair


deep_free, pairs_free = deep_vertices(False)
deep_solved, pairs_solved = deep_vertices(True)
for depth in DEPTHS:
    print(f"MESH vertices closer than {depth} radius to a neighbouring finger's bone, {len(SAMPLE)} poses: "
          f"{deep_free[depth]} -> {deep_solved[depth]}")
top = sorted(((k[:2], v) for k, v in pairs_free.items() if k[2] == 0.35), key=lambda kv: -kv[1])[:3]
top_after = {k[:2]: pairs_solved[k] for k, _ in sorted(((k, v) for k, v in pairs_free.items() if k[2] == 0.35), key=lambda kv: -kv[1])[:3]}
print(f"MESH worst pairs at 0.35 without collision {top}; after {top_after}")
check("premise_mesh_fingers_go_through_each_other", deep_free[0.35] > 500, f"{deep_free[0.35]} vertices")
check("mesh_fingers_stay_out_of_each_other", deep_solved[0.35] <= 0.3 * deep_free[0.35],
      f"{deep_solved[0.35]} left of {deep_free[0.35]} (closer than 0.35 radius)")

print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
