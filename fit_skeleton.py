"""Fit the hand skeleton to the hand mesh. Run headless; the result is a NEW .blend file:

    blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/fit_skeleton.py -- v5 OUT.blend [python=python] [overwrite=1]

Why: the skeleton that came with the mesh does not fit it. Its wrist joint floats below the mesh and its
finger bones are the wrong lengths (the proximal bone is 62% of a finger, the middle and distal bones
28% and 9%; a real finger is about 45%, 30%, 25%), so the skin is bent about the wrong places and
stretches at the knuckles in strong poses.

How: the mesh is rendered from five directions around the palm, MediaPipe (the model the tracker
uses) finds the 21 hand landmarks in every picture, and the landmarks are triangulated into 3D. The
bones keep their names, parents, IK constraints and empties; only their head and tail positions are
set to those joints. Then the finger rolls, the IK joint limits, the empties and the skin weights are
redone for the new bones. Bone lengths now match the mesh and the tracker's own idea of a hand.

The file you ran it on is not changed. `python=` is the interpreter that has MediaPipe (the one you
run sender.py with); Blender's own Python does not.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(globals().get("__file__", "")))
sys.path.insert(0, sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else _here)

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:  # imported outside Blender: only the maths below is usable
    bpy = None

IMAGE_SIZE = 1024
TILTS_DEG = (20.0, 35.0, 50.0)  # how far the extra views tilt away from looking straight at the palm
MAX_RESIDUAL = 0.012      # of the picture width: a view that disagrees with the rest by more is dropped
HAND_SIZE_IN_FRAME = 0.6  # the hand fills this much of the picture (MediaPipe likes some margin)
SKIN = (0.85, 0.62, 0.5)
BACKGROUND = (0.75, 0.75, 0.78)


# ---- pure maths ------------------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rotation(axis, degrees):
    """3x3 rotation about a unit axis."""
    a = _unit(axis)
    t = math.radians(degrees)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


def make_views(normal, up, lateral, tilts=TILTS_DEG):
    """The camera directions to look at the hand from: straight at the palm, and tilted about the
    palm's two axes by each angle in tilts. Each view is (name, towards_camera, up), all unit vectors
    in the armature's own axes; towards_camera is the direction from the hand to the camera."""
    n, u, lat = _unit(normal), _unit(up), _unit(lateral)
    views = [("palm", n, u)]
    for tilt in tilts:
        for sign in (1, -1):
            r = rotation(lat, sign * tilt)
            views.append((f"pitch{sign * tilt:+.0f}", r @ n, r @ u))
            r = rotation(u, sign * tilt)
            views.append((f"roll{sign * tilt:+.0f}", r @ n, u))
    return views


def camera_axes(towards, up):
    """Right, up and towards-camera axes of an orthographic camera."""
    z = _unit(towards)
    y = _unit(np.asarray(up, float) - z * float(np.dot(up, z)))
    return np.cross(y, z), y, z


def project(point, view):
    """Normalised picture position (x right, y down, both 0..1) of a 3D point seen by `view`, a dict with
    the camera's axes x, y, z, its origin and its ortho_scale."""
    rel = np.asarray(point, float) - view["origin"]
    return (np.dot(rel, view["x"]) / view["ortho_scale"] + 0.5, 0.5 - np.dot(rel, view["y"]) / view["ortho_scale"])


def triangulate(observations, views, max_residual=MAX_RESIDUAL):
    """3D position of one landmark seen in several orthographic views.

    observations: {view index: (x, y)} normalised picture positions. Every view gives two linear
    equations on the point; they are solved by least squares, and a view whose picture position the
    solution misses by more than max_residual of the picture width is dropped (MediaPipe sometimes
    puts a landmark somewhere else in one view) while at least three views remain.
    Returns (point, residual_of_the_views_kept, number_of_views_used)."""
    use = dict(observations)
    while True:
        rows, rhs = [], []
        for i, (x, y) in use.items():
            v = views[i]
            a = (x - 0.5) * v["ortho_scale"]
            b = (0.5 - y) * v["ortho_scale"]
            rows += [v["x"], v["y"]]
            rhs += [a + np.dot(v["origin"], v["x"]), b + np.dot(v["origin"], v["y"])]
        point, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        errors = {i: math.dist(project(point, views[i]), use[i]) for i in use}
        worst = max(errors, key=errors.get)
        if errors[worst] <= max_residual or len(use) <= 3:
            return point, float(np.sqrt(np.mean([e * e for e in errors.values()]))), len(use)
        del use[worst]


# ---- Blender part ----------------------------------------------------------------------------------

BONES = {  # bone -> (head landmark, tail landmark)
    "wrist_to_thumb_cmc": (0, 1), "thumb_cmc_to_thumb_mcp": (1, 2), "thumb_mcp_to_thumb_ip": (2, 3),
    "thumb_ip_to_thumb_tip": (3, 4),
    "wrist_to_index_mcp": (0, 5), "index_mcp_to_index_pip": (5, 6), "index_pip_to_index_dip": (6, 7),
    "index_dip_to_index_tip": (7, 8),
    "wrist_to_middle_mcp": (0, 9), "middle_mcp_to_middle_pip": (9, 10), "middle_pip_to_middle_dip": (10, 11),
    "middle_dip_to_middle_tip": (11, 12),
    "wrist_to_ring_mcp": (0, 13), "ring_mcp_to_ring_pip": (13, 14), "ring_pip_to_ring_dip": (14, 15),
    "ring_dip_to_ring_tip": (15, 16),
    "wrist_to_pinky_mcp": (0, 17), "pinky_mcp_to_pinky_pip": (17, 18), "pinky_pip_to_pinky_dip": (18, 19),
    "pinky_dip_to_pinky_tip": (19, 20),
}


def hand_mesh(arm):
    meshes = [c for c in arm.children if c.type == "MESH"]
    if not meshes:
        raise ValueError(f"'{arm.name}' has no mesh to fit the skeleton to")
    return meshes[0]


class _RenderSetup:
    """Everything the renders change, put back afterwards (the output file is saved from this scene)."""

    def __init__(self, arm, mesh):
        self.arm, self.mesh = arm, mesh

    def __enter__(self):
        scene = bpy.context.scene
        self.scene = scene
        shading = scene.display.shading
        self.saved = {
            "engine": scene.render.engine, "rx": scene.render.resolution_x, "ry": scene.render.resolution_y,
            "pct": scene.render.resolution_percentage, "camera": scene.camera, "world": scene.world,
            "light": shading.light, "ctype": shading.color_type, "single": tuple(shading.single_color),
            "filepath": scene.render.filepath,
            "hide": (self.mesh.hide_get(), self.mesh.hide_viewport, self.mesh.hide_render),
            "mods": [(m, m.show_viewport, m.show_render) for m in self.mesh.modifiers if m.type == "ARMATURE"],
            "render_hidden": [(o, o.hide_render) for o in bpy.data.objects],
        }
        for m, _, _ in self.saved["mods"]:  # the mesh as stored: no IK or pose in the picture
            m.show_viewport = m.show_render = False
        self.mesh.hide_set(False)
        self.mesh.hide_viewport = self.mesh.hide_render = False
        for o in bpy.data.objects:
            if o is not self.mesh:
                o.hide_render = True
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.resolution_x = scene.render.resolution_y = IMAGE_SIZE
        scene.render.resolution_percentage = 100
        shading.light, shading.color_type, shading.single_color = "STUDIO", "SINGLE", SKIN
        self.temp_world = bpy.data.worlds.new("v5_fit_world")
        self.temp_world.color = BACKGROUND
        scene.world = self.temp_world
        self.cam_data = bpy.data.cameras.new("v5_fit_cam")
        self.cam_data.type = "ORTHO"
        self.cam = bpy.data.objects.new("v5_fit_cam", self.cam_data)
        scene.collection.objects.link(self.cam)
        scene.camera = self.cam
        return self

    def __exit__(self, *exc):
        scene, s = self.scene, self.saved
        bpy.data.objects.remove(self.cam)
        bpy.data.cameras.remove(self.cam_data)
        scene.world = s["world"]
        bpy.data.worlds.remove(self.temp_world)
        scene.camera = s["camera"]
        scene.render.engine, scene.render.resolution_x, scene.render.resolution_y = s["engine"], s["rx"], s["ry"]
        scene.render.resolution_percentage, scene.render.filepath = s["pct"], s["filepath"]
        scene.display.shading.light, scene.display.shading.color_type = s["light"], s["ctype"]
        scene.display.shading.single_color = s["single"]
        for m, viewport, render in s["mods"]:
            m.show_viewport, m.show_render = viewport, render
        self.mesh.hide_set(s["hide"][0])
        self.mesh.hide_viewport, self.mesh.hide_render = s["hide"][1], s["hide"][2]
        for o, hidden in s["render_hidden"]:
            o.hide_render = hidden


def palm_axes(arm):
    """(normal toward the palm, along the fingers, across) in the armature's own axes, from the rig's rest
    pose as it comes: the way its fingers curl says which side is the palm."""
    import blender_receiver as br
    rig = br.Rig(arm)
    palm = rig.collision.palm if rig.collision is not None else None
    if palm is None:
        raise ValueError("cannot tell which side of the hand is the palm (the rest pose is too straight and "
                         "has no hinge); fit_skeleton needs the original skeleton's finger curl")
    return np.array(palm["n"]), np.array(palm["u"]), np.array(palm["lat"])


def render_views(arm, mesh, views, folder):
    """Render the mesh from every view; returns a list of view dicts (axes, origin, ortho_scale, path)."""
    world = np.array(arm.matrix_world.to_3x3())
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = mesh.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    mesh_eval.to_mesh_clear()
    mw = np.array(mesh_eval.matrix_world)
    pts = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
    extent = float(max(pts.max(axis=0) - pts.min(axis=0)))
    out = []
    with _RenderSetup(arm, mesh) as setup:
        for name, towards, up in views:
            x, y, z = camera_axes(world @ towards, world @ up)
            origin = centre + z * 4.0
            setup.cam_data.ortho_scale = extent / HAND_SIZE_IN_FRAME
            setup.cam.matrix_world = Matrix(((x[0], y[0], z[0], origin[0]), (x[1], y[1], z[1], origin[1]),
                                             (x[2], y[2], z[2], origin[2]), (0, 0, 0, 1)))
            path = os.path.join(folder, f"fit_view_{name}.png")
            bpy.context.scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            out.append({"name": name, "path": path, "x": x, "y": y, "z": z, "origin": origin,
                        "ortho_scale": setup.cam_data.ortho_scale})
    return out


def detect_all(views, python):
    """Run MediaPipe on every picture (in another process); fills view["landmarks"] (21 x 2) or None."""
    detect = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_detect.py")
    for v in views:
        out = v["path"][:-4] + ".json"
        result = subprocess.run([python, detect, v["path"], out], capture_output=True, text=True)
        v["landmarks"] = None
        if result.returncode == 0 and os.path.exists(out):
            data = json.load(open(out))
            if data.get("found"):
                v["landmarks"] = [(p[0], p[1]) for p in data["landmarks"]]
                v["label"], v["score"] = data["label"], data["score"]
    return views


def fit_joints(arm, mesh, views_info, python):
    """Triangulated landmark positions in the armature's own axes: {landmark: (x, y, z)} plus a report."""
    rendered = detect_all(views_info, python)
    good = [v for v in rendered if v["landmarks"] is not None]
    if len(good) < 3:
        raise RuntimeError(f"MediaPipe found a hand in only {len(good)} of {len(rendered)} pictures "
                           f"(need 3); look at fit_view_*.png next to the output")
    inverse = np.array(arm.matrix_world.inverted())
    joints, report = {}, []
    for k in range(21):
        obs = {i: v["landmarks"][k] for i, v in enumerate(good)}
        p_world, residual, used = triangulate(obs, good)
        p_local = inverse[:3, :3] @ p_world + inverse[:3, 3]
        joints[k] = tuple(float(c) for c in p_local)
        report.append((k, residual, used))
    return joints, report, [v["name"] for v in good]


def apply_joints(arm, joints):
    """Put every bone's head and tail on the fitted joints (edit mode), keeping names and parents."""
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        eb = arm.data.edit_bones
        for name, (_, tail) in BONES.items():
            eb[name].tail = Vector(joints[tail])
        for name, (head, _) in BONES.items():
            eb[name].head = Vector(joints[head])
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def inside_mesh(mesh, points):
    """For each point (armature axes) whether it lies inside the mesh as stored: a ray from the point
    crosses the surface an odd number of times (three directions, majority)."""
    import bmesh
    from mathutils.bvhtree import BVHTree
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    tree = BVHTree.FromBMesh(bm)
    to_mesh = mesh.matrix_world.inverted() @ mesh.parent.matrix_world
    result = []
    for p in points:
        origin = to_mesh @ Vector(p)
        votes = 0
        for d in (Vector((1, 0.013, 0.007)), Vector((0.01, 1, 0.021)), Vector((0.017, 0.009, 1))):
            hits, pos = 0, origin.copy()
            for _ in range(60):
                hit = tree.ray_cast(pos, d.normalized())
                if hit[0] is None:
                    break
                hits += 1
                pos = hit[0] + d.normalized() * 1e-5
            votes += hits % 2
        result.append(votes >= 2)
    bm.free()
    return result


def phalanx_ratios(joints):
    """Per finger: the share of the finger (knuckle to tip) taken by each of its three bones."""
    out = {}
    for finger, (m, p, d, t) in {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
                                 "pinky": (17, 18, 19, 20), "thumb": (1, 2, 3, 4)}.items():
        seg = [math.dist(joints[a], joints[b]) for a, b in ((m, p), (p, d), (d, t))]
        total = sum(seg)
        out[finger] = tuple(s / total if total else 0.0 for s in seg)
    return out


# the joints whose position is refined to the centre of the finger around them: (joint, before, after)
CENTRED = ((6, 5, 7), (7, 6, 8), (10, 9, 11), (11, 10, 12), (14, 13, 15), (15, 14, 16), (18, 17, 19),
           (19, 18, 20), (2, 1, 3), (3, 2, 4))
SLAB = 0.006          # thickness of the slice of mesh used to find a finger's centre
REACH = 0.06          # how far from the joint the slice may reach (keeps the neighbouring finger out)
MAX_MOVE = 0.05       # a joint never moves further than this when it is centred (a finger is about 0.04 thick)


def rest_vertices(mesh, arm):
    """The mesh as stored, in the armature's own axes (N, 3)."""
    co = np.empty(len(mesh.data.vertices) * 3, np.float32)
    mesh.data.vertices.foreach_get("co", co)
    m = np.array(arm.matrix_world.inverted() @ mesh.matrix_world)
    return co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]


def _finger_of(bone):
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        if finger in bone:
            return finger
    return bone


def finger_owner(joints, vertices):
    """For every vertex, which finger's bones it is nearest to (by the fitted joints): an array of finger
    names, so that the slice of mesh around a joint can leave out the neighbouring fingers."""
    import skin_mesh
    names = list(BONES)
    heads = np.array([joints[BONES[n][0]] for n in names], dtype=float)
    tails = np.array([joints[BONES[n][1]] for n in names], dtype=float)
    nearest = skin_mesh.segment_distances(vertices, heads, tails).argmin(axis=1)
    owners = np.array([_finger_of(n) for n in names])
    return owners[nearest]


def centre_joints(joints, vertices, palm_normal=(0.0, 1.0, 0.0), iterations=8, owner=None):
    """Move each finger joint to the middle of its own finger: the centre of the finger's skin in the slice
    of mesh through the joint, across the bone's direction. Only vertices nearest to this finger's bones
    count, so the neighbouring finger does not pull the joint sideways. Returns {joint: how far it moved}."""
    owner = finger_owner(joints, vertices) if owner is None else owner
    moved = {}
    finger_of_joint = {2: "thumb", 3: "thumb"}
    finger_of_joint.update({k: f for f, ks in {"index": (6, 7), "middle": (10, 11), "ring": (14, 15), "pinky": (18, 19)}.items() for k in ks})
    for k, before, after in CENTRED:
        mine = vertices[owner == finger_of_joint[k]]
        start = np.array(joints[k])
        p = start.copy()
        for _ in range(iterations):
            axis = _unit(np.array(joints[after]) - np.array(joints[before]))
            rel = mine - p
            along = rel @ axis
            across = rel - np.outer(along, axis)
            keep = (np.abs(along) < SLAB / 2) & (np.linalg.norm(across, axis=1) < REACH)
            if keep.sum() < 20:
                break
            # the middle of the finger's outer extent in the slice (not the average of its points, which
            # skin folds inside a crease would pull sideways)
            ring = across[keep]
            e1 = _unit(np.array(palm_normal) - axis * float(np.dot(palm_normal, axis)))
            e2 = np.cross(axis, e1)
            step = np.zeros(3)
            for e in (e1, e2):
                coord = ring @ e
                step = step + e * float((np.percentile(coord, 2) + np.percentile(coord, 98)) / 2)
            p = p + step
            if np.linalg.norm(step) < 1e-4:
                break
        if np.linalg.norm(p - start) > MAX_MOVE:
            p = start + _unit(p - start) * MAX_MOVE
        joints[k] = tuple(float(c) for c in p)
        moved[k] = float(np.linalg.norm(p - start))
    return moved


def move_origin_to_wrist(arm, mesh, joints):
    """The receiver pins the WRIST empty to the armature's origin, so the origin must be the wrist: shift the
    joints and the mesh together by the wrist's position. Returns the shifted joints."""
    origin = np.array(joints[0])
    t = np.array(arm.matrix_world.inverted() @ mesh.matrix_world)[:3, :3]
    co = np.empty(len(mesh.data.vertices) * 3, np.float32)
    mesh.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) - (np.linalg.inv(t) @ origin).astype(np.float32)
    mesh.data.vertices.foreach_set("co", co.ravel())
    mesh.data.update()
    return {k: tuple(float(c) for c in (np.array(v) - origin)) for k, v in joints.items()}


def finish_rig(arm, mesh, joints, python_hinge=True):
    """Everything that depends on where the bones are: rolls, IK limits, the empties, the skin."""
    import rig_tools as rt
    import skin_mesh
    rolls = rt.align_finger_rolls(arm, True)
    rt.clear_joint_limits(arm)
    plan, notes = rt.plan_joint_limits(arm)
    rt.apply_joint_limits(arm, plan)
    print(f"FIT rolled {len(rolls)} bones, planned IK limits for {len(plan)} of 12 finger bones")
    for note in notes:
        print("FIT   limits:", note)
    for name, target in rt.BONE_TARGETS.items():  # the empties start where their bone tips are
        bpy.data.objects[target].location = arm.matrix_world @ arm.data.bones[name].tail_local
    bpy.data.objects["WRIST"].location = arm.matrix_world @ Vector((0, 0, 0))
    info = skin_mesh.skin_object(mesh, arm)
    print(f"FIT re-skinned: {info['mean_bones_per_vertex']:.2f} bones per vertex; bone thickness "
          f"{ {k: round(v, 3) for k, v in info['radii'].items() if v is not None} }")


def main():
    args = sys.argv[sys.argv.index("--") + 2:]
    if not args:
        sys.exit("usage: ... -- V5_DIR OUT.blend [python=python] [overwrite=1] [inspect=1] [pictures=DIR]")
    out = os.path.abspath(args[0])
    opts = dict(a.split("=", 1) for a in args[1:])
    if opts.get("inspect") != "1":
        if bpy.data.filepath and os.path.abspath(bpy.data.filepath) == out:
            sys.exit("refusing to replace the file this run started from; choose another OUT name")
        if os.path.exists(out) and opts.get("overwrite") != "1":
            sys.exit(f"{out} already exists; choose another name or pass overwrite=1")
    import tempfile
    import rig_tools as rt
    arm = bpy.data.objects[rt.ARMATURE_NAME]
    mesh = hand_mesh(arm)
    normal, up, lateral = palm_axes(arm)
    folder = opts.get("pictures") or tempfile.mkdtemp(prefix="v5_fit_")
    os.makedirs(folder, exist_ok=True)
    views = render_views(arm, mesh, make_views(normal, up, lateral), folder)
    joints, report, used = fit_joints(arm, mesh, views, opts.get("python", "python"))
    print(f"FIT pictures in {folder}; MediaPipe found the hand in {len(used)} of {len(views)} views: {used}")
    worst = max(report, key=lambda r: r[1])
    print(f"FIT worst landmark disagreement between views: {worst[1]:.4f} of the picture width (landmark {worst[0]}); "
          f"fewest views used for a landmark: {min(r[2] for r in report)}")
    moved = centre_joints(joints, rest_vertices(mesh, arm), normal)
    print("FIT centring moved the joints by (max, mean):", round(max(moved.values()), 4), round(sum(moved.values()) / len(moved), 4))
    inside = inside_mesh(mesh, [joints[k] for k in range(21)])
    print("FIT landmarks outside the mesh:", [k for k, ok in enumerate(inside) if not ok] or "none")
    for finger, r in phalanx_ratios(joints).items():
        print(f"FIT {finger:6s} bone shares of the finger: proximal {r[0]:.2f}, middle {r[1]:.2f}, distal {r[2]:.2f}")
    json.dump({str(k): v for k, v in joints.items()}, open(os.path.join(folder, "joints.json"), "w"))
    if opts.get("inspect") == "1":
        return
    joints = move_origin_to_wrist(arm, mesh, joints)
    apply_joints(arm, joints)
    print("FIT wrist -> middle MCP now", round(math.dist(joints[0], joints[9]), 4))
    finish_rig(arm, mesh, joints)
    bpy.ops.wm.save_as_mainfile(filepath=out, copy=True)
    print(f"FIT saved a copy to {out}")


if __name__ == "__main__" and bpy is not None:
    main()
