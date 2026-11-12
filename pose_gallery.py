"""Render the hand mesh in a set of extreme poses, to see whether it still looks like a hand.

    blender -b --disable-autoexec v5/untitled_rig_fitted.blend --python v5/pose_gallery.py -- v5 OUT_PREFIX

Writes OUT_PREFIX_gallery.png: one row per pose, two views each (looking at the palm, and from the
side). The poses are built from the rig's own bones and put on the landmark empties, so the IK
solves them exactly as it does for a live hand. Nothing is saved to the .blend.
"""
import math
import sys

V5_DIR = sys.argv[sys.argv.index("--") + 1]
PREFIX = sys.argv[sys.argv.index("--") + 2]
sys.path.insert(0, V5_DIR)

import bpy
import numpy as np
from mathutils import Matrix, Vector

import blender_receiver as br
import rig_tools as rt

OPTS = dict(a.split("=", 1) for a in sys.argv[sys.argv.index("--") + 3:])  # only=NAME zoom=N ox=F oy=F
IMG = int(OPTS.get("size", 360))
arm = bpy.data.objects[rt.ARMATURE_NAME]
mesh = [c for c in arm.children if c.type == "MESH"][0]
mesh.hide_set(False)
mesh.hide_viewport = mesh.hide_render = False
state = br.make_states()[0]
rig = state.rig
col = rig.collision
palm = col.palm
FINGERS = br.FINGERS4


def rot(axis, deg, v):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    k = br._cross(axis, v)
    d = br._dot(axis, v)
    return tuple(v[i] * c + k[i] * s + axis[i] * d * (1 - c) for i in range(3))


def finger_points(finger, mcp, pip, dip, swing=0.0):
    """Chain points (armature-local) of a finger bent mcp/pip/dip degrees about its hinge, swung
    sideways about the palm normal by `swing` degrees."""
    chain = rig.chains[finger]
    h = chain.hinge
    base = chain.rest_dirs[0]
    pts = [(0.0, 0.0, 0.0)]
    pts.append(tuple(chain.lengths[0] * c for c in base))
    # flat reference for the flexion angles: the finger's own rest proximal direction with its rest bend removed
    ref = rot(h, -sum(math.degrees(math.atan2(br._dot(h, br._cross(a, b)), br._dot(a, b)))
                      for a, b in ((chain.rest_dirs[0], chain.rest_dirs[1]),)), chain.rest_dirs[1])
    cum = 0.0
    for s, bend in zip((1, 2, 3), (mcp, pip, dip)):
        cum += bend
        d = rot(h, cum, ref)
        if swing:
            d = rot(palm["n"], swing, d)
        pts.append(tuple(p + chain.lengths[s] * c for p, c in zip(pts[-1], d)))
    return pts


def thumb_points(adduct, flex1, flex2, flex3):
    chain = rig.chains["thumb"]
    axis = palm["lat"]
    pts = [(0.0, 0.0, 0.0)]
    pts.append(tuple(chain.lengths[0] * c for c in chain.rest_dirs[0]))
    d = chain.rest_dirs[1]
    d = rot(palm["n"], adduct, d)
    for s, bend in zip((1, 2, 3), (flex1, flex2, flex3)):
        d_s = rot(axis, bend, d) if s > 1 else d
        d = d_s
        pts.append(tuple(p + chain.lengths[s] * c for p, c in zip(pts[-1], d)))
    return pts


def rest_points():
    """The rig's own rest pose: every bone tip where the armature has it."""
    pts = [(0.0, 0.0, 0.0)] * 21
    for finger, chain in rig.chains.items():
        pos = (0.0, 0.0, 0.0)
        for i, d, length in zip(br.CHAIN_LANDMARKS[finger][1:], chain.rest_dirs, chain.lengths):
            pos = tuple(p + length * c for p, c in zip(pos, d))
            pts[i] = pos
    return pts


POSES = {
    "rest": dict(rest=True),
    "fist": dict(fingers=(85, 100, 70), thumb=(-35, 20, 25, 20)),
    "claw": dict(fingers=(15, 80, 45), thumb=(0, 0, 0, 0)),
    "hook": dict(fingers=(0, 95, 5), thumb=(0, 0, 0, 0)),
    "point": dict(fingers=(85, 100, 70), thumb=(-30, 20, 20, 15), straight=("index",), thumb_straight=False),
    "spread": dict(fingers=(0, 0, 0), spread=22, thumb=(25, 0, 0, 0)),
    "pinch": dict(fingers=(60, 60, 40), thumb=(-20, 35, 30, 20), index_pinch=True),
}


def build(pose):
    if pose.get("rest"):
        return rest_points()
    pts = [(0.0, 0.0, 0.0)] * 21
    swings = {"index": -1.0, "middle": -0.35, "ring": 0.35, "pinky": 1.0}
    for f in FINGERS:
        mcp, pip, dip = pose["fingers"]
        if f in pose.get("straight", ()):
            mcp, pip, dip = 0, 0, 0
        if pose.get("index_pinch") and f == "index":
            mcp, pip, dip = 50, 55, 25
        sw = pose.get("spread", 0) * swings[f]
        for i, p in zip(br.CHAIN_LANDMARKS[f][1:], finger_points(f, mcp, pip, dip, sw)[1:]):
            pts[i] = p
    for i, p in zip(br.CHAIN_LANDMARKS["thumb"][1:], thumb_points(*pose["thumb"])[1:]):
        pts[i] = p
    return pts


scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = scene.render.resolution_y = IMG
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "SINGLE"
rgb = lambda text, default: tuple(float(c) for c in OPTS.get(text, default).split(","))  # color=r,g,b bg=r,g,b
scene.display.shading.single_color = rgb("color", "0.85,0.7,0.6")
world = scene.world or bpy.data.worlds.new("w")
scene.world = world
world.color = rgb("bg", "0.12,0.12,0.14")
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
for o in bpy.data.objects:
    if o.type in {"EMPTY", "ARMATURE"}:
        o.hide_render = True

M = arm.matrix_world.to_3x3()
n_w = M @ Vector(palm["n"])
u_w = M @ Vector(palm["u"])
l_w = M @ Vector(palm["lat"])
VIEWS = {"palm": (n_w, u_w), "side": (l_w, u_w)}


def evaluated_world_points():
    bpy.context.view_layer.update()
    ev = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    mw = np.array(ev.matrix_world)
    return co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]


def draw_skeleton(path, x, y, origin):
    """Paint the posed bones (red lines) and joints (yellow dots) over a rendered view."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    a = a.reshape(h, w, 4)
    ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mw = ev.matrix_world

    def pix(p):
        rel = mw @ Vector(p) - origin
        return (rel.dot(x) / cam_data.ortho_scale + 0.5) * w, (rel.dot(y) / cam_data.ortho_scale + 0.5) * h

    for pb in ev.pose.bones:
        (x0, y0), (x1, y1) = pix(pb.head), pix(pb.tail)
        for t in np.linspace(0, 1, 200):
            px, py = int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t)
            if 0 <= px < w and 0 <= py < h:
                a[py, px] = (1, 0.1, 0.1, 1)
    for pb in ev.pose.bones:
        for p in (pb.head, pb.tail):
            cx, cy = pix(p)
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    px, py = int(cx) + dx, int(cy) + dy
                    if 0 <= px < w and 0 <= py < h:
                        a[py, px] = (1, 0.9, 0.1, 1)
    img.pixels.foreach_set(a.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def shoot(view, tag):
    eye_dir, up = VIEWS[view]
    z = eye_dir.normalized()               # camera sits on this side
    y = (up - z * up.dot(z)).normalized()
    x = y.cross(z)
    w = evaluated_world_points()
    lo, hi = w.min(axis=0), w.max(axis=0)
    centre = Vector(((lo + hi) / 2).tolist())
    extent = 1.15 * float(max(hi - lo))
    cam_data.ortho_scale = extent / float(OPTS.get("zoom", 1))
    cam.matrix_world = Matrix(((x[0], y[0], z[0], 0), (x[1], y[1], z[1], 0), (x[2], y[2], z[2], 0), (0, 0, 0, 1)))
    # ox, oy: move the view's centre by that fraction of the full framing (right / up in the image)
    cam.matrix_world.translation = (centre + z * 4.0 + x * (float(OPTS.get("ox", 0)) * extent)
                                    + y * (float(OPTS.get("oy", 0)) * extent))
    scene.render.filepath = f"{PREFIX}_{tag}_{view}.png"
    bpy.ops.render.render(write_still=True)
    if OPTS.get("skeleton") == "1":
        draw_skeleton(scene.render.filepath, x, y, cam.matrix_world.translation)
    return scene.render.filepath


# --- numbers per pose: how much any mesh edge grows, and how the volume changes
n_v = len(mesh.data.vertices)
_co = np.empty(n_v * 3, np.float32)
mesh.data.vertices.foreach_get("co", _co)
REST = _co.reshape(-1, 3)
_e = np.empty(len(mesh.data.edges) * 2, np.int32)
mesh.data.edges.foreach_get("vertices", _e)
EDGES = _e.reshape(-1, 2)
REST_LEN = np.linalg.norm(REST[EDGES[:, 0]] - REST[EDGES[:, 1]], axis=1)
mesh.data.calc_loop_triangles()
_t = np.empty(len(mesh.data.loop_triangles) * 3, np.int32)
mesh.data.loop_triangles.foreach_get("vertices", _t)
TRI = _t.reshape(-1, 3)
SCALE = abs(mesh.scale[0])


def _volume(p):
    a, b, c = (p[TRI[:, k]].astype(np.float64) for k in range(3))
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


REST_VOL = _volume(REST)


def pose_metrics(name):
    bpy.context.view_layer.update()
    ev = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    p = co.reshape(-1, 3)
    grow = (np.linalg.norm(p[EDGES[:, 0]] - p[EDGES[:, 1]], axis=1) - REST_LEN) * SCALE
    top = np.argsort(-grow)[:300]
    names = {}
    for k in top:
        pair = []
        for v in EDGES[k]:
            gs = mesh.data.vertices[int(v)].groups
            pair.append(mesh.vertex_groups[max(gs, key=lambda g: g.weight).group].name if gs else "none")
        key = tuple(sorted(pair))
        names[key] = names.get(key, 0) + 1
    print("GALLERY WORST", name, sorted(names.items(), key=lambda kv: -kv[1])[:4])
    print(f"GALLERY METRIC {name:7s} worst edge growth {grow.max():.3f}  edges grown >0.01: {int((grow > 0.01).sum()):6d}  "
          f">0.03: {int((grow > 0.03).sum()):5d}  volume {_volume(p) / REST_VOL:.3f} of rest")


rows = []
for name, pose in POSES.items():
    if "only" in OPTS and name != OPTS["only"]:
        continue
    local = build(pose)
    for i, p in enumerate(local):
        empty = bpy.data.objects.get(br.LANDMARK_NAMES[i])
        empty.location = arm.matrix_world @ Vector(p)
    pose_metrics(name)
    paths = [shoot(v, name) for v in VIEWS]
    rows.append((name, paths))
    print("GALLERY", name)

def load(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(h, w, 4)


# Blender stores images bottom-up, so the first pose goes last to end up on top
rows_img = [np.hstack([load(path) for path in paths]) for _, paths in rows]
sheet = np.vstack(rows_img[::-1])
out = bpy.data.images.new("gallery", sheet.shape[1], sheet.shape[0], alpha=True)
out.pixels.foreach_set(sheet.ravel())
out.filepath_raw = f"{PREFIX}_gallery.png"
out.file_format = "PNG"
out.save()
print("GALLERY wrote", out.filepath_raw, "rows top to bottom:", [name for name, _ in rows], "columns: palm view | side view")
