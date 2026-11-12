"""Re-skin the hand mesh to the 20 hand bones, so it bends like a hand. Run headless; the result is
a NEW .blend file:

    blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/skin_mesh.py -- v5 OUT.blend [power=10] [volume=1|0] [overwrite=1]

Why: in the original file the mesh's vertex weights do not fit the skeleton. 5% of the vertices have
no weights, the ring fingertip's weights sit in a group named with a stray quote (so that bone moves
nothing), one palm bone owns most of the palm, and the thumb knuckle bone owns nothing. Moving the
skeleton then drags the ring fingertip out into a ribbon and distorts the palm.

How: every vertex is weighted to the bones nearest to it. Its weight for a bone is
(distance to the nearest bone / distance to this bone) ** power, only the four strongest are kept
and the four are normalised. A vertex halfway between two bones at a joint therefore blends them
smoothly, and one in the middle of a finger follows its own bone alone. A higher power makes the
joints sharper.
Options (key=value):
    power=10       falloff: 3 (soft joints, more stretching) to 20 (crisp, a little less volume)
    volume=1|0     the Armature modifier's "Preserve Volume" (default on): less pinching and
                   shrinking at bent joints (the worst volume loss over real poses: 3.6% against
                   7.6%), no measurable cost
    overwrite=1    allow replacing an existing OUT (the file you opened is never replaced)
The file you ran it on is not changed. Every hand mesh (one per armature) is re-skinned.

It also stores on every bone how thick the skin is around it ("v5_radius": the 75th percentile of the
distance from the bone to the vertices it owns), which the receiver's finger collision uses to make
a capsule around each finger segment (RADIUS_PERCENTILE here; a higher number keeps fingers further
apart and moves the pose more).
"""
import os
import sys

import numpy as np

try:
    import bpy
except ImportError:  # imported outside Blender: only the weight maths is usable
    bpy = None

DEFAULT_POWER = 10.0  # measured on the fitted skeleton: less stretching than 6 and still a good volume
PALM_POWER = 1.5      # how sharply the wrist-to-knuckle palm bones share the palm between them
PALM_PREFIX = "wrist_to_"
RADIUS_KEY = "v5_radius"   # custom property on each bone: thickness of the skin around it
SMOOTH_ITERATIONS = 6      # passes of averaging the weights over the mesh surface (0 = none); more barely changes the skin
RADIUS_PERCENTILE = 75     # of the distances from a bone to its vertices (50 = median)
KEEP = 4              # bones that may influence one vertex
MIN_WEIGHT = 0.01     # weights below this are dropped before normalising
CHUNK = 100_000


def segment_distances(points, heads, tails):
    """Distance from every point (N, 3) to every bone segment head -> tail (B, 3): an (N, B) array."""
    seg = tails - heads
    length2 = np.maximum((seg * seg).sum(axis=1), 1e-12)
    out = np.empty((len(points), len(heads)), dtype=np.float32)
    for start in range(0, len(points), CHUNK):
        p = points[start:start + CHUNK, None, :]
        t = np.clip(((p - heads) * seg).sum(axis=2) / length2, 0.0, 1.0)
        closest = heads + t[:, :, None] * seg
        out[start:start + CHUNK] = np.sqrt(((p - closest) ** 2).sum(axis=2))
    return out


def skin_weights(points, heads, tails, power=DEFAULT_POWER, keep=KEEP, palm=None, palm_power=PALM_POWER,
                 edges=None, smooth=None, behind=None):
    """(N, B) float32 weights, at most `keep` non-zero per row, each row summing to 1.
    palm: optional boolean mask of the palm bones. Those bones fan out from the wrist and the palm
    is one stiff piece, so they do not compete sharply with each other: the palm's share of a vertex
    is that of its nearest palm bone, and it is spread over the palm bones with the soft palm_power."""
    dist = np.maximum(segment_distances(points, heads, tails), 1e-6)
    nearest = dist.min(axis=1, keepdims=True)
    w = (nearest / dist) ** power
    if palm is not None and palm.any() and not palm.all():
        d_palm = dist[:, palm].min(axis=1, keepdims=True)
        spread = (d_palm / dist[:, palm]) ** palm_power
        w[:, palm] = ((nearest / d_palm) ** power) * spread / spread.sum(axis=1, keepdims=True)
    if behind is not None and palm is not None:
        # the forearm below the wrist moves with the palm and nothing else: a finger or thumb bone that
        # swings must not pull on the cuff (it showed as a flat slab at the wrist)
        origin, axis = behind
        cuff = (points - origin) @ axis < 0.0
        w[np.ix_(cuff, ~palm)] = 0.0
        w[cuff] /= np.maximum(w[cuff].sum(axis=1, keepdims=True), 1e-12)
    if edges is not None:  # neighbouring vertices get similar weights before the strongest bones are chosen
        w /= w.sum(axis=1, keepdims=True)
        w = smooth_weights(w, edges, SMOOTH_ITERATIONS if smooth is None else smooth)
    if keep < w.shape[1]:  # exactly `keep` strongest bones, even where several are equally close
        strongest = np.argpartition(w, -keep, axis=1)[:, -keep:]
        mask = np.zeros(w.shape, dtype=bool)
        np.put_along_axis(mask, strongest, True, axis=1)
        w = np.where(mask, w, 0.0)
    w /= w.sum(axis=1, keepdims=True)
    w = np.where(w >= MIN_WEIGHT, w, 0.0)
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


def smooth_weights(weights, edges, iterations=SMOOTH_ITERATIONS, blend=0.5):
    """Average every vertex's weights with those of the vertices it shares an edge with, a few times.
    Neighbouring vertices then cannot have very different weights, which is what tears the skin open
    when a bone moves (a gap appears between two vertices that follow different bones). The weights
    are spread along the surface only, so two fingers a hair apart stay separate.
    weights (N, B), edges (E, 2): vertex index pairs. Returns (N, B) float32."""
    if iterations <= 0:
        return weights
    a, b = edges[:, 0], edges[:, 1]
    n = len(weights)
    degree = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    degree = np.maximum(degree, 1).astype(np.float32)
    w = weights.astype(np.float32)
    for _ in range(iterations):
        total = np.empty_like(w)
        for column in range(w.shape[1]):
            col = w[:, column]
            total[:, column] = np.bincount(a, weights=col[b], minlength=n) + np.bincount(b, weights=col[a], minlength=n)
        w = (1.0 - blend) * w + blend * total / degree[:, None]
    return w


def bone_radii(points, heads, tails, weights, percentile=None):
    """How thick the skin is around each bone: a high percentile of the distance from the bone to the
    vertices it owns (weight over 0.5), so most of the skin lies inside the capsule built from it.
    None for a bone that owns no vertex. In the armature's units."""
    percentile = RADIUS_PERCENTILE if percentile is None else percentile
    out = []
    for b in range(len(heads)):
        owned = np.nonzero(weights[:, b] > 0.5)[0]
        if len(owned) == 0:
            out.append(None)
            continue
        out.append(float(np.percentile(segment_distances(points[owned], heads[b:b + 1], tails[b:b + 1])[:, 0], percentile)))
    return out


def rest_points(mesh, arm):
    """The mesh's vertices in the armature's own axes, from the mesh as stored (no modifiers)."""
    n = len(mesh.data.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    mesh.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    m = np.array((arm.matrix_world.inverted() @ mesh.matrix_world))
    return co @ m[:3, :3].T + m[:3, 3]


def skin_object(mesh, arm, power=DEFAULT_POWER, volume=True, smooth=None):
    """Replace all of the mesh's vertex weights with distance weights to arm's deform bones, drop its
    stale groups and its empty Armature modifiers, and point one modifier at arm."""
    bpy.context.view_layer.update()
    bones = [b for b in arm.data.bones if b.use_deform]
    heads = np.array([tuple(b.head_local) for b in bones], dtype=np.float64)
    tails = np.array([tuple(b.tail_local) for b in bones], dtype=np.float64)
    palm = np.array([b.name.startswith(PALM_PREFIX) for b in bones])
    points = rest_points(mesh, arm)
    edge_ids = np.empty(len(mesh.data.edges) * 2, dtype=np.int32)
    mesh.data.edges.foreach_get("vertices", edge_ids)
    wrist = heads[palm].mean(axis=0) if palm.any() else None
    axis = (tails[palm] - heads[palm]).sum(axis=0) if palm.any() else None
    behind = (wrist, axis / np.linalg.norm(axis)) if palm.any() and np.linalg.norm(axis) > 1e-9 else None
    weights = skin_weights(points, heads, tails, power, palm=palm, edges=edge_ids.reshape(-1, 2), smooth=smooth,
                           behind=behind)
    radii = bone_radii(points, heads, tails, weights)
    for bone, radius in zip(bones, radii):  # the receiver's finger collision reads this
        if radius is not None:
            bone[RADIUS_KEY] = radius
        elif RADIUS_KEY in bone:
            del bone[RADIUS_KEY]

    for group in list(mesh.vertex_groups):
        mesh.vertex_groups.remove(group)
    quantised = np.rint(weights * 255.0).astype(np.uint8)
    for column, bone in enumerate(bones):
        group = mesh.vertex_groups.new(name=bone.name)
        col = quantised[:, column]
        for level in np.unique(col[col > 0]):
            group.add(np.nonzero(col == level)[0].tolist(), float(level) / 255.0, "REPLACE")

    mods = [m for m in mesh.modifiers if m.type == "ARMATURE"]
    keep = next((m for m in mods if m.object is arm), None) or (mods[0] if mods else None)
    for m in mods:
        if m is not keep:
            mesh.modifiers.remove(m)
    if keep is None:
        keep = mesh.modifiers.new("Armature", "ARMATURE")
    keep.object = arm
    keep.use_vertex_groups = True
    keep.use_bone_envelopes = False
    keep.use_deform_preserve_volume = bool(volume)
    return {"vertices": len(weights), "bones": len(bones), "radii": dict(zip((b.name for b in bones), radii)),
            "mean_bones_per_vertex": float((weights > 0).sum(axis=1).mean())}


def hand_meshes():
    """[(armature, mesh)] for every armature object that has skinned child meshes."""
    pairs = []
    for arm in bpy.data.objects:
        if arm.type == "ARMATURE":
            pairs += [(arm, child) for child in arm.children if child.type == "MESH"]
    return pairs


def main():
    args = sys.argv[sys.argv.index("--") + 2:]
    if not args:
        sys.exit("usage: ... -- V5_DIR OUT.blend [power=10] [volume=1|0] [overwrite=1]")
    out = os.path.abspath(args[0])
    opts = dict(a.split("=", 1) for a in args[1:])
    if bpy.data.filepath and os.path.abspath(bpy.data.filepath) == out:
        sys.exit("refusing to replace the file this run started from; choose another OUT name")
    if os.path.exists(out) and opts.get("overwrite") != "1":
        sys.exit(f"{out} already exists; choose another name or pass overwrite=1")
    pairs = hand_meshes()
    if not pairs:
        sys.exit("no armature with a child mesh in this file")
    for arm, mesh in pairs:
        info = skin_object(mesh, arm, float(opts.get("power", DEFAULT_POWER)), opts.get("volume", "1") == "1")
        print(f"SKIN {mesh.name} -> {arm.name}: {info['vertices']} vertices, {info['bones']} bones, "
              f"{info['mean_bones_per_vertex']:.2f} bones per vertex")
        print("SKIN bone radii:", {k: round(v, 4) for k, v in info["radii"].items() if v is not None})
    bpy.ops.wm.save_as_mainfile(filepath=out, copy=True)
    print(f"SKIN saved a copy to {out}")


if __name__ == "__main__" and bpy is not None:
    main()
