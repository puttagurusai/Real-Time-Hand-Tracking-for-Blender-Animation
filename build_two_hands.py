"""Add a second hand to a rig .blend: a copy of the armature, its 21 landmark empties and the mesh
skinned to it, placed beside the first. Run headless on the fixed copy; the result is a NEW file:

    blender -b --disable-autoexec v5/untitled_rig_fixed_v2.blend --python v5/build_two_hands.py -- v5 OUT.blend [spacing=0.8] [overwrite=1]

The copies get Blender's own ".001" suffix ("Armature.001", "WRIST.001", ...), which is what the
receiver looks for (HAND_SUFFIXES). Each copy's IK constraints point at the copy's own empties.
The first hand (Armature) is driven by the person's LEFT hand and the copy (Armature.001) by the RIGHT
hand (the receiver mirrors a rig to fit, from the finger bends). spacing is the distance of the copy's wrist
along Blender's X axis (the hand is about 0.5 wide); the default 0.8 puts the right-hand rig on the +X side, the left-hand rig on the other
side of it (use -0.8 to swap them back).
The file you ran it on is never changed or replaced.
"""
import os
import sys

V5_DIR = sys.argv[sys.argv.index("--") + 1]
sys.path.insert(0, V5_DIR)

import bpy
from mathutils import Vector

import rig_tools as rt

SUFFIX = ".001"


def add_second_hand(spacing=0.8, suffix=SUFFIX):
    """Copy the first hand next to itself. Returns the new armature object. The first hand is the right
    hand; the first is driven by the left hand, the copy by the right hand, placed at `spacing` along X
    (default +0.8, so the right hand's rig is on the right of the front view)."""
    arm = bpy.data.objects[rt.ARMATURE_NAME]
    if bpy.data.objects.get(rt.ARMATURE_NAME + suffix) is not None:
        raise ValueError(f"'{rt.ARMATURE_NAME + suffix}' already exists: this file already has a second hand")
    shift = Vector((spacing, 0.0, 0.0))
    original_hidden = {o.name: o.hide_get() for o in bpy.data.objects}
    collections = lambda ob: list(ob.users_collection) or [bpy.context.scene.collection]

    def duplicate(ob, data_copy=False):
        new = ob.copy()
        if data_copy and ob.data is not None:
            new.data = ob.data.copy()
            new.data.name = ob.data.name + suffix
        new.name = ob.name + suffix
        for coll in collections(ob):
            coll.objects.link(new)
        new.hide_viewport, new.hide_render = ob.hide_viewport, ob.hide_render
        try:  # the per-view-layer "eye" is not copied with the object
            new.hide_set(ob.hide_get())
        except RuntimeError:
            pass
        return new

    empties = {}
    for name in rt.LANDMARK_EMPTIES:
        new = duplicate(bpy.data.objects[name])
        new.location = Vector(new.location) + shift
        empties[name] = new

    arm2 = duplicate(arm, data_copy=True)
    arm2.location = Vector(arm.location) + shift
    retargeted = 0
    for pb in arm2.pose.bones:
        for con in pb.constraints:
            target = getattr(con, "target", None)
            if target is not None and target.name in empties:
                con.target = empties[target.name]
                retargeted += 1
    if retargeted == 0:
        raise ValueError("the copied armature has no IK constraints that point at the landmark empties")

    meshes = []
    for child in arm.children:
        new = duplicate(child, data_copy=True)
        new.parent = arm2
        new.matrix_parent_inverse = child.matrix_parent_inverse.copy()
        for mod in new.modifiers:
            if mod.type == "ARMATURE" and mod.object == arm:
                mod.object = arm2
        meshes.append(new.name)
    bpy.context.view_layer.update()  # new objects get their world matrices only after an update
    # Linking new objects scrambles the view layer's per-object "eye" flags in the saved file (the
    # armatures came out hidden and the mesh shown): set every flag again, originals and copies
    for name, hidden in original_hidden.items():
        bpy.data.objects[name].hide_set(hidden)
        copy = bpy.data.objects.get(name + suffix)
        if copy is not None:
            copy.hide_set(hidden)
    print(f"BUILD second hand: armature '{arm2.name}' at {spacing:+g} X, {len(empties)} empties, "
          f"{retargeted} IK constraints re-pointed, meshes {meshes}")
    return arm2


def main():
    args = sys.argv[sys.argv.index("--") + 2:]
    if not args:
        sys.exit("usage: ... -- V5_DIR OUT.blend [spacing=0.8] [overwrite=1]")
    out = os.path.abspath(args[0])
    opts = dict(a.split("=", 1) for a in args[1:])
    if bpy.data.filepath and os.path.abspath(bpy.data.filepath) == out:
        sys.exit("refusing to replace the file this run started from; choose another OUT name")
    if os.path.exists(out) and opts.get("overwrite") != "1":
        sys.exit(f"{out} already exists; choose another name or pass overwrite=1")
    try:
        add_second_hand(float(opts.get("spacing", 0.8)))
    except ValueError as exc:
        sys.exit(f"cannot add a second hand: {exc}")
    bpy.ops.wm.save_as_mainfile(filepath=out, copy=True)
    print(f"BUILD saved a copy to {out}")


if __name__ == "__main__":
    main()
