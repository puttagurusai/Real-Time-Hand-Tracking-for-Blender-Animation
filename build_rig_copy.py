"""Build the fixed rig copy. Run headless; the source .blend is read, never saved.

    blender -b --disable-autoexec untitled.blend --python v5/build_rig_copy.py -- v5 v5/untitled_rig_fixed.blend

Changes made to the in-memory file before saving it as a copy:
  * clear the meaningless IK `subtarget` on the 20 constraints that target an empty
  * roll the 12 non-thumb finger bones so one local axis is the finger's hinge axis
    (head/tail unchanged, so the rest pose and skinning are identical)
  * add IK limits for those 12 bones: no twist, hinge range PIP 0-110, DIP 0-90, MCP -20..90
    degrees (absolute flexion), knuckle side swing +-30 degrees
  * remove the superseded or conflicting script text blocks (they listen on ports
    5001/5004/5006/5007 or delete scene objects) and add two loaders that run
    blender_receiver.py and rig_tools.py from next to the .blend
"""
import sys

args = sys.argv[sys.argv.index("--") + 1:]
V5_DIR, OUT = args[0], args[1]
sys.path.insert(0, V5_DIR)

import bpy

import rig_tools as rt

REMOVE_TEXTS = ["IK TESTING", "edit mode", "scale", "small", "Text", "Text.001"]
LOADER = '''# Runs {file} from next to this .blend, so there is one copy of the code to maintain.
import bpy
_path = bpy.path.abspath("//{file}")
with open(_path, encoding="utf-8") as _f:
    exec(compile(_f.read(), _path, "exec"), {{"__name__": "__main__", "__file__": _path}})
'''


def main():
    arm = bpy.data.objects[rt.ARMATURE_NAME]
    print("BUILD source:", bpy.data.filepath)
    before = rt.validate()
    print("BUILD before:", sum(lv == "WARN" for lv, _ in before), "warnings,",
          sum(lv == "ERROR" for lv, _ in before), "errors")

    for change in rt.fix_ik(apply=True):
        print("BUILD fixed:", change)
    for change in rt.align_finger_rolls(arm, apply=True):
        print("BUILD fixed:", change)
    plan, notes = rt.plan_joint_limits(arm)
    for n in notes:
        print("BUILD limits:", n)
    if len(plan) != 12:
        raise SystemExit(f"expected limits for 12 finger bones, planned {len(plan)}; not saving")
    rt.apply_joint_limits(arm, plan)

    for name in REMOVE_TEXTS:
        text = bpy.data.texts.get(name)
        if text is not None:
            bpy.data.texts.remove(text)
            print("BUILD removed text block:", name)
    for name, file in (("v5_run_receiver", "blender_receiver.py"), ("v5_rig_tools", "rig_tools.py")):
        text = bpy.data.texts.new(name)
        text.write(LOADER.format(file=file))
        print("BUILD added text block:", name)

    after = rt.validate()
    problems = [(lv, m) for lv, m in after if lv != "OK"]
    print("BUILD after:", len(problems), "problems", problems)
    bpy.ops.wm.save_as_mainfile(filepath=OUT, copy=True)
    print("BUILD saved copy:", OUT)


main()
