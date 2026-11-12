"""Sidebar panel for the hand receiver: 3D viewport, press N, tab "Hand".

blender_receiver.py loads this file when it starts (nothing to run by hand). The panel offers
Start / Stop for the receiver, a live status (tracking or lost, frames per second, latency, which
way the rig is mirrored) and the mirror choice: automatic from the finger bends, or forced.
It only talks to the receiver through the two entries the receiver publishes in
bpy.app.driver_namespace ("v5_hand_status" and "v5_hand_control"). The receiver works without it.
"""
import bpy

STATUS_KEY = "v5_hand_status"
CONTROL_KEY = "v5_hand_control"
CLASSES_KEY = "v5_hand_panel_classes"
REDRAW_KEY = "v5_hand_panel_redraw"
SKIP_KEY = "v5_hand_panel_skip"    # set by the Start button so the panel is not re-registered mid-click
REDRAW_SECONDS = 0.5

MIRROR_CHOICES = {"AUTO": None, "MIRRORED": True, "NOT_MIRRORED": False}


def _control():
    return bpy.app.driver_namespace.get(CONTROL_KEY)


def _mirror_changed(self, context):
    control = _control()
    if control is not None:  # with two hands each has its own choice; with one it is the default
        control["set_mirror"](MIRROR_CHOICES[self.v5_hand_mirror], 0 if control.get("hands", 1) > 1 else None)


def _mirror2_changed(self, context):
    control = _control()
    if control is not None and control.get("hands", 1) > 1:
        control["set_mirror"](MIRROR_CHOICES[self.v5_hand_mirror2], 1)


class V5HAND_OT_start(bpy.types.Operator):
    bl_idname = "v5hand.start"
    bl_label = "Start receiver"
    bl_description = "Listen for the sender and drive the hand rig"

    def execute(self, context):
        if _control() is not None:
            self.report({"INFO"}, "The receiver is already running")
            return {"CANCELLED"}
        path = RECEIVER_FILE
        try:
            with open(path, encoding="utf-8") as f:
                code = compile(f.read(), path, "exec")
            bpy.app.driver_namespace[SKIP_KEY] = True
            exec(code, {"__name__": "__main__", "__file__": path})
        except Exception as exc:
            bpy.app.driver_namespace.pop(SKIP_KEY, None)
            self.report({"ERROR"}, f"Cannot start the receiver: {exc!r}")
            return {"CANCELLED"}
        bpy.app.driver_namespace.pop(SKIP_KEY, None)
        wm = context.window_manager
        control = _control()
        if control is not None:
            two = control.get("hands", 1) > 1
            if MIRROR_CHOICES[wm.v5_hand_mirror] is not None:
                control["set_mirror"](MIRROR_CHOICES[wm.v5_hand_mirror], 0 if two else None)
            if two and MIRROR_CHOICES[wm.v5_hand_mirror2] is not None:
                control["set_mirror"](MIRROR_CHOICES[wm.v5_hand_mirror2], 1)
        return {"FINISHED"}


class V5HAND_OT_stop(bpy.types.Operator):
    bl_idname = "v5hand.stop"
    bl_label = "Stop receiver"
    bl_description = "Stop listening and put the armature back where it was"

    def execute(self, context):
        control = _control()
        if control is None:
            return {"CANCELLED"}
        control["stop"]()
        return {"FINISHED"}


class V5HAND_PT_panel(bpy.types.Panel):
    bl_idname = "V5HAND_PT_panel"
    bl_label = "Hand tracking"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Hand"

    def draw(self, context):
        layout = self.layout
        status = bpy.app.driver_namespace.get(STATUS_KEY) or {}
        running = bool(status.get("running"))
        if running:
            layout.operator("v5hand.stop", icon="PAUSE")
        else:
            layout.operator("v5hand.start", icon="PLAY")
        hands = status.get("hands") or [status]
        for number, hand in enumerate(hands, 1):
            box = layout.box()
            prefix = f"{hand.get('name', f'Hand {number}')}: " if len(hands) > 1 else "Status: "
            box.label(text=f"{prefix}{hand.get('text', 'not started')}")
            if running:
                if number == 1:
                    box.label(text=f"Listening on port {status.get('port')}")
                if hand.get("fps") is not None:
                    box.label(text=f"{hand['fps']:.0f} frames/s, latency {hand['latency_ms']:.0f} ms")
                mirror = hand.get("mirror")
                shape = "not decided yet" if mirror is None else ("mirrored" if mirror else "not mirrored")
                how = "forced" if hand.get("override") is not None else (
                    "from the finger bends" if hand.get("shape_confirmed") else "first guess")
                box.label(text=f"Rig {shape} ({how})")
        if len(hands) > 1:
            names = [h.get("name", f"Hand {i}") for i, h in enumerate(hands, 1)]
            layout.prop(context.window_manager, "v5_hand_mirror", text=f"{names[0]} mirror")
            layout.prop(context.window_manager, "v5_hand_mirror2", text=f"{names[1]} mirror")
        else:
            layout.prop(context.window_manager, "v5_hand_mirror", text="Mirror")


CLASSES = (V5HAND_OT_start, V5HAND_OT_stop, V5HAND_PT_panel)


def _redraw():
    wm = bpy.context.window_manager
    for window in (wm.windows if wm else ()):
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    return REDRAW_SECONDS


def unregister():
    ns = bpy.app.driver_namespace
    old = ns.pop(REDRAW_KEY, None)
    if old is not None and bpy.app.timers.is_registered(old):
        bpy.app.timers.unregister(old)
    for cls in ns.pop(CLASSES_KEY, ()):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    for name in ("v5_hand_mirror", "v5_hand_mirror2"):
        if hasattr(bpy.types.WindowManager, name):
            delattr(bpy.types.WindowManager, name)


def register():
    ns = bpy.app.driver_namespace
    if ns.get(SKIP_KEY) and ns.get(CLASSES_KEY):
        return  # the Start button re-ran the receiver; the panel it clicked in stays as it is
    unregister()
    bpy.types.WindowManager.v5_hand_mirror = bpy.props.EnumProperty(
        name="Mirror",
        description="Whether the rig is mirrored to match your hand",
        items=[("AUTO", "Automatic", "Decide from the finger bends"),
               ("MIRRORED", "Mirrored", "Always mirror the rig"),
               ("NOT_MIRRORED", "Not mirrored", "Never mirror the rig")],
        default="AUTO", update=_mirror_changed)
    bpy.types.WindowManager.v5_hand_mirror2 = bpy.props.EnumProperty(
        name="Mirror (hand 2)",
        description="Whether the second rig (Armature.001) is mirrored to match the hand that drives it",
        items=[("AUTO", "Automatic", "Decide from the finger bends"),
               ("MIRRORED", "Mirrored", "Always mirror the rig"),
               ("NOT_MIRRORED", "Not mirrored", "Never mirror the rig")],
        default="AUTO", update=_mirror2_changed)
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    ns[CLASSES_KEY] = CLASSES
    ns[REDRAW_KEY] = _redraw
    bpy.app.timers.register(_redraw, persistent=True)


register()
