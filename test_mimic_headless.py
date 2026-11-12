"""How closely does the rig mimic a hand that is NOT built like the rig? Run headless on the
fixed copy (limits and rolls already applied):

    blender -b --disable-autoexec v5/untitled_rig_fixed.blend --python v5/test_mimic_headless.py -- v5

The synthetic user hand has its own finger proportions, per-frame segment-length noise (as
MediaPipe's world landmarks have), a random palm orientation each frame, finger bends inside
the anatomical limits, and either hand shape. Two numbers matter:
  floating  distance between each empty and the tip of the bone that should reach it
  mimic     angle between each rig bone and the same segment of the source hand
Nothing is saved.
"""
import math
import random
import socket
import sys
import time

V5_DIR = sys.argv[sys.argv.index("--") + 1]
sys.path.insert(0, V5_DIR)

import bpy
from mathutils import Matrix, Vector

import blender_receiver as br
import protocol
import rig_tools as rt

arm = bpy.data.objects[rt.ARMATURE_NAME]
bones = arm.data.bones
MW = arm.matrix_world.copy()
THUMB = ("wrist_to_thumb_cmc", "thumb_cmc_to_thumb_mcp", "thumb_mcp_to_thumb_ip", "thumb_ip_to_thumb_tip")
CHAINS = {"thumb": THUMB, **rt.FINGERS}
K = 4.0
TILT_DEG = 25.0  # how far a finger's bend plane may be tilted from the rig's
failures = []


def check(name, ok, detail=""):
    print(f"RESULT {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


def evaluated():
    bpy.context.view_layer.update()
    return arm.evaluated_get(bpy.context.evaluated_depsgraph_get())


rest_flex = {f: [math.degrees(rt._signed_angle(rt.finger_hinge(bones, n), rt._dir(bones[n[i]]),
                                               rt._dir(bones[n[i + 1]]))) for i in range(3)]
             for f, n in rt.FINGERS.items()}


def random_rotation(rng):
    axis = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))).normalized()
    return Matrix.Rotation(math.radians(rng.uniform(0, 160)), 3, axis)


def make_user_hand(rng, abs_flex, props, noise, thumb_jitter):
    """Landmarks (armature-local) of a hand with the rig's joint layout but its own bone lengths,
    plus the true direction of every segment."""
    pts = {"WRIST": Vector((0, 0, 0))}
    src = {}
    for finger, names in CHAINS.items():
        d = [rt._dir(bones[n]) for n in names]
        if finger == "thumb":
            dirs = []
            for v in d:
                axis = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))).normalized()
                dirs.append(Matrix.Rotation(math.radians(thumb_jitter[len(dirs)]), 3, axis) @ v)
        else:
            # A real finger: its proximal, middle and distal segments lie in one flat plane, and that
            # plane is tilted away from the rig's hinge plane by up to TILT_DEG (fingers curl toward
            # the thumb). The palm segment is the rig's, so it is abducted out of that plane.
            h = rt.finger_hinge(bones, names)
            normal = (Matrix.Rotation(math.radians(rng.uniform(-TILT_DEG, TILT_DEG)), 3, d[0]) @ h).normalized()
            u = (d[0] - normal * d[0].dot(normal)).normalized()
            w = normal.cross(u)
            dirs, cum = [d[0]], 0.0
            for i in range(1, 4):
                cum += math.radians(abs_flex[finger][i - 1])
                dirs.append((u * math.cos(cum) + w * math.sin(cum)).normalized())
        pos = Vector((0, 0, 0))
        for i, n in enumerate(names):
            length = bones[n].length * props[finger][i] * (1 + noise * rng.gauss(0, 1))
            pos = pos + dirs[i] * length
            pts[rt.BONE_TARGETS[n]] = pos.copy()
            src[n] = dirs[i]
    return pts, src


def to_mp(world, centre=Vector((0.01, 0.02, -0.03))):
    return [(world[n].x / K + centre.x, -world[n].z / K + centre.y, world[n].y / K + centre.z)
            for n in br.LANDMARK_NAMES]


def reflected(points):
    frame = br.palm_frame(tuple(points["WRIST"]), tuple(points["INDEX_FINGER_MCP"]),
                          tuple(points["MIDDLE_FINGER_MCP"]), tuple(points["PINKY_MCP"]))
    x = Vector(frame[0])
    return {k: v - 2 * v.dot(x) * x for k, v in points.items()}


PER_BONE = {}


def measure(ev, src):
    float_mu, angle_deg = [], []
    for bone, target in rt.BONE_TARGETS.items():
        pb = ev.pose.bones[bone]
        tail_w = ev.matrix_world @ pb.tail
        float_mu.append((tail_w - bpy.data.objects[target].matrix_world.translation).length * 1000)
        d = (pb.tail - pb.head).normalized()
        angle_deg.append(math.degrees(d.angle(src[bone], 0.0)))
        worst = PER_BONE.setdefault(bone, [0.0, 0.0, 0])
        worst[0] = max(worst[0], float_mu[-1])
        worst[1] = max(worst[1], angle_deg[-1])
        worst[2] += angle_deg[-1] > 3.0
    return float_mu, angle_deg


def pip_flexions(ev):
    """Bend at each finger's PIP in degrees: the plain angle between the proximal and middle bones,
    which for a flat finger equals its joint flexion whatever way the plane is tilted."""
    out = []
    for names in rt.FINGERS.values():
        d = [((ev.pose.bones[n].tail - ev.pose.bones[n].head).normalized()) for n in names]
        out.append(math.degrees(d[1].angle(d[2], 0.0)))
    return out


def dip_flexions(ev):
    out = []
    for names in rt.FINGERS.values():
        d = [((ev.pose.bones[n].tail - ev.pose.bones[n].head).normalized()) for n in names]
        out.append(math.degrees(d[2].angle(d[3], 0.0)))
    return out


def run(label, opposite_shape, frames=40, seed=11, flex_overshoot=0.0, straight_noise=None):
    """straight_noise: finger bends are drawn from 0..this many degrees instead of the curl ranges."""
    rng = random.Random(seed)
    props = {f: [rng.uniform(0.85, 1.25) for _ in range(4)] for f in CHAINS}
    palm_factor = rng.uniform(0.9, 1.15)
    for f in props:
        props[f][0] *= palm_factor
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    st = br.State()
    st.rig = br.Rig(arm)
    sid = f"mimic{seed}{int(opposite_shape)}"
    all_float, all_angle = [], []
    for seq in range(1, frames + 1):
        if straight_noise is None:
            abs_flex = {f: (rng.uniform(-15, 80) + flex_overshoot, rng.uniform(5, 105) + flex_overshoot,
                            rng.uniform(5, 85) + flex_overshoot) for f in rt.FINGERS}
        else:
            abs_flex = {f: (rng.uniform(-3, 3), rng.uniform(0, straight_noise), rng.uniform(0, straight_noise))
                        for f in rt.FINGERS}
        # the palm is rigid (RIGID_PALM), so the thumb's wrist-to-base segment is not jittered
        local, src = make_user_hand(rng, abs_flex, props, 0.08, [0.0] + [rng.uniform(0, 20) for _ in range(3)])
        world = {k: MW @ v for k, v in local.items()}
        if opposite_shape:
            world = reflected(world)
        q = random_rotation(rng)
        world = {k: q @ v for k, v in world.items()}
        tx.sendto(protocol.encode_frame(sid, seq, time.time(), label, to_mp(world)), sock.getsockname())
        time.sleep(0.002)
        br.process(sock, st)
        if seq > 5:  # first frames still calibrate the scale
            ev = evaluated()
            f, a = measure(ev, src)
            all_float += f
            all_angle += a
            RUN_MAX_PIP[0] = max(RUN_MAX_PIP[0], max(pip_flexions(ev)))
            RUN_MAX_DIP[0] = max(RUN_MAX_DIP[0], max(dip_flexions(ev)))
    br.restore_armature(st)
    sock.close()
    tx.close()
    return all_float, all_angle


RUN_MAX_PIP = [-999.0]
RUN_MAX_DIP = [-999.0]


def stats(v):
    return sum(v) / len(v), max(v)


if __name__ == "__main__":
    if "nolimits" in sys.argv:  # diagnostic: is any remaining error caused by the IK limits?
        rt.clear_joint_limits(arm)
        print("DIAGNOSTIC: IK limits cleared in memory")
    for arg in sys.argv:  # diagnostic: re-plan the IK limits with a given slack, in memory
        if arg.startswith("slack="):
            rt.IK_SLACK_DEG = float(arg.split("=")[1])
            rt.clear_joint_limits(arm)
            rt.apply_joint_limits(arm, rt.plan_joint_limits(arm)[0])
            print(f"DIAGNOSTIC: IK limits re-planned with slack {rt.IK_SLACK_DEG} deg")
    results = {}
    # The exact-direction checks below run without the straight dead zone, which deliberately
    # changes small PIP/DIP bends; the dead zone has its own scenarios at the end.
    straight_zone = dict(br.STRAIGHT_ZONE_DEG)
    br.STRAIGHT_ZONE_DEG.clear()
    collision = br.COLLISION and "nocollision" not in sys.argv  # diagnostic: nocollision
    # exact source directions: collision moves overlapping fingers on purpose
    br.COLLISION = False
    for name, label, opposite in (("same_shape", "Left", False), ("opposite_shape", "Right", True)):
        fl, an = run(label, opposite)
        results[name] = (stats(fl), stats(an))
        print(f"MIMIC {name:15s} floating mean {stats(fl)[0]:6.2f} max {stats(fl)[1]:7.2f} milli-units | "
              f"direction error mean {stats(an)[0]:5.2f} max {stats(an)[1]:6.2f} deg")
    for bone, (f_max, a_max, n_bad) in sorted(PER_BONE.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"WORST {bone:28s} floating max {f_max:7.2f}  direction max {a_max:6.2f} deg  "
              f"frames over 3 deg: {n_bad}")
    for name, ((f_mean, f_max), (a_mean, a_max)) in results.items():
        check(f"{name}_empties_attached_to_bone_tips", f_mean < 0.5 and f_max < 3.0,
              f"mean {f_mean:.2f} max {f_max:.2f}")
        check(f"{name}_bones_mimic_source_directions", a_mean < 1.0 and a_max < 5.0,
              f"mean {a_mean:.2f} max {a_max:.2f}")
    # fingers pushed past their limits: the receiver stops them at the anatomical limit and the
    # empties stay on the bone tips (the pose differs from the source only where it must)
    RUN_MAX_PIP[0] = -999.0
    fl, an = run("Left", False, seed=23, flex_overshoot=30.0)
    f_mean, f_max = stats(fl)
    print(f"OVERSHOOT floating mean {f_mean:.2f} max {f_max:.2f} milli-units | "
          f"largest PIP flexion reached {RUN_MAX_PIP[0]:.1f} deg (limit 110)")
    check("overshoot_empties_stay_attached", f_mean < 0.5 and f_max < 3.0, f"mean {f_mean:.2f} max {f_max:.2f}")
    check("overshoot_stops_at_the_110_degree_limit", 100.0 < RUN_MAX_PIP[0] < 111.0, f"{RUN_MAX_PIP[0]:.1f}")

    # straight fingers: every PIP/DIP of a hand held flat reads up to 5 degrees bent (as MediaPipe's
    # does); with the dead zone the rig's fingers are straight, and the empties are still reached
    br.STRAIGHT_ZONE_DEG.update(straight_zone)
    # collision stays off for the dead-zone scenarios: it may nudge a finger the thumb overlaps, and has
    # its own tests (test_collision_headless.py)
    for name, label, opposite in (("same_shape", "Left", False), ("opposite_shape", "Right", True)):
        RUN_MAX_PIP[0] = RUN_MAX_DIP[0] = -999.0
        fl, an = run(label, opposite, seed=31, straight_noise=5.0)
        print(f"STRAIGHT {name}: largest rig PIP bend {RUN_MAX_PIP[0]:.2f} deg, DIP {RUN_MAX_DIP[0]:.2f} deg, "
              f"floating max {max(fl):.2f}")
        check(f"straight_{name}_fingers_stay_straight",
              RUN_MAX_PIP[0] < 1.0 and RUN_MAX_DIP[0] < 1.0, f"pip {RUN_MAX_PIP[0]:.2f} dip {RUN_MAX_DIP[0]:.2f}")
        check(f"straight_{name}_empties_attached", max(fl) < 3.0, f"max {max(fl):.2f}")
    RUN_MAX_PIP[0] = -999.0
    run("Left", False, seed=32, straight_noise=60.0)  # real curls are not flattened
    check("curled_fingers_still_bend_with_dead_zone", RUN_MAX_PIP[0] > 25.0, f"{RUN_MAX_PIP[0]:.1f}")
    br.COLLISION = collision
    print("FAILURES:", failures or "none")
    sys.exit(1 if failures else 0)
