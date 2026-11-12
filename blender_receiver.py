"""Blender-side receiver for v5/sender.py. Paste into a Text block and Run Script.

What it does each tick (every TIMER_INTERVAL seconds):
  1. drains the UDP socket and keeps only the newest valid packet
  2. converts MediaPipe world landmarks (metres) to Blender axes, relative to the wrist
  3. scales by one factor calibrated once per sender session (not per frame)
  4. moves the existing landmark empties (WRIST, THUMB_CMC, ... PINKY_TIP)
  5. TRACK_PALM: rotates the Armature object to match the live palm orientation, so the
     per-bone IK and the finger joint limits always see the hand in the rig's own frame;
     if the hand's shape is the opposite of the rig's, the rig is mirrored to match
  6. optionally (TRANSLATE_HAND) also moves the hand through space

Smoothing and outlier rejection are done in the sender. This script does not change
the IK setup and does not switch Blender modes. Re-running it stops the previous
instance first (and puts the Armature back), so sockets and timers don't leak.
"""
import json
import math
import os
import socket
import statistics
import time

try:
    import bpy
except ImportError:  # imported outside Blender (tests)
    bpy = None

# ---- configuration ---------------------------------------------------------
HOST = "127.0.0.1"
PORT = 5004                       # must match sender.py --port
TIMER_INTERVAL = 1.0 / 60.0       # was 0.1 s; the socket is drained fully each tick
REFERENCE_DISTANCE = 0.3914       # fallback only: with a rig the scale comes from the armature's own
                                  # WRIST -> MIDDLE_FINGER_MCP bone length (Rig.span), so a skeleton
                                  # of any size works; this is used when no armature is available
FIXED_WORLD_SCALE = None          # set a number to skip calibration
CALIBRATION_FRAMES = 30           # frames whose median hand span fixes the scale
MAX_FRAME_AGE_S = 0.5             # hold instead of applying frames older than this; None = off
NO_DATA_WARN_SECONDS = 1.0
STATS_SECONDS = 5.0               # 0 disables the periodic stats line
STOP_ON_SENDER_STOP = False       # False: keep listening so the sender can be restarted

# Palm orientation. The armature object is rotated to follow the palm, which keeps the finger
# joint limits valid when the hand turns (palm to camera, back of hand, rolled, tilted).
TRACK_PALM = True
ARMATURE_NAME = "Armature"
PALM_BONES = {"index": "wrist_to_index_mcp", "middle": "wrist_to_middle_mcp",
              "pinky": "wrist_to_pinky_mcp"}
RIG_SHAPE = "Right"               # the rig is a right-hand shape (thumb on the +X side when its
                                  # palm faces -Y, the camera side)
# MediaPipe reports handedness as if the image were a mirror view, so the *shape* of the hand
# it measured is the opposite of its label. When that shape differs from RIG_SHAPE the rig is
# mirrored. Set to True or False to force it if the rig looks wrong for your hand.
MIRROR_OVERRIDE = None
# Two hands. Each hand the sender reports has a slot (0, 1). Slot 0 drives the Armature and the
# landmark empties named as above; slot 1 drives the objects with the second suffix appended (the
# copies Blender makes: "Armature.001", "WRIST.001", ...; v5/build_two_hands.py makes them). A slot
# with no armature in the scene is not used, so a one-hand file works exactly as before.
HAND_SUFFIXES = ("", ".001")
HAND_NAMES = ("Left hand", "Right hand")   # slot 0 is the person's left hand, slot 1 the right (see slots.py)
LABEL_VOTE_FRAMES = 15            # the label must disagree for a while before the rig flips

# Retargeting. A bone has a fixed length and can only rotate, but a tracked hand has different
# proportions and its measured segment lengths wobble from frame to frame. Copying landmark
# positions therefore leaves empties floating out of reach. With RETARGET each empty is placed
# where its bone tip can actually be: the live DIRECTION of every segment, the rig's own bone
# LENGTH. With CLAMP_TO_LIMITS the finger joint limits are applied to those directions too, so
# the IK limits never leave a target out of reach. Keep the numbers below equal to rig_tools.py.
RETARGET = True
CLAMP_TO_LIMITS = True
FLEXION_RANGE = {"mcp": (-20.0, 90.0), "pip": (0.0, 110.0), "dip": (0.0, 90.0)}  # degrees
MCP_ABDUCTION_DEG = 30.0
SIDE_MARGIN_DEG = 5.0
# MediaPipe never reports a straight finger as perfectly straight: each joint of a finger held
# flat reads about 5 degrees bent, so the fingertip ends up visibly curled. A PIP or DIP bend
# below the first number is shown as 0 (straight); between the two numbers it fades in smoothly,
# and above the second it passes through unchanged. Remove a joint, or set (0, 0), to switch off.
STRAIGHT_ZONE_DEG = {"pip": (6.0, 20.0), "dip": (6.0, 20.0)}
# A real palm is one stiff piece, and the mesh skinned to the five wrist-to-knuckle bones tears and
# wobbles if they follow the landmark noise of the palm. With RIGID_PALM those five bones (and the
# wrist-to-thumb-base bone) keep their rest direction in the palm's frame; the fingers still follow
# the live finger directions, measured from the knuckle the rig's palm puts them at.
RIGID_PALM = True
# Finger collision. Every finger segment is a capsule as thick as the mesh around its bone (the
# skin_mesh.py tool stores that on each bone as "v5_radius"; without it a fraction of the segment
# length is used). Neighbouring fingers (and the thumb and index) may not overlap more than they do
# in the rig's rest pose, and fingertips may not sink into the palm: overlapping fingers are pushed
# apart and the joint limits applied again. COLLISION_MARGIN lets soft fingers press together a
# little (1.0 = touching exactly).
COLLISION = True
PALM_COLLISION = True
COLLISION_MARGIN = 0.7
COLLISION_ITERATIONS = 4
RADIUS_KEY = "v5_radius"
DEFAULT_RADIUS = (0.1, 0.23, 0.37, 0.22)   # of the segment length: palm, proximal, middle, distal
PALM_THICKNESS = 0.9                       # palm surface height above the palm bones, in finger radii
COLLISION_PAIRS = (("index", "middle"), ("middle", "ring"), ("ring", "pinky"), ("thumb", "index"))
# Thumb: its first joint (CMC) moves in every direction and is left alone; the knuckle (MCP) and
# tip joint (IP) bend at most this far (degrees between the two segments they join).
THUMB_BEND_MAX_DEG = {"mcp": 70.0, "ip": 90.0}

# When the sender reports no hand, the rig holds its last pose for LOST_HOLD_S seconds and then
# relaxes to its rest pose over LOST_EASE_S seconds (the hand keeps its last position and
# orientation). Set LOST_EASE_S = 0 to hold the last pose until the hand comes back.
LOST_HOLD_S = 0.5
LOST_EASE_S = 1.0
STATUS_KEY = "v5_hand_status"     # live status for the Blender panel (blender_panel.py)
CONTROL_KEY = "v5_hand_control"   # what the panel may call: stop, set_mirror

# No mirror. The sender flips the camera picture like a mirror before MediaPipe sees it (that is what
# MediaPipe's left/right labels expect), so the landmarks that arrive are a MIRROR IMAGE of the hand:
# left and right are right, but depth is reversed, and a right hand arrives with a left hand's shape.
# With UNMIRROR_HAND the receiver reflects the depth (Blender's Y) back, so Blender shows the hand as it
# is: you and Blender's front view both look along +Y; hold your palm toward the camera (forward, +Y)
# and the palm faces +Y in Blender too, a right hand stays a right hand and moving it to your right
# moves it to the right. Set it to False to get the old mirror view.
UNMIRROR_HAND = True
# Which way the hand faces. Every hand is turned by this many degrees about its OWN long axis (through
# the wrist toward the middle knuckle; with the fingers up that is the vertical Z axis): 180 = palm and back
# swapped, so a palm held toward the webcam shows its palm to Blender's front view. Only the orientation
# turns; where the hand sits does not. Because the turn is about the hand's own axis, every movement keeps
# its direction: tilt the hand right and Blender's hand tilts right, push forward and it goes forward
# (the same hand turned about the fixed world Z axis would send every sideways tilt the other way).
# HAND_TURN_ABOUT = "world" turns about the world's vertical axis instead. 0 = no turn.
HAND_YAW_DEG = 180.0
HAND_TURN_ABOUT = "hand"      # "hand" (its own long axis) or "world" (the vertical Z axis)

# Global hand movement. Off by default: the wrist then stays at the world origin, which is
# where the rig's root bones are. When on, the Armature object and all empties move together.
TRANSLATE_HAND = False
TRANSLATE_GAIN = (1.0, 1.0, 1.0)  # per Blender axis; set an axis to 0 to freeze it
                                  # (depth, the Y axis, is the least accurate)
# ----------------------------------------------------------------------------

PROTOCOL_VERSION = 1
ACK = b"ack"
LANDMARK_NAMES = (
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
)
# bones of each chain, root (wrist to knuckle) first, and the landmarks they join
BONE_CHAINS = {
    "thumb": ("wrist_to_thumb_cmc", "thumb_cmc_to_thumb_mcp", "thumb_mcp_to_thumb_ip",
              "thumb_ip_to_thumb_tip"),
    "index": ("wrist_to_index_mcp", "index_mcp_to_index_pip", "index_pip_to_index_dip",
              "index_dip_to_index_tip"),
    "middle": ("wrist_to_middle_mcp", "middle_mcp_to_middle_pip", "middle_pip_to_middle_dip",
               "middle_dip_to_middle_tip"),
    "ring": ("wrist_to_ring_mcp", "ring_mcp_to_ring_pip", "ring_pip_to_ring_dip",
             "ring_dip_to_ring_tip"),
    "pinky": ("wrist_to_pinky_mcp", "pinky_mcp_to_pinky_pip", "pinky_pip_to_pinky_dip",
              "pinky_dip_to_pinky_tip"),
}
CHAIN_LANDMARKS = {"thumb": (0, 1, 2, 3, 4), "index": (0, 5, 6, 7, 8), "middle": (0, 9, 10, 11, 12),
                   "ring": (0, 13, 14, 15, 16), "pinky": (0, 17, 18, 19, 20)}
_JOINTS = ("mcp", "pip", "dip")
FINGERS4 = ("index", "middle", "ring", "pinky")
# How strongly the rig's hinge steers the bend plane of a nearly straight finger (a bent finger
# uses its own plane), and how the hand's shape is decided from the finger bends.
PLANE_BLEND = 0.2
EVIDENCE_MIN = 0.2       # mean bend-axis size below which a frame says nothing about the shape
EVIDENCE_FULL = 0.5      # ... and above which it counts fully
EVIDENCE_SMOOTH = 0.4    # how fast the shape score follows new evidence
_WRIST = LANDMARK_NAMES.index("WRIST")
_INDEX_MCP = LANDMARK_NAMES.index("INDEX_FINGER_MCP")
_MIDDLE_MCP = LANDMARK_NAMES.index("MIDDLE_FINGER_MCP")
_PINKY_MCP = LANDMARK_NAMES.index("PINKY_MCP")
_STOP_KEY = "hand_receiver_v5_stop"
_ZERO = (0.0, 0.0, 0.0)
_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# ---- small vector / 3x3 helpers (pure Python so they can be tested outside Blender) ----
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(v):
    n = math.sqrt(_dot(v, v))
    return None if n < 1e-9 else (v[0] / n, v[1] / n, v[2] / n)


def _matvec(m, v):
    return (_dot(m[0], v), _dot(m[1], v), _dot(m[2], v))


def _matmul(a, b):
    cols = list(zip(*b))
    return tuple(tuple(_dot(row, col) for col in cols) for row in a)


def _inv3(m):
    c0, c1, c2 = _cross(m[1], m[2]), _cross(m[2], m[0]), _cross(m[0], m[1])
    det = _dot(m[0], c0)
    return tuple(tuple(c[i] / det for c in (c0, c1, c2)) for i in range(3))


def palm_frame(wrist, index_mcp, middle_mcp, pinky_mcp):
    """Right-handed orthonormal palm frame (x across the palm toward the index finger,
    y along the fingers, z = x cross y), or None if the points are degenerate."""
    y = _unit(_sub(middle_mcp, wrist))
    if y is None:
        return None
    across = _sub(index_mcp, pinky_mcp)
    x = _unit(_sub(across, tuple(c * _dot(across, y) for c in y)))
    if x is None:
        return None
    return (x, y, _cross(x, y))


def rotation_between(rig_frame, live_frame):
    """3x3 rotation R with R @ rig_axis_k = live_axis_k for the three palm axes."""
    return tuple(tuple(sum(live_frame[k][i] * rig_frame[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value, low, high):
    return max(low, min(high, value))


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _any_perpendicular(n):
    return _unit(_cross(n, (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)))


def _straighten(flexion, zone):
    """Flexion (radians, 0 = straight) with the straight dead zone applied: 0 up to zone[0], fading
    in to the unchanged value at zone[1]. The factor rises smoothly from 0 to 1, so the result
    never decreases as the finger bends further."""
    lo, hi = math.radians(zone[0]), math.radians(zone[1])
    if flexion <= lo:
        return 0.0
    if flexion >= hi:
        return flexion
    t = (flexion - lo) / (hi - lo)
    return flexion * t * t * (3.0 - 2.0 * t)


def _bend_axis(dirs):
    """Sum of the joint bend axes between consecutive segment directions. Rotating about it bends
    each segment toward the next, so it points the same way for every finger that curls."""
    total = _ZERO
    for a, b in zip(dirs, dirs[1:]):
        total = _add(total, _cross(a, b))
    return total


def _segment_dirs(points):
    """Unit directions of the 4 segments between 5 chain points, or None if one is degenerate."""
    dirs = []
    for s in range(4):
        v = _unit(_sub(points[s + 1], points[s]))
        if v is None:
            return None
        dirs.append(v)
    return dirs


class Chain:
    """One wrist-to-fingertip chain of the rig, in armature-local axes: bone lengths, rest
    directions and (for the four fingers) the hinge axis that fixes which way the finger curls."""

    def __init__(self, finger, bones, names):
        segs = []
        for n in names:
            vec = _sub(tuple(bones[n].tail_local), tuple(bones[n].head_local))
            segs.append((math.sqrt(_dot(vec, vec)), _unit(vec)))
        self.lengths = [s[0] for s in segs]
        self.rest_dirs = [s[1] for s in segs]
        self.is_thumb = finger == "thumb"
        self.radii = [_bone_radius(bones[n], i, self.lengths[i]) for i, n in enumerate(names)]
        self.hinge = None
        if finger != "thumb":
            d = self.rest_dirs
            # a straight rest pose has no bend to read the hinge from; then no limits apply
            self.hinge = _unit(_add(_cross(d[1], d[2]), _cross(d[2], d[3])))

    def limit(self, dirs, straighten=True):
        """Apply the anatomical joint limits to the 4 live segment directions (straighten: also the
        dead zone that shows a nearly straight joint as straight).

        The bend plane is the live finger's own (its bend axis), oriented like the rig's hinge so
        a bend toward the palm counts as flexion and a bend backwards as hyperextension. A nearly
        straight finger has no bend axis of its own and uses the rig's hinge. Each joint's
        flexion (0 = straight) and its swing out of the plane are clamped to their ranges, and the
        directions are rebuilt joint by joint, so a clamped joint carries on into the next."""
        b = _bend_axis(dirs[1:])  # the finger's own joints; the knuckle also depends on the palm
        side = 1.0 if _dot(b, self.hinge) >= 0.0 else -1.0
        n = _unit(_add(_scale(b, side), _scale(self.hinge, PLANE_BLEND)))
        ref0 = _unit(_sub(dirs[0], _scale(n, _dot(dirs[0], n))))
        if ref0 is None:  # the palm segment points along the bend axis: any reference will do
            ref0 = _any_perpendicular(n)
        ref1 = _cross(n, ref0)
        angles = [(math.atan2(_dot(v, ref1), _dot(v, ref0)), math.asin(_clamp(_dot(v, n), -1.0, 1.0)))
                  for v in dirs]
        flex, swing = angles[0]
        out = [dirs[0]]
        for s in (1, 2, 3):
            joint = _JOINTS[s - 1]
            lo, hi = FLEXION_RANGE[joint]
            side_limit = math.radians(MCP_ABDUCTION_DEG if s == 1 else SIDE_MARGIN_DEG)
            bend = _clamp(_wrap(angles[s][0] - angles[s - 1][0]), math.radians(lo), math.radians(hi))
            out_of_plane = _clamp(angles[s][1] - angles[s - 1][1], -side_limit, side_limit)
            if straighten and joint in STRAIGHT_ZONE_DEG:  # the joint's whole bend, in the plane and out of it
                total = math.hypot(bend, out_of_plane)
                if total > 0.0:
                    k = _straighten(total, STRAIGHT_ZONE_DEG[joint]) / total
                    bend, out_of_plane = bend * k, out_of_plane * k
            flex += bend
            swing += out_of_plane
            cs = math.cos(swing)
            out.append(tuple(cs * (math.cos(flex) * a + math.sin(flex) * c) + math.sin(swing) * h
                             for a, c, h in zip(ref0, ref1, n)))
        return out

    def limit_thumb(self, dirs):
        """Keep the thumb's knuckle (MCP) and tip joint (IP) inside THUMB_BEND_MAX_DEG: a segment
        bent further from the previous one is swung back along the same direction of bend."""
        out = [dirs[0], dirs[1]]
        for s, joint in ((2, "mcp"), (3, "ip")):
            prev, v = out[s - 1], dirs[s]
            top = math.radians(THUMB_BEND_MAX_DEG[joint])
            cos = _clamp(_dot(prev, v), -1.0, 1.0)
            if math.acos(cos) > top:
                side = _unit(_sub(v, _scale(prev, cos))) or _any_perpendicular(prev)
                v = tuple(math.cos(top) * p + math.sin(top) * q for p, q in zip(prev, side))
            out.append(v)
        return out

    def retarget(self, points, apply_limits):
        """points: the chain's 5 live landmarks (armature-local, wrist first). Returns the 5
        landmark positions the rig can reach: live directions, rig lengths, joints limited."""
        dirs = []
        for s in range(4):
            v = _unit(_sub(points[s + 1], points[s]))
            dirs.append(v if v is not None else self.rest_dirs[s])
        if RIGID_PALM:
            dirs[0] = self.rest_dirs[0]
        if apply_limits and self.hinge is not None:
            dirs = self.limit(dirs)
        elif apply_limits and self.is_thumb:
            dirs = self.limit_thumb(dirs)
        out = [points[0]]
        for s in range(4):
            out.append(tuple(p + self.lengths[s] * c for p, c in zip(out[-1], dirs[s])))
        return out


def bend_evidence(points, lateral):
    """Mean over the four fingers of (bend axis dot the palm's across axis). Its sign tells which
    way the fingers curl relative to the palm, i.e. the hand's shape (left or right); its size
    tells how bent the fingers are, so a flat hand gives about zero."""
    total, count = 0.0, 0
    for finger in FINGERS4:
        dirs = _segment_dirs([points[i] for i in CHAIN_LANDMARKS[finger]])
        if dirs is not None:
            total += _dot(_bend_axis(dirs), lateral)
            count += 1
    return total / count if count else 0.0


def _bone_radius(bone, index, length):
    """Thickness of the skin around a bone: the "v5_radius" stored on it by skin_mesh.py, else a
    fraction of its length."""
    getter = getattr(bone, "get", None)
    value = getter(RADIUS_KEY) if callable(getter) else None
    try:
        if value is not None and float(value) > 0.0:
            return float(value)
    except (TypeError, ValueError):
        pass
    return DEFAULT_RADIUS[index] * length


def _closest_segments(p1, q1, p2, q2):
    """Closest points of the segments p1-q1 and p2-q2: (s, t, point on the first, point on the second),
    s and t being how far along each (0..1)."""
    d1, d2, r = _sub(q1, p1), _sub(q2, p2), _sub(p1, p2)
    a, e, f = _dot(d1, d1), _dot(d2, d2), _dot(d2, r)
    if a <= 1e-12 and e <= 1e-12:
        s = t = 0.0
    elif a <= 1e-12:
        s, t = 0.0, _clamp(f / e, 0.0, 1.0)
    else:
        c = _dot(d1, r)
        if e <= 1e-12:
            s, t = _clamp(-c / a, 0.0, 1.0), 0.0
        else:
            b = _dot(d1, d2)
            denom = a * e - b * b
            s = _clamp((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, _clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, _clamp((b - c) / a, 0.0, 1.0)
    return s, t, _add(p1, _scale(d1, s)), _add(p2, _scale(d2, t))


class Collision:
    """What finger collision needs from the rig's rest pose: for each neighbouring finger pair and
    each pair of their segments, the radii and the distance they have at rest (the rest pose is the
    contact the mesh was made with, so it never counts as an overlap), and the palm's plane."""

    def __init__(self, chains):
        rest = {}
        for finger, chain in chains.items():
            pts = [_ZERO]
            for length, d in zip(chain.lengths, chain.rest_dirs):
                pts.append(_add(pts[-1], _scale(d, length)))
            rest[finger] = pts
        self.pairs, self.axis = {}, {}
        for fa, fb in COLLISION_PAIRS:
            entries = []
            for sa in (1, 2, 3):
                for sb in (1, 2, 3):
                    _, _, c1, c2 = _closest_segments(rest[fa][sa], rest[fa][sa + 1], rest[fb][sb], rest[fb][sb + 1])
                    gap = math.sqrt(_dot(_sub(c1, c2), _sub(c1, c2)))
                    entries.append((sa, sb, chains[fa].radii[sa] + chains[fb].radii[sb], gap))
            self.pairs[(fa, fb)] = entries
            self.axis[(fa, fb)] = _unit(_sub(rest[fa][1], rest[fb][1])) or (1.0, 0.0, 0.0)
        # The palm: the plane through the wrist and the knuckle row, with its normal pointing the way
        # the fingers curl (the way a fingertip moves when it flexes: hinge x direction). The palm's
        # skin is PALM_THICKNESS finger radii above it, between the wrist and the knuckle line.
        self.palm = None
        if all(chains[f].hinge is not None for f in FINGERS4):
            n = _unit(_cross(_sub(rest["index"][1], rest["pinky"][1]), rest["middle"][1]))
            curl = tuple(sum(_cross(chains[f].hinge, chains[f].rest_dirs[1])[k] for f in FINGERS4)
                         for k in range(3))
            if n is not None and _dot(n, curl) < 0.0:
                n = _scale(n, -1.0)
            mean_mcp = tuple(sum(rest[f][1][k] for f in FINGERS4) / 4.0 for k in range(3))
            u = n and _unit(_sub(mean_mcp, _scale(n, _dot(mean_mcp, n))))
            if n is not None and u is not None:
                lat = _cross(n, u)
                lats = [_dot(rest[f][1], lat) for f in FINGERS4]
                pad = chains["middle"].radii[1] * 0.5
                self.palm = {"n": n, "u": u, "lat": lat,
                             "offset": PALM_THICKNESS * sum(chains[f].radii[1] for f in FINGERS4) / 4.0,
                             "a_mcp": {f: _dot(rest[f][1], u) for f in FINGERS4},
                             "l_range": (min(lats) - pad, max(lats) + pad)}


def _push(points, seg, u, vec):
    """Move a chain's segment seg (points seg and seg + 1) so the point of it at fraction u moves by
    vec. The knuckle (point 1) never moves, so a contact near it moves the segment's far end."""
    movable = ([(seg, 1.0 - u)] if seg >= 2 else []) + [(seg + 1, u)]
    total = sum(w * w for _, w in movable)
    if total < 1e-6:
        return
    for index, w in movable:
        points[index] = _add(points[index], _scale(vec, min(w / total, 3.0)))


def resolve_collisions(rig, out, apply_limits):
    """Push neighbouring fingers apart where they overlap more than at rest, and fingertips out of
    the palm, in the retargeted pose `out` (changed in place). The knuckles stay where they are, the
    bones keep their lengths, and a finger that was moved gets its joint limits applied again."""
    col = rig.collision
    pts = {f: [out[i] for i in CHAIN_LANDMARKS[f]] for f in rig.chains}
    moved = set()

    def restore_lengths(finger):
        chain, p = rig.chains[finger], pts[finger]
        for s in (1, 2, 3):
            d = _unit(_sub(p[s + 1], p[s])) or chain.rest_dirs[s]
            p[s + 1] = _add(p[s], _scale(d, chain.lengths[s]))

    for _ in range(COLLISION_ITERATIONS):
        touched = set()
        for (fa, fb), entries in col.pairs.items():
            pa, pb = pts[fa], pts[fb]
            for sa, sb, radii_sum, gap in entries:
                need = min(radii_sum * COLLISION_MARGIN, gap * 0.98)
                s, t, c1, c2 = _closest_segments(pa[sa], pa[sa + 1], pb[sb], pb[sb + 1])
                diff = _sub(c1, c2)
                dist = math.sqrt(_dot(diff, diff))
                if dist >= need:
                    continue
                normal = _scale(diff, 1.0 / dist) if dist > 1e-9 else col.axis[(fa, fb)]
                half = _scale(normal, (need - dist) * 0.5)
                _push(pa, sa, s, half)
                _push(pb, sb, t, _scale(half, -1.0))
                touched.update((fa, fb))
        palm = col.palm
        if PALM_COLLISION and palm is not None:
            lo, hi = palm["l_range"]
            for finger in FINGERS4:
                p, chain = pts[finger], rig.chains[finger]
                for k in (2, 3, 4):
                    a, l, h = _dot(p[k], palm["u"]), _dot(p[k], palm["lat"]), _dot(p[k], palm["n"])
                    if not (0.0 <= a < palm["a_mcp"][finger] and lo <= l <= hi):
                        continue  # not over the palm
                    need = palm["offset"] + chain.radii[k - 1] * COLLISION_MARGIN
                    if -0.5 * need < h < need:
                        p[k] = _add(p[k], _scale(palm["n"], need - h))
                        touched.add(finger)
        if not touched:
            break
        for finger in touched:
            restore_lengths(finger)
        moved |= touched

    for finger in moved:
        chain, p = rig.chains[finger], pts[finger]
        if apply_limits:
            dirs = [_unit(_sub(p[s + 1], p[s])) or chain.rest_dirs[s] for s in range(4)]
            dirs = chain.limit(dirs, straighten=False) if chain.hinge is not None else chain.limit_thumb(dirs)
            for s in range(4):
                p[s + 1] = _add(p[s], _scale(dirs[s], chain.lengths[s]))
        for i, q in zip(CHAIN_LANDMARKS[finger][1:], p[1:]):
            out[i] = q


def retarget(rig, local_points, apply_limits=True):
    """21 live landmarks in armature-local axes (wrist at the origin) to the 21 positions the
    rig's bones can reach. Falls back to the input when the rig has no chain data."""
    if rig.chains is None:
        return list(local_points)
    out = [local_points[_WRIST]] * len(local_points)
    for finger, chain in rig.chains.items():
        idx = CHAIN_LANDMARKS[finger]
        chain_points = chain.retarget([local_points[i] for i in idx], apply_limits)
        for i, p in zip(idx[1:], chain_points[1:]):
            out[i] = p
    if COLLISION and rig.collision is not None:
        resolve_collisions(rig, out, apply_limits)
    return out


def ease_to_rest(rig, local_points, k):
    """The pose k of the way from local_points to the rig's rest pose (k 0..1, armature-local axes,
    wrist where it was). Segment directions are blended and the rig's bone lengths kept, so
    every empty stays where its bone tip can reach it all the way."""
    out = list(local_points)
    for finger, chain in rig.chains.items():
        idx = CHAIN_LANDMARKS[finger]
        pos = local_points[idx[0]]
        for s in range(4):
            live = _unit(_sub(local_points[idx[s + 1]], local_points[idx[s]])) or chain.rest_dirs[s]
            rest = chain.rest_dirs[s]
            d = _unit(_add(_scale(live, 1.0 - k), _scale(rest, k))) or rest
            pos = tuple(p + chain.lengths[s] * c for p, c in zip(pos, d))
            out[idx[s + 1]] = pos
    return out


class Rig:
    """What the receiver needs to know about the armature's rest pose."""

    def __init__(self, arm):
        m = arm.matrix_world
        self.matrix0 = m.copy() if hasattr(m, "copy") else m  # put back on stop, exactly as found
        m0 = tuple(tuple(m[i][j] for j in range(3)) for i in range(3))
        self.loc0 = (m[0][3], m[1][3], m[2][3])
        bones = arm.data.bones
        head = tuple(bones[PALM_BONES["middle"]].head_local)
        local = {k: _sub(tuple(bones[n].tail_local), head) for k, n in PALM_BONES.items()}
        self.span = math.sqrt(_dot(local["middle"], local["middle"]))  # wrist -> middle knuckle, the scale reference
        local_frame = palm_frame(_ZERO, local["index"], local["middle"], local["pinky"])
        if local_frame is None:
            raise ValueError("palm bones are degenerate")
        # Reflection across the palm's symmetry plane, in armature-local axes. It comes from the
        # bones themselves, so it does not depend on the object's current transform.
        lateral = local_frame[0]
        self.reflect = tuple(tuple((1.0 if i == j else 0.0) - 2.0 * lateral[i] * lateral[j]
                                   for j in range(3)) for i in range(3))
        # The rest orientation must be an ordinary rotation. A .blend saved while the receiver had
        # the rig mirrored holds a mirrored transform; take the mirror back out of it, or "mirror"
        # would mean the opposite and the fingers would curl the wrong way.
        if _dot(m0[0], _cross(m0[1], m0[2])) < 0:
            m0 = _matmul(m0, self.reflect)
        self.m0 = m0
        world = {k: _matvec(self.m0, v) for k, v in local.items()}
        frame = palm_frame(_ZERO, world["index"], world["middle"], world["pinky"])
        mirrored = {k: _matvec(self.m0, _matvec(self.reflect, v)) for k, v in local.items()}
        self.frames = {
            False: frame,
            True: palm_frame(_ZERO, mirrored["index"], mirrored["middle"], mirrored["pinky"]),
        }
        try:
            self.chains = {f: Chain(f, bones, names) for f, names in BONE_CHAINS.items()}
        except KeyError:  # some bones are missing: no retargeting, positions are copied
            self.chains = None
        self.collision = Collision(self.chains) if self.chains is not None else None
        # Which way the rig's own fingers curl relative to its palm (+1 or -1), read from its rest
        # pose with the same measure used on the live hand. None if the rest pose is too straight.
        self.bend_sign = None
        if self.chains is not None:
            total = sum(_dot(_bend_axis(self.chains[f].rest_dirs), local_frame[0]) for f in FINGERS4)
            if abs(total / len(FINGERS4)) > 0.05:
                self.bend_sign = 1.0 if total > 0 else -1.0


def _new_stats():
    return {"packets": 0, "dropped": 0, "stale": 0, "applied": 0, "max_backlog": 0,
            "lat_sum": 0.0, "lat_max": 0.0, "ms_sum": 0.0, "tick_sum": 0.0,
            "tick_max": 0.0, "ticks": 0}


class State:
    """Everything the receiver tracks for one hand (one armature and its landmark empties)."""

    def __init__(self):
        self.slot = 0                  # which of the sender's hands this state follows
        self.suffix = ""               # appended to every object name of this hand
        self.armature_name = ARMATURE_NAME
        self.origin = _ZERO            # where this hand's wrist sits relative to slot 0's
        self.mirror_override = None    # this hand's own mirror choice; None uses MIRROR_OVERRIDE
        self.tag = "[hand]"            # prefix of its console messages
        self.sid = None
        self.last_seq = -1
        self.span_samples = []
        self.scale = FIXED_WORLD_SCALE
        self.anchor = None             # first root position of the session (Blender units)
        self.delta = _ZERO             # current hand translation
        self.rig = None                # Rig, set by start() when the armature is usable
        self.armature_moved = False
        self.armature_loc0 = None      # original location when there is no Rig (translation only)
        self.votes = []                # recent handedness label votes: +1 mirror the rig, -1 don't
        self.shape_score = 0.0         # -1 .. 1 from the finger bends: +1 same shape as the rig
        self.shape_confirmed = False   # True once the finger bends have decided the shape
        self.mirror = None             # None until decided
        self.hand_present = None       # None until the first frame/lost packet
        self.last_local = None         # last retargeted pose (armature-local), for relaxing to rest
        self.last_matrix = None        # armature matrix that pose belongs to
        self.last_offset = _ZERO
        self.lost_since = None         # time.monotonic() when the hand was lost
        self.relaxed = False           # True once the rest pose has been reached
        self.missing_empties = set()
        self.last_data_time = time.monotonic()
        self.warned_no_data = False
        self.stats_since = time.monotonic()
        self.stats = _new_stats()
        self.last_report = None

    def new_session(self, sid):
        self.sid = sid
        self.last_seq = -1
        self.span_samples = []
        self.scale = FIXED_WORLD_SCALE
        self.anchor = None
        self.delta = _ZERO
        self.votes = []
        self.shape_score = 0.0
        self.shape_confirmed = False
        self.mirror = None
        self.hand_present = None
        self.lost_since = None
        self.relaxed = False


def decode(data):
    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(msg, dict) or msg.get("v") != PROTOCOL_VERSION:
        return None
    if "sid" not in msg or "seq" not in msg:
        return None
    return msg


def drain_all(sock, states):
    """Read every queued packet and hand each to the state of its slot (a packet without a
    "slot" belongs to slot 0). Returns ({slot: newest frame/lost message}, stop)."""
    newest = {}
    stop = False
    backlog = {slot: 0 for slot in states}
    first_slot = min(states)
    first = states[first_slot]
    while True:
        try:
            data, addr = sock.recvfrom(4096)
        except BlockingIOError:
            break
        except ConnectionResetError:  # Windows: ICMP from an earlier send; harmless
            continue
        msg = decode(data)
        if msg is None:
            first.stats["dropped"] += 1
            backlog[first_slot] += 1
            continue
        slot = msg.get("slot", 0)
        state = states.get(slot)
        if state is None:  # a hand this file has no rig for
            first.stats["dropped"] += 1
            backlog[first_slot] += 1
            continue
        backlog[slot] += 1
        if msg["sid"] != state.sid:
            state.new_session(msg["sid"])
            print(f"{state.tag} sender session {msg['sid']} connected")
            try:  # the reply tells the sender how many hand rigs this file has
                sock.sendto(ACK if len(states) == 1 else ACK + b" hands=%d" % len(states), addr)
            except OSError:
                pass
        if msg["seq"] <= state.last_seq:
            state.stats["dropped"] += 1
            continue
        state.last_seq = msg["seq"]
        if msg.get("stop"):
            stop = True
        elif msg.get("lost") or "lm" in msg:
            newest[slot] = msg
    for slot, count in backlog.items():
        state = states[slot]
        if count:
            state.last_data_time = time.monotonic()
            state.warned_no_data = False
        state.stats["packets"] += count
        state.stats["max_backlog"] = max(state.stats["max_backlog"], count)
    return newest, stop


def drain(sock, state):
    """Single-hand form of drain_all: returns (newest, stop) for this one state."""
    newest, stop = drain_all(sock, {state.slot: state})
    return newest.get(state.slot), stop


def hand_span(points):
    """Real WRIST -> MIDDLE_FINGER_MCP distance in metres (the palm is rigid)."""
    a, b = points[_WRIST], points[_MIDDLE_MCP]
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def update_scale(state, points):
    """Fix the metres -> Blender-units scale from the first CALIBRATION_FRAMES frames."""
    if FIXED_WORLD_SCALE is not None or len(state.span_samples) >= CALIBRATION_FRAMES:
        return
    span = hand_span(points)
    if span > 1e-4:
        state.span_samples.append(span)
        reference = state.rig.span if state.rig is not None and state.rig.span else REFERENCE_DISTANCE
        state.scale = reference / statistics.median(state.span_samples)


def to_blender_axes(x, y, z):
    """MediaPipe (x right, y down, z away) -> Blender (x, y=z, z=-y). This is a proper
    rotation (determinant +1), so it never mirrors the hand."""
    return (x, z, -y)


def yaw_about_z(points, degrees):
    """The points turned `degrees` about the Z axis through the origin (the wrist, for points relative to it)."""
    if not degrees:
        return list(points)
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return [(c * x - s * y, s * x + c * y, z) for x, y, z in points]


def turn_about_hand_axis(points, degrees):
    """The points (relative to the wrist) turned `degrees` about the hand's own long axis, the line from the
    wrist through the middle knuckle. The turn is rigid and follows the hand wherever it points, so a hand
    that is tilted, rolled or swung is turned the same way and keeps moving in the direction it moves.
    If the palm cannot be measured the points are returned unchanged."""
    if not degrees:
        return list(points)
    frame = palm_frame(points[_WRIST], points[_INDEX_MCP], points[_MIDDLE_MCP], points[_PINKY_MCP])
    if frame is None:
        return list(points)
    axis = frame[1]
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    out = []
    for p in points:
        along = _dot(axis, p)
        cross = _cross(axis, p)
        out.append(tuple(p[i] * c + cross[i] * s + axis[i] * along * (1.0 - c) for i in range(3)))
    return out


def turn_hand(points, degrees):
    """HAND_YAW_DEG applied the way HAND_TURN_ABOUT says."""
    if HAND_TURN_ABOUT == "world":
        return yaw_about_z(points, degrees)
    return turn_about_hand_axis(points, degrees)


def world_to_blender(points, scale):
    """Landmarks relative to the wrist, in Blender axes and units."""
    wx, wy, wz = points[_WRIST]
    return [to_blender_axes((x - wx) * scale, (y - wy) * scale, (z - wz) * scale)
            for x, y, z in points]


def root_translation(state, root):
    """Hand translation in Blender units relative to where the hand first appeared this
    session. Returns the previous translation when the sender sent no usable root."""
    if root is None or len(root) != 3 or state.scale is None:
        return state.delta
    here = to_blender_axes(*(v * state.scale for v in root))
    if UNMIRROR_HAND:  # the same reflection as the hand's own landmarks (see UNMIRROR_HAND)
        here = (here[0], -here[1], here[2])
    if state.anchor is None:
        state.anchor = here
    state.delta = tuple((h - a) * g for h, a, g in zip(here, state.anchor, TRANSLATE_GAIN))
    return state.delta


def hand_shape(label):
    """Shape of the measured hand from MediaPipe's label (see MIRROR comment above)."""
    return {"Right": "Left", "Left": "Right"}.get(label)


def _set_mirror(state, mirror, why):
    if mirror != state.mirror:
        print(f"{state.tag} rig {'mirrored' if mirror else 'not mirrored'} ({why})")
    state.mirror = mirror


def update_mirror(state, msg, rel):
    """Decide whether the rig must be mirrored to match the live hand's shape.

    The fingers' own bends decide it: for a given hand shape they curl a fixed way relative to
    the palm, and the rig's rest pose gives the same measure for its shape. This needs no label
    and cannot be fooled by how MediaPipe or the camera mirrors the image. Until the fingers have
    bent enough to say, MediaPipe's handedness label is used as a weak first guess, and a wrong
    guess is corrected by the first clearly bent frame. MIRROR_OVERRIDE forces the answer."""
    override = state.mirror_override if state.mirror_override is not None else MIRROR_OVERRIDE
    if override is not None:
        state.mirror = bool(override)
        return
    rig_sign = state.rig.bend_sign if state.rig is not None else None
    frame = palm_frame(rel[_WRIST], rel[_INDEX_MCP], rel[_MIDDLE_MCP], rel[_PINKY_MCP])
    if rig_sign is not None and frame is not None:
        evidence = bend_evidence(rel, frame[0])
        if abs(evidence) > EVIDENCE_MIN:
            agree = (1.0 if evidence * rig_sign > 0 else -1.0) * min(1.0, abs(evidence) / EVIDENCE_FULL)
            state.shape_score += EVIDENCE_SMOOTH * (agree - state.shape_score)
    if state.shape_confirmed:
        if abs(state.shape_score) > 0.5:  # flip only on solid, lasting opposite evidence
            _set_mirror(state, state.shape_score < 0, "finger bends")
        return
    if abs(state.shape_score) > 0.3:
        state.shape_confirmed = True
        _set_mirror(state, state.shape_score < 0, "finger bends")
        return
    shape = hand_shape(msg.get("hand"))
    if shape is not None:
        weight = msg.get("score")
        weight = 1.0 if weight is None else max(0.0, min(1.0, float(weight)))
        state.votes.append(weight if shape != RIG_SHAPE else -weight)
        del state.votes[:-LABEL_VOTE_FRAMES]
    if not state.votes:
        if state.mirror is None:
            state.mirror = False
        return
    avg = sum(state.votes) / len(state.votes)
    decided = state.mirror
    if decided is None:
        decided = avg > 0
    elif decided and avg < -0.4:
        decided = False
    elif not decided and avg > 0.4:
        decided = True
    _set_mirror(state, decided, f"{msg.get('hand')} label, a first guess until the fingers bend")


def _matrix4(rot3, loc):
    rows = [list(rot3[0]) + [loc[0]], list(rot3[1]) + [loc[1]],
            list(rot3[2]) + [loc[2]], [0.0, 0.0, 0.0, 1.0]]
    try:
        from mathutils import Matrix
        return Matrix(rows)
    except ImportError:
        return tuple(tuple(r) for r in rows)


def drive_armature(state, rel, offset):
    """Rotate (and mirror) the Armature object so its rest-pose palm lines up with the live
    palm, and translate it by offset. Returns the armature's 3x3 matrix (armature-local to
    world), or None if the live palm frame is degenerate."""
    arm = bpy.data.objects.get(state.armature_name)
    rig = state.rig
    if arm is None or rig is None:
        return None
    live = palm_frame(rel[_WRIST], rel[_INDEX_MCP], rel[_MIDDLE_MCP], rel[_PINKY_MCP])
    if live is None:
        return None
    mirror = bool(state.mirror)
    rot = rotation_between(rig.frames[mirror], live)
    base = _matmul(rig.m0, rig.reflect) if mirror else rig.m0
    loc = tuple(l + o for l, o in zip(rig.loc0, offset))
    matrix = _matmul(rot, base)
    arm.matrix_world = _matrix4(matrix, loc)
    state.armature_moved = True
    return matrix


def retarget_world(state, rel, matrix):
    """Move live world points into the armature's own frame, retarget them to what the bones can
    reach (rig lengths, joint limits), and move them back to world axes."""
    to_local = _inv3(matrix)
    local = [_matvec(to_local, p) for p in rel]
    reachable = retarget(state.rig, local, CLAMP_TO_LIMITS)
    state.last_local, state.last_matrix = reachable, matrix
    return [_matvec(matrix, p) for p in reachable]


def relax_lost_hand(state, now):
    """While the hand is lost: hold the last pose for LOST_HOLD_S, then ease the fingers to the
    rig's rest pose over LOST_EASE_S. Returns True if it moved the empties."""
    if (LOST_EASE_S <= 0 or state.hand_present is not False or state.lost_since is None
            or state.relaxed or state.last_local is None or state.last_matrix is None
            or state.rig is None or state.rig.chains is None):
        return False
    t = (now - state.lost_since - LOST_HOLD_S) / LOST_EASE_S
    if t <= 0:
        return False
    k = min(1.0, t)
    k = k * k * (3.0 - 2.0 * k)
    local = ease_to_rest(state.rig, state.last_local, k)
    apply_to_empties(state, [_matvec(state.last_matrix, p) for p in local], state.last_offset)
    if t >= 1.0:
        state.relaxed = True
        print(f"{state.tag} hand still lost; fingers relaxed to the rest pose")
    return True


def translate_armature(state, offset):
    """Translation only (no palm tracking)."""
    arm = bpy.data.objects.get(state.armature_name)
    if arm is None:
        return
    if state.rig is not None:
        loc0 = state.rig.loc0
    else:
        loc0 = state.armature_loc0 if state.armature_loc0 is not None else tuple(arm.location)
    state.armature_loc0 = loc0
    arm.location = tuple(l + o for l, o in zip(loc0, offset))
    state.armature_moved = True


def restore_armature(state):
    if not state.armature_moved or bpy is None:
        return
    arm = bpy.data.objects.get(state.armature_name)
    if arm is not None:
        if state.rig is not None:
            arm.matrix_world = state.rig.matrix0
        else:
            arm.location = state.armature_loc0
    state.armature_moved = False


def apply_to_empties(state, positions, offset=_ZERO):
    ox, oy, oz = offset
    for name, (x, y, z) in zip(LANDMARK_NAMES, positions):
        empty = bpy.data.objects.get(name + state.suffix)
        if empty is None:
            if name != "WRIST" and name not in state.missing_empties:
                state.missing_empties.add(name)
                print(f"{state.tag} no empty named '{name + state.suffix}' in the scene; skipping it")
            continue
        empty.location = (x + ox, y + oy, z + oz)


def _apply_frame(state, msg):
    points = msg["lm"]
    if len(points) != len(LANDMARK_NAMES):
        state.stats["dropped"] += 1
        return
    if state.hand_present is False:
        print(f"{state.tag} hand found")
    state.hand_present = True
    state.lost_since = None
    state.relaxed = False

    sent_at = msg.get("t")
    if sent_at is not None:
        age = time.time() - sent_at
        if MAX_FRAME_AGE_S is not None and age > MAX_FRAME_AGE_S:
            state.stats["stale"] += 1
            return
        state.stats["lat_sum"] += age
        state.stats["lat_max"] = max(state.stats["lat_max"], age)
    state.stats["ms_sum"] += msg.get("ms", 0.0)

    update_scale(state, points)
    if state.scale is None:
        return
    rel = world_to_blender(points, state.scale)
    if UNMIRROR_HAND:
        rel = [(x, -y, z) for x, y, z in rel]
    rel = turn_hand(rel, HAND_YAW_DEG)
    offset = _ZERO
    if TRANSLATE_HAND:
        offset = root_translation(state, msg.get("root"))
    matrix = None
    if TRACK_PALM and state.rig is not None:
        update_mirror(state, msg, rel)
        matrix = drive_armature(state, rel, offset)
    else:
        if TRANSLATE_HAND:
            translate_armature(state, offset)
        if state.rig is not None:
            matrix = state.rig.m0  # the armature keeps its own orientation
    state.last_local = None
    if RETARGET and matrix is not None and state.rig.chains is not None:
        rel = retarget_world(state, rel, matrix)
    where = _add(offset, state.origin)  # the empties of a second hand sit next to the first's
    state.last_offset = where
    apply_to_empties(state, rel, where)
    state.stats["applied"] += 1


def _report(state, now):
    s = state.stats
    n = max(s["applied"], 1)
    t = max(s["ticks"], 1)
    print(f"{state.tag} last {now - state.stats_since:.0f}s: packets={s['packets']} "
          f"applied={s['applied']} dropped={s['dropped']} stale={s['stale']} "
          f"max_backlog={s['max_backlog']} | latency sender->apply "
          f"avg={s['lat_sum'] / n * 1000:.1f}ms max={s['lat_max'] * 1000:.1f}ms | "
          f"sender work avg={s['ms_sum'] / n:.1f}ms | "
          f"receiver tick avg={s['tick_sum'] / t * 1000:.2f}ms max={s['tick_max'] * 1000:.2f}ms")
    state.last_report = {"fps": s["applied"] / max(now - state.stats_since, 1e-6),
                         "latency_ms": s["lat_sum"] / n * 1000, "max_backlog": s["max_backlog"],
                         "tick_ms": s["tick_sum"] / t * 1000}
    state.stats_since = now
    state.stats = _new_stats()


def process(sock, states):
    """One tick of work for one State or a {slot: State} dict. Returns False only if the receiver
    should end itself."""
    if isinstance(states, State):
        states = {states.slot: states}
    t0 = time.perf_counter()
    newest, stop = drain_all(sock, states)
    now = time.monotonic()

    for slot, state in states.items():
        msg = newest.get(slot)
        if msg is not None:
            if msg.get("lost"):
                if state.hand_present is not False:
                    print(f"{state.tag} hand lost; holding last pose"
                          + (f", then relaxing to rest after {LOST_HOLD_S:g}s" if LOST_EASE_S > 0 else ""))
                    state.lost_since = now
                    state.relaxed = False
                state.hand_present = False
            else:
                _apply_frame(state, msg)
        relax_lost_hand(state, time.monotonic())
        if (not state.warned_no_data and state.sid is not None
                and now - state.last_data_time > NO_DATA_WARN_SECONDS):
            state.warned_no_data = True
            print(f"{state.tag} no packets received; is the sender running"
                  + (" with two hands (--hands 2)?" if slot else "?"))

    busy = time.perf_counter() - t0
    for state in states.values():
        state.stats["ticks"] += 1
        state.stats["tick_sum"] += busy
        state.stats["tick_max"] = max(state.stats["tick_max"], busy)
        if STATS_SECONDS and now - state.stats_since >= STATS_SECONDS:
            _report(state, now)

    if stop:
        print("[hand] sender sent stop" + ("" if STOP_ON_SENDER_STOP else "; still listening"))
        return not STOP_ON_SENDER_STOP
    return True


def status_of(state):
    """What the Blender panel shows: a short summary of the receiver's state."""
    if state.sid is None:
        text = "waiting for the sender"
    elif state.hand_present is False:
        text = "hand lost (relaxed to rest)" if state.relaxed else "hand lost"
    elif state.hand_present:
        text = "tracking"
    else:
        text = "waiting for a hand"
    report = state.last_report or {}
    override = state.mirror_override if state.mirror_override is not None else MIRROR_OVERRIDE
    return {"running": True, "slot": state.slot, "name": HAND_NAMES[state.slot % 2], "text": text, "mirror": state.mirror, "override": override,
            "shape_confirmed": state.shape_confirmed, "scale": state.scale,
            "fps": report.get("fps"), "latency_ms": report.get("latency_ms"),
            "tick_ms": report.get("tick_ms"), "port": PORT}


def status_all(states):
    """Status of the first hand, plus every hand's under "hands"."""
    hands = [status_of(states[slot]) for slot in sorted(states)]
    return dict(hands[0], hands=hands)


def _load_panel():
    """Load blender_panel.py from next to this file (a small Blender sidebar panel with
    start/stop, status and the mirror choice). The receiver works without it."""
    here = globals().get("__file__")
    path = os.path.join(os.path.dirname(os.path.abspath(here)), "blender_panel.py") if here else None
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            exec(compile(f.read(), path, "exec"), {"__name__": "v5_hand_panel", "__file__": path,
                                                   "RECEIVER_FILE": here})
    except ImportError:  # outside Blender (tests): there is no panel to show
        return
    except Exception as exc:  # the panel is optional
        print(f"[hand] panel not loaded: {exc!r}")


def make_states():
    """One State per hand slot that has an armature in the scene. Slot 0 comes first."""
    states = {}
    for slot, suffix in enumerate(HAND_SUFFIXES):
        name = ARMATURE_NAME + suffix
        arm = bpy.data.objects.get(name)
        if arm is None and slot:
            continue
        state = State()
        state.slot, state.suffix, state.armature_name = slot, suffix, name
        if (TRACK_PALM or TRANSLATE_HAND) and arm is not None:
            try:
                state.rig = Rig(arm)
            except (KeyError, ValueError) as exc:
                print(f"[hand] palm tracking disabled for '{name}': {exc!r} "
                      f"(needs bones {', '.join(PALM_BONES.values())})")
        elif TRACK_PALM:
            print(f"[hand] palm tracking disabled: no object named '{name}'")
        states[slot] = state
    if len(states) > 1:
        base = states[0].rig.loc0 if states[0].rig else _ZERO
        for slot, state in states.items():
            state.tag = f"[{HAND_NAMES[slot % 2].lower()}]"
            if state.rig is not None:
                state.origin = _sub(state.rig.loc0, base)
    return states


def start():
    previous = bpy.app.driver_namespace.get(_STOP_KEY)
    if previous:
        previous()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((HOST, PORT))
    except OSError as exc:
        sock.close()
        print(f"[hand] cannot bind {HOST}:{PORT}: {exc}. "
              "Another Blender script or program may still hold the port.")
        return
    sock.setblocking(False)
    states = make_states()

    def cleanup():
        sock.close()
        for state in states.values():
            restore_armature(state)
        bpy.app.driver_namespace.pop(_STOP_KEY, None)
        bpy.app.driver_namespace.pop(CONTROL_KEY, None)
        bpy.app.driver_namespace[STATUS_KEY] = {"running": False, "text": "stopped"}
        print("[hand] receiver stopped")

    def set_mirror(value, slot=None):
        """None = decide from the finger bends, True / False = force it. slot None sets the
        default for every hand; a slot number sets that hand's own choice."""
        global MIRROR_OVERRIDE
        if slot is None:
            MIRROR_OVERRIDE = value
            targets = [st for st in states.values() if st.mirror_override is None]
        else:
            targets = [states[slot]] if slot in states else []
            for st in targets:
                st.mirror_override = value
        if value is None:
            for st in targets:
                st.mirror = None
                st.shape_confirmed = False
                st.shape_score = 0.0
                st.votes = []

    def tick():
        try:
            if process(sock, states):
                bpy.app.driver_namespace[STATUS_KEY] = status_all(states)
                return TIMER_INTERVAL
        except Exception as exc:  # keep the timer alive across a bad frame
            print(f"[hand] error: {exc!r}")
            return TIMER_INTERVAL
        cleanup()
        return None  # returning None ends the timer

    def stop():
        if bpy.app.timers.is_registered(tick):
            bpy.app.timers.unregister(tick)
        cleanup()

    bpy.app.driver_namespace[_STOP_KEY] = stop
    bpy.app.driver_namespace[CONTROL_KEY] = {"stop": stop, "set_mirror": set_mirror, "hands": len(states),
                                                 "states": states}
    bpy.app.driver_namespace[STATUS_KEY] = status_all(states)
    bpy.app.timers.register(tick)
    _load_panel()
    first = states[0]
    for state in states.values():
        if RETARGET and state.rig is not None and state.rig.chains is None:
            print(f"{state.tag} retargeting disabled: some of the 20 hand bones are missing; "
                  "landmark positions are copied as they are")
    print(f"[hand] listening on {HOST}:{PORT}, interval {TIMER_INTERVAL * 1000:.0f} ms"
          f"{', ' + str(len(states)) + ' hands' if len(states) > 1 else ''}"
          f"{', palm tracking ON' if first.rig and TRACK_PALM else ''}"
          f"{', retargeting ON' if first.rig and first.rig.chains and RETARGET else ''}"
          f"{', hand translation ON' if TRANSLATE_HAND else ''}")


if __name__ == "__main__":
    start()
