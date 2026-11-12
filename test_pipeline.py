"""Offline checks for the v5 pipeline. Run: python v5/test_pipeline.py

Uses real loopback UDP and a stand-in for Blender's `bpy`. It does NOT start
Blender or open the webcam, so it cannot prove the rig moves correctly in Blender.
"""
import math
import random
import socket
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import protocol
from one_euro import OneEuroFilter, LandmarkFilter
import blender_receiver as br

br.HAND_YAW_DEG = 0.0  # most tests check the mapping itself; test_hand_yaw covers turning each hand
br.UNMIRROR_HAND = False  # ... and the raw axis mapping; test_unmirror covers undoing the camera's mirror


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def make_hand(span=0.09, offset=(0.0, 0.0, 0.0)):
    """Synthetic 21-point hand: WRIST at offset, MIDDLE_FINGER_MCP `span` away along -y."""
    pts = [(offset[0] + 0.005 * i, offset[1] - span * (i / 9 if i else 0), offset[2]) for i in range(21)]
    pts[0] = offset
    pts[9] = (offset[0], offset[1] - span, offset[2])
    return pts


def test_one_euro():
    rng = random.Random(1)
    f = OneEuroFilter()
    assert f(0.5, 0.0) == 0.5, "first sample must pass through"

    # still hand with 2 mm noise at 30 fps: output jitter must drop clearly. The defaults trade a
    # little smoothness (they keep ~65% of the noise) for little lag on fast finger curls.
    f = OneEuroFilter()
    raw = [0.002 * rng.gauss(0, 1) for _ in range(300)]
    out = [f(x, i / 30) for i, x in enumerate(raw)]
    std = lambda v: (sum(x * x for x in v) / len(v)) ** 0.5
    assert std(out[30:]) < 0.75 * std(raw[30:]), (std(out[30:]), std(raw[30:]))

    # fast sweep (0.5 m/s): lag must stay small
    f = OneEuroFilter()
    lag = 0.0
    for i in range(90):
        t = i / 30
        y = f(0.5 * t, t)
        lag = 0.5 * t - y
    assert 0 <= lag < 0.015, f"lag {lag:.4f} m at 0.5 m/s"

    lf = LandmarkFilter()
    out = lf(make_hand(), 0.0)
    assert len(out) == 21 and len(out[0]) == 3


def test_protocol():
    pts = make_hand()
    data = protocol.encode_frame("abc", 7, 12.3456789, "Right", pts, (0.1, 0.2, 0.5), 12.34)
    assert len(data) < protocol.MAX_PACKET_BYTES
    msg = protocol.decode(data)
    assert msg["seq"] == 7 and msg["sid"] == "abc" and len(msg["lm"]) == 21
    assert msg["root"] == [0.1, 0.2, 0.5] and msg["ms"] == 12.34
    assert "root" not in protocol.decode(protocol.encode_frame("a", 1, 0.0, "Right", pts))
    assert protocol.decode(b"garbage") is None
    assert protocol.decode(b'{"v":99,"sid":"a","seq":1}') is None
    assert protocol.decode(b'"stop"') is None
    assert protocol.decode(protocol.encode_lost("a", 1))["lost"] is True
    assert protocol.decode(protocol.encode_stop("a", 2))["stop"] is True
    assert protocol.LANDMARK_NAMES == br.LANDMARK_NAMES, "name lists diverged"
    assert protocol.PROTOCOL_VERSION == br.PROTOCOL_VERSION
    print(f"  frame packet size: {len(data)} bytes")


def make_rx():
    port = free_port()
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", port))
    rx.setblocking(False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.setblocking(False)
    return rx, tx, ("127.0.0.1", port)


def test_drain():
    rx, tx, addr = make_rx()
    st = br.State()
    for seq in range(1, 101):
        tx.sendto(protocol.encode_frame("s1", seq, seq / 30, "Right", make_hand()), addr)
    import time; time.sleep(0.05)
    newest, stop = br.drain(rx, st)
    assert newest["seq"] == 100 and not stop, "must keep only the newest packet"
    assert st.stats["max_backlog"] == 100
    assert br.drain(rx, st) == (None, False), "queue must be empty afterwards"

    # the sender gets an ack on a new session
    time.sleep(0.05)
    data, _ = tx.recvfrom(64)
    assert data == br.ACK

    # stale / duplicate seq dropped
    tx.sendto(protocol.encode_frame("s1", 50, 1.0, "Right", make_hand()), addr)
    tx.sendto(protocol.encode_frame("s1", 100, 1.0, "Right", make_hand()), addr)
    time.sleep(0.05)
    newest, _ = br.drain(rx, st)
    assert newest is None and st.stats["dropped"] >= 2

    # a restarted sender (new sid, seq back to 1) is accepted
    tx.sendto(protocol.encode_frame("s2", 1, 0.1, "Right", make_hand()), addr)
    time.sleep(0.05)
    newest, _ = br.drain(rx, st)
    assert newest is not None and newest["sid"] == "s2" and st.last_seq == 1

    # junk, wrong version, and the old plain-text "stop" are ignored, not fatal
    tx.sendto(b"stop", addr)
    tx.sendto(b"\xff\xfe", addr)
    tx.sendto(b'{"status":"ready"}', addr)
    tx.sendto(protocol.encode_stop("s2", 2), addr)
    time.sleep(0.05)
    newest, stop = br.drain(rx, st)
    assert newest is None and stop is True
    rx.close(); tx.close()


def test_mapping_and_scale():
    st = br.State()
    for _ in range(5):
        br.update_scale(st, make_hand(span=0.09))
    assert abs(st.scale - br.REFERENCE_DISTANCE / 0.09) < 1e-9

    # tilting the hand must not change the scale: world span is rigid
    tilted = [(x, y * math.cos(0.8) - z * math.sin(0.8), y * math.sin(0.8) + z * math.cos(0.8))
              for x, y, z in make_hand(span=0.09)]
    assert abs(br.hand_span(tilted) - 0.09) < 1e-9

    out = br.world_to_blender(make_hand(span=0.09, offset=(0.3, -0.2, 0.1)), st.scale)
    assert out[0] == (0.0, 0.0, 0.0), "wrist must be the origin"
    d = math.dist(out[0], out[9])
    assert abs(d - br.REFERENCE_DISTANCE) < 1e-9, d

    pts = make_hand()
    pts[4] = (0.1, 0.2, 0.3)  # relative to wrist at origin: x right, y down, z away
    x, y, z = br.world_to_blender(pts, 1.0)[4]
    assert (x, y, z) == (0.1, 0.3, -0.2), (x, y, z)

    # calibration freezes after CALIBRATION_FRAMES
    st = br.State()
    for _ in range(br.CALIBRATION_FRAMES):
        br.update_scale(st, make_hand(span=0.09))
    frozen = st.scale
    br.update_scale(st, make_hand(span=0.2))
    assert st.scale == frozen


class FakeEmpty:
    def __init__(self):
        self.location = (0.0, 0.0, 0.0)


def install_fake_bpy(empty_names):
    empties = {n: FakeEmpty() for n in empty_names}
    timers = types.SimpleNamespace(funcs=[])
    timers.register = lambda f: timers.funcs.append(f)
    timers.unregister = lambda f: timers.funcs.remove(f)
    timers.is_registered = lambda f: f in timers.funcs
    br.bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(objects=types.SimpleNamespace(get=empties.get)),
        app=types.SimpleNamespace(timers=timers, driver_namespace={}),
    )
    return empties, timers


def test_end_to_end_with_fake_bpy():
    import time
    names = [n for n in br.LANDMARK_NAMES if n != "PINKY_MCP"]  # one empty deliberately missing
    empties, timers = install_fake_bpy(names)
    br.PORT = free_port()
    br.start()
    assert len(timers.funcs) == 1
    tick = timers.funcs[0]

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (br.HOST, br.PORT)
    pts = make_hand(span=0.09)
    pts[8] = (0.01, -0.05, 0.02)
    for seq in range(1, 21):
        tx.sendto(protocol.encode_frame("e2e", seq, time.time(), "Right", pts), addr)
    time.sleep(0.05)
    assert tick() == br.TIMER_INTERVAL
    scale = br.REFERENCE_DISTANCE / 0.09
    exp = (0.01 * scale, 0.02 * scale, 0.05 * scale)
    got = empties["INDEX_FINGER_TIP"].location
    assert all(abs(a - b) < 1e-4 for a, b in zip(got, exp)), (got, exp)
    assert empties["WRIST"].location == (0.0, 0.0, 0.0)

    # hand lost: pose is held
    held = empties["INDEX_FINGER_TIP"].location
    tx.sendto(protocol.encode_lost("e2e", 21), addr)
    time.sleep(0.05)
    tick()
    assert empties["INDEX_FINGER_TIP"].location == held

    # re-running the script replaces the old instance instead of leaking it
    br.start()
    assert len(timers.funcs) == 1 and timers.funcs[0] is not tick

    # a stale frame (sent 2 s ago) is not applied; the pose is held
    tick2 = timers.funcs[0]
    pts2 = make_hand(span=0.09)
    pts2[8] = (0.05, -0.09, 0.0)
    tx.sendto(protocol.encode_frame("e2e", 22, time.time() - 2.0, "Right", pts2), addr)
    time.sleep(0.05)
    tick2()
    assert empties["INDEX_FINGER_TIP"].location == held, "stale frame must not move the hand"

    # the sender stopping leaves the receiver listening so the sender can be restarted
    tx.sendto(protocol.encode_stop("e2e", 23), addr)
    time.sleep(0.05)
    assert tick2() == br.TIMER_INTERVAL
    br.start()  # restart cleanly
    tick3 = timers.funcs[0]
    br.STOP_ON_SENDER_STOP = True
    tx.sendto(protocol.encode_stop("e2e", 24), addr)
    time.sleep(0.05)
    assert tick3() is None, "with STOP_ON_SENDER_STOP a stop ends the timer"
    br.STOP_ON_SENDER_STOP = False
    tx.close()
    br.bpy = None


def test_outlier_gate():
    from outlier import OutlierGate
    rng = random.Random(3)

    def frame(jitter=0.001):
        return [(x + rng.gauss(0, jitter), y + rng.gauss(0, jitter), z + rng.gauss(0, jitter))
                for x, y, z in make_hand()]

    g = OutlierGate()
    t = 0.0
    for _ in range(20):
        t += 1 / 30
        assert g.check(frame(), t)[0], "normal jitter must pass"

    bad = frame()
    bad[8] = (bad[8][0] + 0.25, bad[8][1], bad[8][2])  # fingertip teleports 25 cm
    t += 1 / 30
    ok, reason = g.check(bad, t)
    assert not ok and "jumped" in reason, reason
    t += 1 / 30
    assert g.check(frame(), t)[0], "the next normal frame passes again"

    big = [(x * 1.8, y * 1.8, z * 1.8) for x, y, z in make_hand()]  # palm span +80 %
    t += 1 / 30
    ok, reason = g.check(big, t)
    assert not ok and "palm span" in reason, reason

    # a real, lasting change is accepted after max_consecutive rejections
    g2 = OutlierGate(max_consecutive=3)
    for i in range(10):
        g2.check(frame(), (i + 1) / 30)
    shifted = frame()
    shifted[8] = (shifted[8][0] + 0.25, shifted[8][1], shifted[8][2])
    results = [g2.check(shifted, (11 + i) / 30)[0] for i in range(5)]
    assert results[:3] == [False] * 3 and results[3] is True, results
    g.reset()
    assert g.check(bad, 99.0)[0], "after reset the first frame is always accepted"


def test_estimate_root():
    import sender
    W, H, FOV = 640, 480, 60.0
    f = (W / 2) / math.tan(math.radians(FOV / 2))
    rel = []
    for i in range(21):  # hand-centred world points: spread in x/y, small z, slightly tilted
        rel.append(((i % 5 - 2) * 0.02, ((i * 3) % 7 - 3) * 0.015, ((i * 5) % 3 - 1) * 0.01))
    # hand tilted about 40 degrees: the wrist is nearer the camera than the knuckles
    rel[0] = (0.0, 0.05, -0.04)
    rel[5], rel[9], rel[13], rel[17] = ((-0.03, -0.02, 0.025), (0.0, -0.03, 0.03),
                                        (0.03, -0.02, 0.025), (0.05, 0.0, 0.01))
    rng = random.Random(5)
    centres = ((0.0, 0.0, 0.5), (0.12, -0.05, 0.7), (-0.1, 0.08, 0.9))

    def run(pixel_noise, world_noise):
        errs = []
        for centre in centres:
            norm, noisy_world = [], []
            for x, y, z in rel:
                X, Y, Z = centre[0] + x, centre[1] + y, centre[2] + z
                u = f * X / Z + W / 2 + rng.gauss(0, pixel_noise)
                v = f * Y / Z + H / 2 + rng.gauss(0, pixel_noise)
                norm.append(types.SimpleNamespace(x=u / W, y=v / H))
                noisy_world.append((x + rng.gauss(0, world_noise), y + rng.gauss(0, world_noise),
                                    z + rng.gauss(0, world_noise)))
            root = sender.estimate_root(norm, noisy_world, W, H, FOV)
            truth = (centre[0] + rel[0][0], centre[1] + rel[0][1], centre[2] + rel[0][2])
            errs.append(math.dist(root, truth))
        return errs

    clean = run(0.0, 0.0)
    noisy = run(0.7, 0.005)  # 0.7 px landmark noise, 5 mm world-landmark noise
    print(f"  wrist position error, clean data: {[round(e * 1000, 2) for e in clean]} mm; "
          f"noisy data: {[round(e * 1000, 1) for e in noisy]} mm (at 0.5/0.7/0.9 m)")
    assert max(clean) < 0.001, clean
    assert max(noisy) < 0.05, noisy

    # hand pointing straight at the camera: no lateral extent, so no estimate
    flat = [(0.0, 0.0, 0.02 * i) for i in range(21)]
    norm = [types.SimpleNamespace(x=0.5, y=0.5)] * 21
    assert sender.estimate_root(norm, flat, W, H, FOV) is None


def test_translation_logic():
    st = br.State()
    st.scale = 4.0
    assert br.root_translation(st, None) == (0.0, 0.0, 0.0)
    assert br.root_translation(st, [0.1, 0.2, 0.5]) == (0.0, 0.0, 0.0), "first position is the anchor"
    d = br.root_translation(st, [0.2, 0.2, 0.6])
    assert all(abs(a - b) < 1e-9 for a, b in zip(d, (0.4, 0.4, 0.0))), d  # x, z=depth->y, -y->z
    assert br.root_translation(st, None) == d, "no root: keep the last translation"
    br.TRANSLATE_GAIN = (1.0, 0.0, 1.0)
    d = br.root_translation(st, [0.2, 0.2, 0.9])
    br.TRANSLATE_GAIN = (1.0, 1.0, 1.0)
    assert d[1] == 0.0, "an axis with gain 0 stays frozen"


def test_axis_mapping_is_a_rotation():
    cols = [br.to_blender_axes(*v) for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    det = (cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1])
           - cols[1][0] * (cols[0][1] * cols[2][2] - cols[0][2] * cols[2][1])
           + cols[2][0] * (cols[0][1] * cols[1][2] - cols[0][2] * cols[1][1]))
    assert det == 1, f"mapping must not mirror the hand (det={det})"


def _rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return {"x": ((1, 0, 0), (0, c, -s), (0, s, c)),
            "y": ((c, 0, s), (0, 1, 0), (-s, 0, c)),
            "z": ((c, -s, 0), (s, c, 0), (0, 0, 1))}[axis]


def _rig_points():
    """wrist, index, middle, pinky knuckles of a small right-hand-shaped rig (fingers +Z)."""
    return {"wrist": (0.0, 0.0, 0.0), "index": (0.05, 0.05, 0.38),
            "middle": (-0.04, 0.05, 0.39), "pinky": (-0.19, 0.02, 0.35)}


def test_palm_math():
    p = _rig_points()
    frame = br.palm_frame(p["wrist"], p["index"], p["middle"], p["pinky"])
    x, y, z = frame
    assert abs(br._dot(x, y)) < 1e-9 and abs(br._dot(y, z)) < 1e-9 and abs(br._dot(x, z)) < 1e-9
    assert all(abs(br._dot(a, a) - 1) < 1e-9 for a in frame)
    assert max(abs(a - b) for a, b in zip(br._cross(x, y), z)) < 1e-9, "frame must be right-handed"

    q = br._matmul(_rot("z", 120), br._matmul(_rot("x", -40), _rot("y", 25)))
    live = {k: br._matvec(q, v) for k, v in p.items()}
    live_frame = br.palm_frame(live["wrist"], live["index"], live["middle"], live["pinky"])
    r = br.rotation_between(frame, live_frame)
    assert max(abs(r[i][j] - q[i][j]) for i in range(3) for j in range(3)) < 1e-9, "must recover the rotation"
    assert br.palm_frame((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0)) is None, "degenerate points"
    inv = br._inv3(q)
    ident = br._matmul(q, inv)
    assert max(abs(ident[i][j] - (i == j)) for i in range(3) for j in range(3)) < 1e-9


def test_mirror_votes():
    flat = [(0.0, 0.0, 0.0)] * 21  # no usable palm: only the label can speak
    st = br.State()
    send = lambda label, score=1.0: br.update_mirror(st, {"hand": label, "score": score}, flat)
    send("Left")  # MediaPipe "Left" means the measured shape is Right, like the rig
    assert st.mirror is False
    send("Right")
    for _ in range(14):
        send("Left")
    assert st.mirror is False, "one stray label must not flip the rig"
    for _ in range(14):
        send("Right")
    assert st.mirror is True, "a sustained label change flips it"
    send("Unknown")
    assert st.mirror is True, "an unknown label changes nothing"
    st2 = br.State()
    for _ in range(14):
        br.update_mirror(st2, {"hand": "Right", "score": 0.05}, flat)  # unconfident
    for _ in range(14):
        br.update_mirror(st2, {"hand": "Left", "score": 0.9}, flat)
    assert st2.mirror is False
    br.MIRROR_OVERRIDE = True
    send("Left")
    forced = st.mirror
    br.MIRROR_OVERRIDE = None
    assert forced is True and br.hand_shape("Right") == "Left" and br.hand_shape("x") is None


def test_drive_armature_with_fake_armature():
    m0 = _rot("z", -199)
    rows = [list(m0[i]) + [0.0] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
    p = _rig_points()
    local = {k: br._matvec(br._inv3(m0), v) for k, v in p.items()}  # armature-local rest tails
    bone = lambda t: types.SimpleNamespace(head_local=(0.0, 0.0, 0.0), tail_local=t)
    arm = types.SimpleNamespace(
        matrix_world=tuple(map(tuple, rows)), location=(0.0, 0.0, 0.0),
        data=types.SimpleNamespace(bones={br.PALM_BONES["index"]: bone(local["index"]),
                                          br.PALM_BONES["middle"]: bone(local["middle"]),
                                          br.PALM_BONES["pinky"]: bone(local["pinky"])}))
    br.bpy = types.SimpleNamespace(data=types.SimpleNamespace(objects=types.SimpleNamespace(
        get=lambda n: arm if n == br.ARMATURE_NAME else None)))
    rig = br.Rig(arm)
    st = br.State()
    st.rig = rig
    ident = br._IDENTITY
    q = br._matmul(_rot("z", 180), _rot("x", 30))

    def rel_for(rot, reflect):
        pts = {}
        for k, v in p.items():
            v = br._matvec(rig.reflect, br._matvec(br._inv3(rig.m0), v)) if reflect else v
            v = br._matvec(rig.m0, v) if reflect else v
            pts[k] = br._matvec(rot, v)
        rel = [(0.0, 0.0, 0.0)] * 21
        rel = list(rel)
        rel[br._WRIST], rel[br._INDEX_MCP] = pts["wrist"], pts["index"]
        rel[br._MIDDLE_MCP], rel[br._PINKY_MCP] = pts["middle"], pts["pinky"]
        return rel

    st.mirror = False
    assert br.drive_armature(st, rel_for(q, False), (0.0, 0.0, 0.0))
    got = [[arm.matrix_world[i][j] for j in range(3)] for i in range(3)]
    want = br._matmul(q, m0)
    assert max(abs(got[i][j] - want[i][j]) for i in range(3) for j in range(3)) < 1e-9

    st.mirror = True  # a mirrored hand: the armature matrix gets a negative determinant
    assert br.drive_armature(st, rel_for(q, True), (0.1, 0.2, 0.3))
    a = [[arm.matrix_world[i][j] for j in range(3)] for i in range(3)]
    det = br._dot(a[0], br._cross(a[1], a[2]))
    assert det < 0 and abs(abs(det) - 1) < 1e-9, det
    assert tuple(arm.matrix_world[i][3] for i in range(3)) == (0.1, 0.2, 0.3)
    br.restore_armature(st)
    assert arm.matrix_world == rig.matrix0
    br.bpy = None


def _fake_rig_arm(m0=None):
    """A synthetic armature: 5 chains with planar finger bends about the X axis. m0 is the
    object's 3x3 transform as saved in the file (identity by default)."""
    def rx(deg, v=(0.0, 0.0, 1.0)):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        return (v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c)

    lateral = {"index": 0.05, "middle": 0.0, "ring": -0.05, "pinky": -0.1, "thumb": 0.12}
    bones = {}
    for finger, names in br.BONE_CHAINS.items():
        root = br._unit((lateral[finger], 0.0, 0.38))
        if finger == "thumb":
            dirs = [br._unit((0.3, 0.1, 0.9)), br._unit((0.5, 0.0, 0.8)),
                    br._unit((0.4, -0.2, 0.9)), br._unit((0.2, -0.3, 0.9))]
            lengths = [0.24, 0.09, 0.09, 0.12]
        else:  # rest bends: knuckle 10, middle joint 50, tip joint 30 degrees
            dirs = [root, rx(10), rx(60), rx(90)]
            lengths = [0.39, 0.18, 0.1, 0.09]
        pos = (0.0, 0.0, 0.0)
        for n, d, length in zip(names, dirs, lengths):
            tail = tuple(p + length * c for p, c in zip(pos, d))
            bones[n] = types.SimpleNamespace(head_local=pos, tail_local=tail)
            pos = tail
    m0 = m0 or br._IDENTITY
    rows = tuple(tuple(m0[i]) + (0.0,) for i in range(3)) + ((0.0, 0.0, 0.0, 1.0),)
    return types.SimpleNamespace(matrix_world=rows, data=types.SimpleNamespace(bones=bones))


def _rot_about(axis, deg, v):
    """Rodrigues rotation of v about a unit axis."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    k, d = br._cross(axis, v), br._dot(axis, v)
    return tuple(v[i] * c + k[i] * s + axis[i] * d * (1 - c) for i in range(3))


def _live_finger_dirs(chain, bends, tilt=0.0):
    """Segment directions of a real-shaped finger: the proximal, middle and distal segments lie in
    one flat plane (normal = the rig's hinge tilted by `tilt` degrees about the palm direction)
    and bend there by the joint flexion bends = (mcp, pip, dip) degrees; the palm segment is
    whatever the rig has, so a sideways-angled finger is abducted out of that plane."""
    root = chain.rest_dirs[0]
    normal = _rot_about(root, tilt, chain.hinge)
    u = br._unit(br._sub(root, br._scale(normal, br._dot(root, normal))))
    w = br._cross(normal, u)
    dirs, cum = [root], 0.0
    for b in bends:
        cum += math.radians(b)
        dirs.append(tuple(math.cos(cum) * a + math.sin(cum) * c for a, c in zip(u, w)))
    return dirs


def _hand_points(rig, bends, factors=None, jitter=0.0, rng=None, tilt=0.0):
    """Live landmarks (armature-local) of a hand with the rig's layout and its own bone lengths:
    bends[finger] = joint flexion (mcp, pip, dip) in degrees."""
    pts = [(0.0, 0.0, 0.0)] * 21
    for finger, chain in rig.chains.items():
        dirs = chain.rest_dirs if chain.hinge is None else _live_finger_dirs(chain, bends[finger], tilt)
        pos = (0.0, 0.0, 0.0)
        for s in range(4):
            factor = (factors or {}).get(finger, [1.0] * 4)[s]
            if jitter:
                factor *= 1 + jitter * rng.gauss(0, 1)
            pos = tuple(p + chain.lengths[s] * factor * c for p, c in zip(pos, dirs[s]))
            pts[br.CHAIN_LANDMARKS[finger][s + 1]] = pos
    return pts


def _flexions(axis, points):
    """Signed joint flexion (degrees) at MCP, PIP, DIP of one chain's 5 points, about axis."""
    d = br._segment_dirs(points)
    return [math.degrees(math.atan2(br._dot(axis, br._cross(d[s], d[s + 1])), br._dot(d[s], d[s + 1])))
            for s in range(3)]


FOUR = ("index", "middle", "ring", "pinky")


def test_retarget():
    """The retargeting contract (live directions, rig lengths, limits) without finger collision, which
    changes the pose on purpose where fingers overlap and has its own tests."""
    saved = br.COLLISION
    br.COLLISION = False
    try:
        _check_retarget()
    finally:
        br.COLLISION = saved


def _check_retarget():
    rig = br.Rig(_fake_rig_arm())
    assert rig.chains is not None
    rest = [(0.0, 0.0, 0.0)] * 21
    for finger, chain in rig.chains.items():
        pos = (0.0, 0.0, 0.0)
        for i, d, length in zip(br.CHAIN_LANDMARKS[finger][1:], chain.rest_dirs, chain.lengths):
            pos = tuple(p + length * c for p, c in zip(pos, d))
            rest[i] = pos
    out = br.retarget(rig, rest, True)
    assert max(br._dot(br._sub(a, b), br._sub(a, b)) for a, b in zip(out, rest)) < 1e-18, "rest is a fixed point"

    rng = random.Random(9)
    factors = {f: [rng.uniform(0.8, 1.3) for _ in range(4)] for f in br.BONE_CHAINS}
    for tilt in (0.0, 20.0, 35.0):  # real fingers curl in tilted planes: the live plane must be followed
        # PIP/DIP bends stay above the straight dead zone, which changes small bends on purpose
        bends = {f: (rng.uniform(0, 60), rng.uniform(25, 100), rng.uniform(25, 80)) for f in FOUR}
        live = _hand_points(rig, bends, factors, 0.08, rng, tilt)  # other proportions, noisy lengths
        out = br.retarget(rig, live, True)
        for finger, chain in rig.chains.items():
            idx = br.CHAIN_LANDMARKS[finger]
            for s in range(4):
                seg = br._sub(out[idx[s + 1]], out[idx[s]])
                assert abs(math.sqrt(br._dot(seg, seg)) - chain.lengths[s]) < 1e-12, "rig bone length"
                live_d = br._unit(br._sub(live[idx[s + 1]], live[idx[s]]))
                assert br._dot(live_d, br._unit(seg)) > 1 - 1e-9, (tilt, finger, s, "direction kept")
    scaled = [tuple(3.7 * c for c in p) for p in live]  # uniform scale of the live hand changes nothing
    assert max(abs(a - b) for p, q in zip(br.retarget(rig, scaled, True), out) for a, b in zip(p, q)) < 1e-12

    # beyond the limits the joints stop at the anatomical range, hyperextension at the lower one
    chain = rig.chains["middle"]
    idx = br.CHAIN_LANDMARKS["middle"]
    limits = [br.FLEXION_RANGE[j] for j in br._JOINTS]
    over = br.retarget(rig, _hand_points(rig, {f: (120.0, 160.0, 140.0) for f in FOUR}), True)
    for value, (lo, hi) in zip(_flexions(chain.hinge, [over[i] for i in idx]), limits):
        assert abs(value - hi) < 1e-6, ("stops at the upper limit", value)
    free = br.retarget(rig, _hand_points(rig, {f: (120.0, 160.0, 140.0) for f in FOUR}), False)
    assert _flexions(chain.hinge, [free[i] for i in idx])[1] > 150, "without limits the bend is kept"
    back = br.retarget(rig, _hand_points(rig, {f: (-60.0, -30.0, -30.0) for f in FOUR}), True)
    for value, (lo, hi) in zip(_flexions(chain.hinge, [back[i] for i in idx]), limits):
        assert abs(value - lo) < 1e-6, ("stops at the lower limit", value)

    # A segment pushed out of the finger's plane is pulled back: relative to the plane the limiter
    # fits to the live finger, no PIP/DIP joint swings out by more than SIDE_MARGIN_DEG.
    dirs = _live_finger_dirs(chain, (30.0, 80.0, 50.0))
    dirs[3] = _rot_about(dirs[2], 25.0, dirs[3])
    b = br._bend_axis(dirs[1:])
    side = 1.0 if br._dot(b, chain.hinge) >= 0 else -1.0
    plane = br._unit(br._add(br._scale(b, side), br._scale(chain.hinge, br.PLANE_BLEND)))
    swing = lambda ds: [math.degrees(math.asin(br._dot(d, plane))) for d in ds]
    live_pts = [(0.0, 0.0, 0.0)]
    for s in range(4):
        live_pts.append(tuple(p + chain.lengths[s] * c for p, c in zip(live_pts[-1], dirs[s])))
    out_dirs = br._segment_dirs(chain.retarget(live_pts, True))
    before, after = swing(dirs), swing(out_dirs)
    assert abs(before[3] - before[2]) > 8.0, "the test finger really is out of plane"
    for s in (2, 3):
        assert abs(after[s] - after[s - 1]) <= br.SIDE_MARGIN_DEG + 1e-6, (before, after)

    # degenerate live segments fall back to the rig's rest direction instead of failing
    assert len(br.retarget(rig, [(0.0, 0.0, 0.0)] * 21, True)) == 21


def test_straight_dead_zone():
    lo, hi = br.STRAIGHT_ZONE_DEG["pip"]
    assert lo > 0 and hi > lo
    # the curve: 0 inside the zone, the input above it, and never decreasing in between
    assert br._straighten(math.radians(lo - 0.01), (lo, hi)) == 0.0
    assert br._straighten(math.radians(hi + 0.01), (lo, hi)) == math.radians(hi + 0.01)
    previous = 0.0
    for tenth in range(0, 600):
        out = br._straighten(math.radians(tenth / 10.0), (lo, hi))
        assert out >= previous - 1e-12 and out <= math.radians(tenth / 10.0) + 1e-12, tenth
        previous = out
    assert br._straighten(math.radians(hi), (lo, hi)) == math.radians(hi), "continuous at the top"

    rig = br.Rig(_fake_rig_arm())
    chain, idx = rig.chains["middle"], br.CHAIN_LANDMARKS["middle"]

    def flexions(bends, apply_limits=True):
        live = _hand_points(rig, {f: bends for f in FOUR})
        out = chain.retarget([live[i] for i in idx], apply_limits)
        for s in range(4):  # still the rig's own bone lengths: the empty can be reached
            seg = br._sub(out[s + 1], out[s])
            assert abs(math.sqrt(br._dot(seg, seg)) - chain.lengths[s]) < 1e-12
        return _flexions(chain.hinge, out)

    # a flat finger that MediaPipe reads as 5 degrees bent at PIP and DIP is shown straight
    for pip, dip in ((5.0, 5.0), (0.0, 4.0), (5.9, 2.0)):
        f = flexions((3.0, pip, dip))
        assert abs(f[1]) < 1e-6 and abs(f[2]) < 1e-6, (pip, dip, f)
        assert abs(f[0] - 3.0) < 1e-6, "the knuckle is not part of the dead zone"
    # a real curl is not flattened; in between it fades in (never more than the live bend)
    f = flexions((20.0, 60.0, 40.0))
    assert abs(f[1] - 60.0) < 1e-6 and abs(f[2] - 40.0) < 1e-6, f
    f = flexions((20.0, 12.0, 12.0))
    assert 0.5 < f[1] < 12.0 and 0.5 < f[2] < 12.0, f
    # switched off, or with limits off, the small bend is kept
    saved = dict(br.STRAIGHT_ZONE_DEG)
    try:
        br.STRAIGHT_ZONE_DEG.clear()
        assert abs(flexions((3.0, 5.0, 5.0))[1] - 5.0) < 1e-6
    finally:
        br.STRAIGHT_ZONE_DEG.update(saved)
    assert abs(flexions((3.0, 5.0, 5.0), apply_limits=False)[1] - 5.0) < 1e-6


def _capsule_depths(rig, out):
    """How far each neighbouring-finger segment pair overlaps (beyond what the rest pose allows)."""
    col, depths = rig.collision, []
    for (fa, fb), entries in col.pairs.items():
        pa = [out[i] for i in br.CHAIN_LANDMARKS[fa]]
        pb = [out[i] for i in br.CHAIN_LANDMARKS[fb]]
        for sa, sb, radii_sum, gap in entries:
            need = min(radii_sum * br.COLLISION_MARGIN, gap * 0.98)
            _, _, c1, c2 = br._closest_segments(pa[sa], pa[sa + 1], pb[sb], pb[sb + 1])
            depths.append(max(0.0, need - math.dist(c1, c2)))
    return depths


def _swing_finger(rig, points, finger, degrees):
    """Turn a finger about its knuckle, across the palm (about the palm normal)."""
    pts = list(points)
    n = rig.collision.palm["n"]
    idx = br.CHAIN_LANDMARKS[finger]
    pivot = pts[idx[1]]
    for i in idx[2:]:
        rel = br._sub(pts[i], pivot)
        pts[i] = br._add(pivot, _rot_about(n, degrees, rel))
    return pts


def test_finger_collision():
    rig = br.Rig(_fake_rig_arm())
    assert rig.collision is not None and rig.collision.palm is not None
    flat = _hand_points(rig, {f: (0.0, 0.0, 0.0) for f in FOUR})

    # the rest pose is the contact the mesh was made with: nothing moves
    rest = [(0.0, 0.0, 0.0)] * 21
    for finger, chain in rig.chains.items():
        pos = (0.0, 0.0, 0.0)
        for i, d, length in zip(br.CHAIN_LANDMARKS[finger][1:], chain.rest_dirs, chain.lengths):
            pos = tuple(p + length * c for p, c in zip(pos, d))
            rest[i] = pos
    assert max(br._dot(br._sub(a, b), br._sub(a, b)) for a, b in zip(br.retarget(rig, rest, True), rest)) < 1e-18
    # an open hand with no overlap is left exactly as retargeting makes it
    br.COLLISION = False
    free_flat = br.retarget(rig, flat, True)
    br.COLLISION = True
    assert br.retarget(rig, flat, True) == free_flat

    # the index finger swung across the middle finger overlaps it; collision pushes them apart
    best = None
    for sign in (1.0, -1.0):
        crossed = _swing_finger(rig, flat, "index", sign * 22.0)
        br.COLLISION = False
        free = br.retarget(rig, crossed, True)
        br.COLLISION = True
        depth = max(_capsule_depths(rig, free))
        if best is None or depth > best[0]:
            best = (depth, crossed, free, sign)
    depth_before, crossed, free, sign_of_overlap = best
    assert depth_before > 0.008, depth_before
    solved = br.retarget(rig, crossed, True)
    depth_after = max(_capsule_depths(rig, solved))  # the joint limits applied after the push give a little back
    assert depth_after < 0.25 * depth_before and depth_after < 0.012, (depth_before, depth_after)
    for finger, chain in rig.chains.items():  # same bones, same knuckles, finite
        idx = br.CHAIN_LANDMARKS[finger]
        assert solved[idx[1]] == free[idx[1]], "the knuckle never moves"
        for s in range(4):
            seg = br._sub(solved[idx[s + 1]], solved[idx[s]])
            assert abs(math.sqrt(br._dot(seg, seg)) - chain.lengths[s]) < 1e-9, (finger, s)
    # the joint limits still hold after the push
    for finger in FOUR:
        chain, idx = rig.chains[finger], br.CHAIN_LANDMARKS[finger]
        for value, joint in zip(_flexions(chain.hinge, [solved[i] for i in idx]), br._JOINTS):
            lo, hi = br.FLEXION_RANGE[joint]
            assert lo - 2.0 <= value <= hi + 2.0, (finger, joint, value)
    # no popping: sweeping the index finger across the middle finger in 1 degree steps, the result
    # moves smoothly (a tip is about 0.7 units from the wrist, so 1 degree is about 0.012 of travel;
    # the step where a contact first engages may be a few times that, the real recording is checked
    # for jumps in test_collision_headless.py)
    sweep = [br.retarget(rig, _swing_finger(rig, flat, "index", sign_of_overlap * d), True) for d in range(0, 31)]
    jumps = [max(math.dist(p, q) for p, q in zip(a, b)) for a, b in zip(sweep, sweep[1:])]
    assert max(jumps) < 0.08, max(jumps)
    # switched off, the fingers go through each other
    br.COLLISION = False
    try:
        assert max(_capsule_depths(rig, br.retarget(rig, crossed, True))) == depth_before
    finally:
        br.COLLISION = True

    # A fingertip pushed down into the palm comes out of it. The fake rig's fingers are short, so the
    # test palm is made twice as thick for this check, and the joint limits are off so tips can sink.
    saved_thickness = br.PALM_THICKNESS
    br.PALM_THICKNESS = 2.0
    try:
        prig = br.Rig(_fake_rig_arm())
        palm = prig.collision.palm
        heights = lambda pose: [br._dot(pose[br.CHAIN_LANDMARKS[f][k]], palm["n"]) - palm["offset"]
                                - prig.chains[f].radii[k - 1] * br.COLLISION_MARGIN
                                for f in FOUR for k in (3, 4)
                                if 0 <= br._dot(pose[br.CHAIN_LANDMARKS[f][k]], palm["u"]) < palm["a_mcp"][f]]
        curled = _hand_points(prig, {f: (90.0, 150.0, 90.0) for f in FOUR})
        br.COLLISION = False
        sunk = br.retarget(prig, curled, False)
        br.COLLISION = True
        assert heights(sunk) and min(heights(sunk)) < -0.01, "the test hand really sinks into the palm"
        lifted = br.retarget(prig, curled, False)
        assert min(heights(lifted)) > -0.003, (min(heights(sunk)), min(heights(lifted)))
        for finger, chain in prig.chains.items():  # same bones
            idx = br.CHAIN_LANDMARKS[finger]
            for s in range(4):
                assert abs(math.dist(lifted[idx[s]], lifted[idx[s + 1]]) - chain.lengths[s]) < 1e-9
        br.PALM_COLLISION = False
        try:
            assert min(heights(br.retarget(prig, curled, False))) < -0.01
        finally:
            br.PALM_COLLISION = True
    finally:
        br.PALM_THICKNESS = saved_thickness


def test_fit_triangulation():
    """fit_skeleton.py's maths: landmarks seen by several orthographic cameras are put back in 3D."""
    import numpy as np
    import fit_skeleton as fs
    rng = random.Random(1)
    n, u, lat = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    defs = fs.make_views(n, u, lat)
    assert len(defs) == 1 + 4 * len(fs.TILTS_DEG) and defs[0][0] == "palm"
    views = []
    for name, towards, up in defs:
        x, y, z = fs.camera_axes(towards, up)
        assert abs(np.dot(x, y)) < 1e-9 and abs(np.dot(y, z)) < 1e-9 and abs(np.linalg.norm(x) - 1) < 1e-9
        views.append({"x": x, "y": y, "z": z, "origin": np.array([0.1, 0.2, 0.3]) + z * 4.0, "ortho_scale": 0.9})
    points = [np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.1, 0.1), rng.uniform(0, 0.7)]) for _ in range(150)]
    for p in points:  # exact pictures give the exact point
        q, residual, used = fs.triangulate({i: fs.project(p, v) for i, v in enumerate(views)}, views)
        assert np.linalg.norm(q - p) < 1e-9 and residual < 1e-9 and used == len(views)
    errors = []  # 4 pixels of noise in a 1024 picture, on every view
    for p in points:
        obs = {i: tuple(c + rng.gauss(0, 4 / 1024) for c in fs.project(p, v)) for i, v in enumerate(views)}
        errors.append(float(np.linalg.norm(fs.triangulate(obs, views)[0] - p)))
    assert sum(errors) / len(errors) < 0.006, sum(errors) / len(errors)  # the hand is 0.7 long
    p = np.array([0.05, 0.0, 0.4])  # one picture with the landmark somewhere else is ignored
    obs = {i: fs.project(p, v) for i, v in enumerate(views)}
    obs[2] = (obs[2][0] + 0.15, obs[2][1] - 0.1)
    q, residual, used = fs.triangulate(obs, views)
    assert used == len(views) - 1 and np.linalg.norm(q - p) < 1e-6
    q, residual, used = fs.triangulate({0: fs.project(p, views[0]), 1: fs.project(p, views[1]), 2: (0.9, 0.9)}, views)
    assert used == 3, "never fewer than three pictures"
    # the bone shares of a finger and the phalanx order
    joints = {5: (0, 0, 0), 6: (0, 0, 0.45), 7: (0, 0, 0.75), 8: (0, 0, 1.0)}
    assert all(abs(a - b) < 1e-9 for a, b in zip(fs.phalanx_ratios({**{k: (0, 0, 0) for k in range(21)}, **joints})["index"],
                                                  (0.45, 0.30, 0.25)))


def test_hand_yaw():
    """HAND_TURN_ABOUT = "world": HAND_YAW_DEG turns each hand about the vertical axis through its wrist."""
    names = list(br.LANDMARK_NAMES)
    pts = make_hand(span=0.09)
    pts[8] = (0.01, -0.05, 0.02)
    pts[4] = (-0.03, -0.02, -0.01)
    saved, saved_mode = br.HAND_YAW_DEG, br.HAND_TURN_ABOUT
    br.HAND_TURN_ABOUT = "world"

    def place(yaw, origin=(0.0, 0.0, 0.0)):
        empties, _ = install_fake_bpy(names)
        br.HAND_YAW_DEG = yaw
        st = br.State()
        st.origin = origin
        br._apply_frame(st, {"lm": pts, "hand": "Right"})
        got = {n: tuple(e.location) for n, e in empties.items()}
        br.bpy = None
        return got

    try:
        base = place(0.0)
        for yaw in (180.0, 90.0, -35.0):
            turned = place(yaw)
            c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            for n in names:
                x, y, z = base[n]
                want = (c * x - s_ * y, s_ * x + c * y, z)
                assert all(abs(a - b) < 1e-9 for a, b in zip(turned[n], want)), (yaw, n, turned[n], want)
        half = place(180.0)
        # a hand that sits elsewhere (the second rig) turns about its own wrist, not about the world origin
        beside = place(180.0, origin=(0.8, 0.0, 0.0))
        assert all(abs(beside[n][0] - 0.8 - half[n][0]) < 1e-9 and abs(beside[n][1] - half[n][1]) < 1e-9
                   and abs(beside[n][2] - half[n][2]) < 1e-9 for n in names)
        assert beside["WRIST"] == (0.8, 0.0, 0.0)
        # 0 changes nothing, a proper rotation never mirrors (the palm frame stays right-handed)
        assert br.yaw_about_z(pts, 0.0) == list(pts)
        rel = br.world_to_blender(pts, 4.0)
        frame = br.palm_frame(rel[0], rel[5], rel[9], rel[17])
        turned_frame = br.palm_frame(*[br.yaw_about_z(rel, 180.0)[i] for i in (0, 5, 9, 17)])
        det = lambda f: br._dot(f[0], br._cross(f[1], f[2]))
        assert det(frame) > 0.99 and det(turned_frame) > 0.99
    finally:
        br.HAND_YAW_DEG, br.HAND_TURN_ABOUT = saved, saved_mode


def test_unmirror():
    """UNMIRROR_HAND reflects the depth (Blender Y) of every hand and of its translation, nothing else."""
    names = list(br.LANDMARK_NAMES)
    pts = make_hand(span=0.09)
    pts[8] = (0.01, -0.05, 0.02)
    pts[4] = (-0.03, -0.02, -0.01)
    saved = (br.UNMIRROR_HAND, br.HAND_YAW_DEG)

    def place(unmirror, yaw=0.0):
        empties, _ = install_fake_bpy(names)
        br.UNMIRROR_HAND, br.HAND_YAW_DEG = unmirror, yaw
        st = br.State()
        br._apply_frame(st, {"lm": pts, "hand": "Right"})
        got = {n: tuple(e.location) for n, e in empties.items()}
        br.bpy = None
        return got

    try:
        raw, fixed = place(False), place(True)
        for n in names:  # only Y changes sign
            assert all(abs(a - b) < 1e-9 for a, b in zip(fixed[n], (raw[n][0], -raw[n][1], raw[n][2]))), n
        # the reflection really turns a right-hand shape into a left-hand one: the palm frame's handedness
        rel = br.world_to_blender(pts, 4.0)
        mirrored = [(x, -y, z) for x, y, z in rel]
        thumb_side = lambda p: br._dot(br._sub(p[4], p[0]), br._cross(br._sub(p[5], p[0]), br._sub(p[17], p[0])))
        assert thumb_side(rel) * thumb_side(mirrored) < 0
        # depth is the only thing that flips, so a turn about Z still composes with it
        both = place(True, 90.0)
        c, s_ = 0.0, 1.0
        for n in names:
            x, y, z = fixed[n]
            assert all(abs(a - b) < 1e-9 for a, b in zip(both[n], (c * x - s_ * y, s_ * x + c * y, z))), n
        # the translation follows the same reflection
        st_raw, st_fixed = br.State(), br.State()
        st_raw.scale = st_fixed.scale = 4.0
        br.UNMIRROR_HAND = False
        br.root_translation(st_raw, (0.0, 0.0, 0.5))
        raw_delta = br.root_translation(st_raw, (0.05, -0.02, 0.6))
        br.UNMIRROR_HAND = True
        br.root_translation(st_fixed, (0.0, 0.0, 0.5))
        fixed_delta = br.root_translation(st_fixed, (0.05, -0.02, 0.6))
        assert abs(fixed_delta[0] - raw_delta[0]) < 1e-9 and abs(fixed_delta[1] + raw_delta[1]) < 1e-9             and abs(fixed_delta[2] - raw_delta[2]) < 1e-9, (raw_delta, fixed_delta)
        # as shipped, the receiver undoes the camera's mirror and adds no extra turn
        text = Path(br.__file__).read_text(encoding="utf-8").splitlines()
        assert "UNMIRROR_HAND = True" in text and "HAND_YAW_DEG = 180.0" in text
        assert any(line.startswith('HAND_TURN_ABOUT = "hand"') for line in text)
    finally:
        br.UNMIRROR_HAND, br.HAND_YAW_DEG = saved


def test_turn_about_the_hands_own_axis():
    """The default turn: 180 degrees about the line from the wrist through the middle knuckle."""
    rig = br.Rig(_fake_rig_arm())
    rng = random.Random(8)
    pts = _hand_points(rig, {f: (rng.uniform(0, 40), rng.uniform(10, 70), rng.uniform(5, 50)) for f in FOUR})
    pts = [br._matvec(_random_rotation(rng), p) for p in pts]  # held at some angle
    saved = br.HAND_TURN_ABOUT
    br.HAND_TURN_ABOUT = "hand"
    try:
        turned = br.turn_hand(pts, 180.0)
        assert br.turn_hand(pts, 0.0) == list(pts)
        d = lambda a, b: math.dist(a, b)
        assert all(abs(d(pts[i], pts[j]) - d(turned[i], turned[j])) < 1e-12 for i in range(21) for j in range(i))  # rigid
        assert d(turned[br._WRIST], pts[br._WRIST]) < 1e-12 and d(turned[br._MIDDLE_MCP], pts[br._MIDDLE_MCP]) < 1e-12
        frame = br.palm_frame(pts[0], pts[5], pts[9], pts[17])
        after = br.palm_frame(turned[0], turned[5], turned[9], turned[17])
        assert br._dot(frame[1], after[1]) > 1 - 1e-12, "the fingers keep pointing where they pointed"
        assert br._dot(frame[0], after[0]) < -1 + 1e-12 and br._dot(frame[2], after[2]) < -1 + 1e-12, "palm and back swap sides"
        # a turn, not a mirror: a right hand stays a right hand
        side = lambda p: br._dot(br._sub(p[4], p[0]), br._cross(br._sub(p[5], p[0]), br._sub(p[17], p[0])))
        assert side(pts) * side(turned) > 0
        # turning twice by 90 is turning once by 180, and 360 is no turn
        twice = br.turn_hand(br.turn_hand(pts, 90.0), 90.0)
        assert max(d(a, b) for a, b in zip(twice, turned)) < 1e-12
        assert max(d(a, b) for a, b in zip(br.turn_hand(pts, 360.0), pts)) < 1e-12
        # THE POINT: the turn goes with the hand. Tilt, roll or swing the real hand by any rotation R and the
        # turned hand is tilted, rolled or swung by the same R, so every movement keeps its direction.
        for _ in range(8):
            r = _random_rotation(rng)
            moved = [br._matvec(r, p) for p in pts]
            want = [br._matvec(r, p) for p in turned]
            assert max(d(a, b) for a, b in zip(br.turn_hand(moved, 180.0), want)) < 1e-9
        # a turn about the fixed vertical axis does not: tilting the real hand tilts the turned hand the other way
        br.HAND_TURN_ABOUT = "world"
        r = _rot("y", 25)  # fingers tilted 25 degrees sideways
        moved = [br._matvec(r, p) for p in pts]
        want = [br._matvec(r, p) for p in br.turn_hand(pts, 180.0)]
        assert max(d(a, b) for a, b in zip(br.turn_hand(moved, 180.0), want)) > 0.01
        # no measurable palm: nothing happens and nothing breaks
        br.HAND_TURN_ABOUT = "hand"
        flat = [(0.0, 0.0, 0.0)] * 21
        assert br.turn_hand(flat, 180.0) == flat
    finally:
        br.HAND_TURN_ABOUT = saved


def test_movement_directions_match_the_user():
    """Your frame: x to your right, y forward (toward the webcam), z up. The sender's root position is in the
    flipped picture's camera frame (x right, y down, z away from the camera)."""
    saved = br.UNMIRROR_HAND
    br.UNMIRROR_HAND = True
    try:
        def moved(dx=0.0, dy=0.0, dz=0.0):
            st = br.State()
            st.scale = 4.0
            br.root_translation(st, (0.0, 0.0, 0.5))
            return br.root_translation(st, (dx, dy, 0.5 + dz))

        right, toward_camera, up = moved(dx=0.1), moved(dz=-0.1), moved(dy=-0.1)
        assert right[0] > 0.3 and abs(right[1]) < 1e-9 and abs(right[2]) < 1e-9, right      # to your right: +X
        assert toward_camera[1] > 0.3 and abs(toward_camera[0]) < 1e-9, toward_camera       # forward: +Y
        assert up[2] > 0.3 and abs(up[0]) < 1e-9, up                                        # up: +Z
        # and the turn of the hand's orientation does not touch where the hand goes
        for turn in (0.0, 180.0):
            br.HAND_YAW_DEG = turn
            assert moved(dx=0.1) == right
    finally:
        br.UNMIRROR_HAND = saved
        br.HAND_YAW_DEG = 0.0


def test_collision_radii_and_cost():
    # radii come from the bones when skin_mesh.py stored them, else from the segment lengths
    arm = _fake_rig_arm()
    for name, bone in arm.data.bones.items():
        bone.get = lambda key, name=name: {"v5_radius": 0.05}.get(key) if name == "index_mcp_to_index_pip" else None
    rig = br.Rig(arm)
    assert rig.chains["index"].radii[1] == 0.05
    assert abs(rig.chains["index"].radii[2] - br.DEFAULT_RADIUS[2] * rig.chains["index"].lengths[2]) < 1e-12
    # about what a tick can spare: random poses, average time of retarget() with collision on
    import time
    rig = br.Rig(_fake_rig_arm())
    rng = random.Random(3)
    poses = [_hand_points(rig, {f: (rng.uniform(0, 80), rng.uniform(0, 100), rng.uniform(0, 80)) for f in FOUR},
                          tilt=rng.uniform(-20, 20)) for _ in range(150)]
    start = time.perf_counter()
    for pose in poses:
        br.retarget(rig, pose, True)
    with_ms = (time.perf_counter() - start) / len(poses) * 1000
    br.COLLISION = False
    start = time.perf_counter()
    for pose in poses:
        br.retarget(rig, pose, True)
    without_ms = (time.perf_counter() - start) / len(poses) * 1000
    br.COLLISION = True
    print(f"collision cost: {with_ms:.2f} ms per frame with, {without_ms:.2f} ms without")
    assert with_ms < 5.0, with_ms


def test_rigid_palm():
    rig = br.Rig(_fake_rig_arm())
    rng = random.Random(5)
    bends = {f: (rng.uniform(0, 40), rng.uniform(25, 90), rng.uniform(25, 70)) for f in FOUR}
    live = _hand_points(rig, bends)
    # a palm that is not the rig's: every wrist-to-knuckle segment turned by 15 degrees
    skewed = list(live)
    for finger, chain in rig.chains.items():
        idx = br.CHAIN_LANDMARKS[finger]
        axis = br._unit(br._cross(chain.rest_dirs[0], (0.2, 0.9, 0.3)))
        turned = _rot_about(axis, 15.0, chain.rest_dirs[0])
        shift = br._sub(tuple(p + chain.lengths[0] * c for p, c in zip((0, 0, 0), turned)), live[idx[1]])
        for i in idx[1:]:
            skewed[i] = tuple(a + b for a, b in zip(live[i], shift))
    assert br.RIGID_PALM is True
    out = br.retarget(rig, skewed, True)
    for finger, chain in rig.chains.items():
        idx = br.CHAIN_LANDMARKS[finger]
        palm = br._unit(br._sub(out[idx[1]], out[idx[0]]))
        assert br._dot(palm, chain.rest_dirs[0]) > 1 - 1e-12, (finger, "the palm segment stays at rest")
        if finger != "thumb":  # the finger itself still follows its own live directions
            for s in (1, 2, 3):
                live_d = br._unit(br._sub(skewed[idx[s + 1]], skewed[idx[s]]))
                got = br._unit(br._sub(out[idx[s + 1]], out[idx[s]]))
                assert br._dot(live_d, got) > 1 - 1e-9, (finger, s)
    try:
        br.RIGID_PALM = False
        free = br.retarget(rig, skewed, True)
        idx = br.CHAIN_LANDMARKS["middle"]
        live_palm = br._unit(br._sub(skewed[idx[1]], skewed[idx[0]]))
        assert br._dot(br._unit(br._sub(free[idx[1]], free[idx[0]])), live_palm) > 1 - 1e-12, "off: live palm"
    finally:
        br.RIGID_PALM = True


def _thumb_points(chain, bend_mcp, bend_ip):
    """A thumb whose knuckle and tip joint bend by the given angles (degrees) about one axis."""
    d1 = chain.rest_dirs[1]
    axis = br._unit(br._cross(d1, (0.3, 0.8, 0.5)))
    d2 = _rot_about(axis, bend_mcp, d1)
    d3 = _rot_about(axis, bend_mcp + bend_ip, d1)
    dirs = [chain.rest_dirs[0], d1, d2, d3]
    pts = [(0.0, 0.0, 0.0)]
    for s in range(4):
        pts.append(tuple(p + chain.lengths[s] * c for p, c in zip(pts[-1], dirs[s])))
    return pts


def _angle_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, br._dot(a, b)))))


def test_thumb_limits():
    rig = br.Rig(_fake_rig_arm())
    chain = rig.chains["thumb"]
    assert chain.is_thumb and chain.hinge is None

    def bends(points):
        d = br._segment_dirs(points)
        return _angle_deg(d[1], d[2]), _angle_deg(d[2], d[3])

    # natural bends are kept exactly, with the rig's own bone lengths
    live = _thumb_points(chain, 35.0, 50.0)
    out = chain.retarget(live, True)
    want = br._segment_dirs(live)
    for got_d, want_d in zip(br._segment_dirs(out), want):
        assert br._dot(got_d, want_d) > 1 - 1e-12
    for s in range(4):
        seg = br._sub(out[s + 1], out[s])
        assert abs(math.sqrt(br._dot(seg, seg)) - chain.lengths[s]) < 1e-12
    # bends past the limits stop at them, in the same direction of bend
    over = chain.retarget(_thumb_points(chain, 110.0, 150.0), True)
    mcp, ip = bends(over)
    assert abs(mcp - br.THUMB_BEND_MAX_DEG["mcp"]) < 1e-6 and abs(ip - br.THUMB_BEND_MAX_DEG["ip"]) < 1e-6, (mcp, ip)
    # a segment folded straight back on the previous one (the bend direction is undefined) is fine
    back = _thumb_points(chain, 0.0, 0.0)
    back[4] = tuple(2 * a - b for a, b in zip(back[3], back[4]))  # tip folded back over the IP joint
    folded = chain.retarget(back, True)
    assert all(math.isfinite(c) for p in folded for c in p)
    assert abs(bends(folded)[1] - br.THUMB_BEND_MAX_DEG["ip"]) < 1e-6
    # with limits off nothing changes
    free = chain.retarget(_thumb_points(chain, 110.0, 150.0), False)
    assert bends(free)[0] > 100.0


def test_relax_when_hand_is_lost():
    rig = br.Rig(_fake_rig_arm())
    empties, _ = install_fake_bpy(br.LANDMARK_NAMES)
    saved = (br.LOST_HOLD_S, br.LOST_EASE_S)
    try:
        br.LOST_HOLD_S, br.LOST_EASE_S = 0.5, 1.0
        st = br.State()
        st.rig = rig
        live = _hand_points(rig, {f: (30.0, 90.0, 60.0) for f in FOUR})
        st.last_local = br.retarget(rig, live, True)
        st.last_matrix, st.last_offset = br._IDENTITY, (0.0, 0.0, 0.0)
        st.hand_present, st.lost_since = False, 100.0
        tip = lambda: empties["MIDDLE_FINGER_TIP"].location
        rest = [(0.0, 0.0, 0.0)] * 21
        for finger, chain in rig.chains.items():
            pos = (0.0, 0.0, 0.0)
            for i, d, length in zip(br.CHAIN_LANDMARKS[finger][1:], chain.rest_dirs, chain.lengths):
                pos = tuple(p + length * c for p, c in zip(pos, d))
                rest[i] = pos
        dist = lambda a, b: math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        i_tip = br.LANDMARK_NAMES.index("MIDDLE_FINGER_TIP")

        assert br.relax_lost_hand(st, 100.3) is False, "the last pose is held first"
        assert tip() == (0.0, 0.0, 0.0), "nothing was moved while holding"
        assert br.relax_lost_hand(st, 101.0) is True
        halfway = dist(tip(), rest[i_tip])
        assert 0.0 < halfway < dist(st.last_local[i_tip], rest[i_tip]), "moving toward rest"
        for finger, chain in rig.chains.items():  # every bone tip stays reachable: rig bone lengths
            idx = br.CHAIN_LANDMARKS[finger]
            for s in range(4):
                a = empties[br.LANDMARK_NAMES[idx[s]]].location if s else (0.0, 0.0, 0.0)
                b = empties[br.LANDMARK_NAMES[idx[s + 1]]].location
                assert abs(dist(a, b) - chain.lengths[s]) < 1e-9
        assert not st.relaxed
        assert br.relax_lost_hand(st, 105.0) is True and st.relaxed
        assert dist(tip(), rest[i_tip]) < 1e-9, "ends exactly at the rest pose"
        assert br.relax_lost_hand(st, 106.0) is False, "once relaxed it does nothing more"

        st.relaxed, st.hand_present = False, True  # a hand is present: never relax
        assert br.relax_lost_hand(st, 200.0) is False
        st.hand_present = False
        br.LOST_EASE_S = 0.0  # switched off: hold until the hand comes back
        assert br.relax_lost_hand(st, 200.0) is False
    finally:
        br.LOST_HOLD_S, br.LOST_EASE_S = saved
        br.bpy = None


def test_status_and_control_keys():
    names = list(br.LANDMARK_NAMES)
    empties, timers = install_fake_bpy(names)
    br.PORT = free_port()
    br.start()
    ns = br.bpy.app.driver_namespace
    assert ns[br.STATUS_KEY]["running"] is True and ns[br.STATUS_KEY]["text"] == "waiting for the sender"
    control = ns[br.CONTROL_KEY]
    control["set_mirror"](True)
    assert br.MIRROR_OVERRIDE is True
    control["set_mirror"](None)
    assert br.MIRROR_OVERRIDE is None
    control["stop"]()
    assert br.CONTROL_KEY not in ns and ns[br.STATUS_KEY] == {"running": False, "text": "stopped"}
    br.bpy = None


def test_protocol_slot():
    pts = make_hand()
    one = protocol.decode(protocol.encode_frame("s", 1, 1.0, "Right", pts))
    two = protocol.decode(protocol.encode_frame("s", 2, 1.0, "Left", pts, slot=1))
    assert "slot" not in one, "slot 0 is left out, so one-hand packets are unchanged"
    assert two["slot"] == 1
    assert "slot" not in protocol.decode(protocol.encode_lost("s", 3))
    assert protocol.decode(protocol.encode_lost("s", 4, slot=1))["slot"] == 1
    assert br.decode(protocol.encode_frame("s", 5, 1.0, "Left", pts, slot=1))["slot"] == 1


def test_receiver_and_sender_agree_on_which_hand_is_which():
    import slots
    assert br.HAND_NAMES[slots.RIGHT] == "Right hand" and br.HAND_NAMES[slots.LEFT] == "Left hand"
    assert slots.SLOT_NAMES[slots.RIGHT] == "right" and slots.SLOT_NAMES[slots.LEFT] == "left"
    assert slots.RIGHT == 1 and slots.LEFT == 0, "the right hand drives Armature.001, on the +X side of the left one"


def test_slot_assigner():
    from slots import LEFT, RIGHT, SlotAssigner
    # the label names the rig: the right hand drives slot 0 and the left hand slot 1, wherever they are
    a = SlotAssigner(slots=2)
    assert a.assign([]) == {}
    assert a.assign([(0.7, 0.5), (0.3, 0.5)], ["Right", "Left"]) == {RIGHT: 0, LEFT: 1}
    crossed = SlotAssigner(slots=2)  # arms crossed: the right hand is on the left of the picture
    assert crossed.assign([(0.3, 0.5), (0.7, 0.5)], ["Right", "Left"]) == {RIGHT: 0, LEFT: 1}
    swapped_list = SlotAssigner(slots=2)
    assert swapped_list.assign([(0.3, 0.5), (0.7, 0.5)], ["Left", "Right"]) == {LEFT: 0, RIGHT: 1}
    # one hand alone: its label, not its side
    for label, slot in (("Right", RIGHT), ("Left", LEFT)):
        for x in (0.2, 0.8):
            assert SlotAssigner(slots=2).assign([(x, 0.5)], [label]) == {slot: 0}, (label, x)
    # no usable label: the side of the picture decides, the right hand being on the right
    side = SlotAssigner(slots=2)
    assert side.assign([(0.8, 0.5)]) == {RIGHT: 0}
    assert SlotAssigner(slots=2).assign([(0.2, 0.5)]) == {LEFT: 0}
    assert SlotAssigner(slots=2).assign([(0.3, 0.5), (0.7, 0.5)], ["Unknown", None]) == {RIGHT: 1, LEFT: 0}
    # two hands both called "Right": the second one still gets the other slot (by its side)
    both = SlotAssigner(slots=2)
    assert both.assign([(0.3, 0.5), (0.7, 0.5)], ["Right", "Right"]) == {RIGHT: 1, LEFT: 0}

    # a flickering label cannot make a hand change rig; a label that keeps saying so can
    f = SlotAssigner(slots=2)
    f.assign([(0.7, 0.5), (0.3, 0.5)], ["Right", "Left"])
    for step in range(1, 5):  # the right hand is called "Left" for four frames
        got = f.assign([(0.7 + 0.01 * step, 0.5), (0.3, 0.5)], ["Left", "Left"])
        assert got == {RIGHT: 0, LEFT: 1}, (step, got)
    single = SlotAssigner(slots=2)
    single.assign([(0.7, 0.5)], ["Right"])
    for step in range(12):  # called "Left" for good: it was the left hand all along
        got = single.assign([(0.7, 0.5)], ["Left"])
    assert got == {LEFT: 0}, got
    # two hands that both disagree with their slots swap
    sw = SlotAssigner(slots=2)
    sw.assign([(0.7, 0.5), (0.3, 0.5)], ["Right", "Left"])
    for step in range(14):
        got = sw.assign([(0.7, 0.5), (0.3, 0.5)], ["Left", "Right"])
    assert got == {LEFT: 0, RIGHT: 1}, got

    # a hand that moves keeps its slot, and so do two hands passing each other (labels missing)
    b = SlotAssigner(slots=2)
    b.assign([(0.7, 0.5), (0.3, 0.5)], ["Right", "Left"])
    for step in range(1, 12):
        x = 0.3 + 0.03 * step
        got = b.assign([(0.7, 0.5), (x, 0.5)] if step % 2 else [(x, 0.5), (0.7, 0.5)])
        assert set(got) == {0, 1}
        assert got[LEFT] == (1 if step % 2 else 0), step  # the moving hand is the left one
    c = SlotAssigner(slots=2)
    c.assign([(0.2, 0.3), (0.8, 0.7)])
    for step in range(1, 21):
        t = step / 20.0
        got = c.assign([(0.2 + 0.6 * t, 0.3), (0.8 - 0.6 * t, 0.7)])
        assert got == {LEFT: 0, RIGHT: 1}, (step, got)
    # a hand that drops out for a moment returns to its slot; after a long absence the label decides again
    d = SlotAssigner(slots=2)
    assert d.assign([(0.8, 0.5)], ["Right"]) == {RIGHT: 0}
    for _ in range(5):
        d.assign([])
    assert d.assign([(0.78, 0.5)], ["Right"]) == {RIGHT: 0}
    for _ in range(40):
        d.assign([])
    assert d.assign([(0.2, 0.5)], ["Left"]) == {LEFT: 0}
    # a second hand joins the free slot
    e = SlotAssigner(slots=2)
    e.assign([(0.8, 0.5)], ["Right"])
    assert e.assign([(0.8, 0.5), (0.25, 0.5)], ["Right", "Left"]) == {RIGHT: 0, LEFT: 1}
    # one slot: always slot 0
    one = SlotAssigner(slots=1)
    assert one.assign([(0.9, 0.5)], ["Left"]) == {0: 0} and one.assign([(0.1, 0.5)], ["Right"]) == {0: 0}


def test_two_hands_are_routed_by_slot():
    import time
    names = list(br.LANDMARK_NAMES) + [n + ".001" for n in br.LANDMARK_NAMES]
    empties, _ = install_fake_bpy(names)
    s0, s1 = br.State(), br.State()
    s1.slot, s1.suffix, s1.tag = 1, ".001", "[left hand]"
    s1.origin = (0.8, 0.0, 0.0)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.setblocking(False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = rx.getsockname()
    h0, h1 = make_hand(), make_hand()
    h0[8] = (0.01, -0.05, 0.02)
    h1[8] = (-0.02, -0.06, 0.03)
    states = {0: s0, 1: s1}
    for seq in range(1, 40):
        tx.sendto(protocol.encode_frame("two", seq, time.time(), "Right", h0), addr)           # no slot: slot 0
        tx.sendto(protocol.encode_frame("two", seq, time.time(), "Left", h1, slot=1), addr)
    tx.sendto(protocol.encode_frame("two", 50, time.time(), "Left", h1, slot=2), addr)         # no rig for slot 2
    time.sleep(0.05)
    assert br.process(rx, states) is True
    sc = br.REFERENCE_DISTANCE / 0.09
    t0 = empties["INDEX_FINGER_TIP"].location
    t1 = empties["INDEX_FINGER_TIP.001"].location
    assert all(abs(a - b) < 1e-4 for a, b in zip(t0, (0.01 * sc, 0.02 * sc, 0.05 * sc))), t0
    assert all(abs(a - b) < 1e-4 for a, b in zip(t1, (-0.02 * sc + 0.8, 0.03 * sc, 0.06 * sc))), t1
    assert empties["WRIST"].location == (0.0, 0.0, 0.0) and empties["WRIST.001"].location == (0.8, 0.0, 0.0)
    assert s0.stats["dropped"] >= 1, "the packet for slot 2 is dropped"
    assert s0.hand_present and s1.hand_present
    # only hand 2 is lost: hand 1 keeps its pose and state
    before = empties["INDEX_FINGER_TIP"].location
    tx.sendto(protocol.encode_lost("two", 60, slot=1), addr)
    time.sleep(0.02)
    br.process(rx, states)
    assert s1.hand_present is False and s0.hand_present is True
    assert empties["INDEX_FINGER_TIP"].location == before
    # a restart of the sender resets each hand separately and answers once per hand
    tx.sendto(protocol.encode_frame("again", 1, time.time(), "Right", h0), addr)
    tx.sendto(protocol.encode_frame("again", 1, time.time(), "Left", h1, slot=1), addr)
    time.sleep(0.02)
    br.process(rx, states)
    assert s0.sid == "again" and s1.sid == "again" and s1.hand_present is True
    rx.close()
    tx.close()
    br.bpy = None


def _run_sender_loop(argv, ack=None, ack_after=3, frames=14):
    """Run sender.main() for `frames` camera frames with a fake camera and a fake MediaPipe that sees
    two hands (it reports at most num_hands of them, like the real one). ack: what the fake
    receiver answers once, after ack_after frames. Returns (packets, stdout, lit_pixels, builds)."""
    import contextlib
    import io
    import numpy as np
    import sender
    P = lambda x, y, z=0.0: types.SimpleNamespace(x=x, y=y, z=z)
    world_a = [P(*p) for p in make_hand(span=0.09)]
    world_b = [P(p.x + 0.002, p.y, p.z) for p in world_a]

    def norm(cx, cy):
        return [P(cx + 0.012 * (i % 5), cy + 0.012 * (i // 5)) for i in range(21)]

    seen = {"calls": 0, "builds": [], "pixels": [], "acked": False}

    class FakeLandmarker:
        def __init__(self, num_hands):
            self.num_hands = num_hands

        def detect_for_video(self, image, ts):
            seen["calls"] += 1
            both = [(norm(0.65, 0.5), world_b, "Right"), (norm(0.15, 0.5), world_a, "Left")][: self.num_hands]
            return types.SimpleNamespace(
                hand_landmarks=[n for n, _, _ in both], hand_world_landmarks=[w for _, w, _ in both],
                handedness=[[types.SimpleNamespace(category_name=label, score=0.9)] for _, _, label in both])

        def close(self):
            pass

    class FakeCap:
        isOpened = lambda self: True
        read = lambda self: (True, np.zeros((480, 640, 3), np.uint8))
        release = lambda self: None

    sent = []

    class FakeSock:
        def sendto(self, data, addr):
            sent.append(protocol.decode(data))

        def recvfrom(self, n):
            if ack is not None and not seen["acked"] and seen["calls"] >= ack_after:
                seen["acked"] = True
                return ack, ("127.0.0.1", 1)
            raise BlockingIOError

        def close(self):
            pass

    def build(model_path=None, num_hands=1, detect_confidence=None):
        seen["builds"].append((num_hands, detect_confidence))
        return FakeLandmarker(num_hands)

    def imshow(name, frame):
        seen["pixels"].append(int((frame.sum(axis=2) > 0).sum()))

    patches = {"open_camera": lambda *a, **k: FakeCap(), "build_landmarker": build, "make_socket": lambda: FakeSock()}
    old = {k: getattr(sender, k) for k in patches}
    old_cv = (sender.cv2.imshow, sender.cv2.waitKey, sender.cv2.destroyAllWindows)
    old_argv = sys.argv
    try:
        for k, v in patches.items():
            setattr(sender, k, v)
        sender.cv2.imshow = imshow
        sender.cv2.waitKey = lambda ms: 27 if len(seen["pixels"]) >= frames else -1
        sender.cv2.destroyAllWindows = lambda: None
        sys.argv = ["sender.py", "--port", "5999", *argv]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sender.main()
    finally:
        for k, v in old.items():
            setattr(sender, k, v)
        sender.cv2.imshow, sender.cv2.waitKey, sender.cv2.destroyAllWindows = old_cv
        sys.argv = old_argv
    return sent, out.getvalue(), seen["pixels"], seen["builds"]


def _per_slot(packets):
    frames = [m for m in packets if m and "lm" in m]
    return {s: sum(1 for m in frames if m.get("slot", 0) == s) for s in (0, 1)}


def test_sender_loop_follows_the_receivers_hand_count():
    # forced two hands: both hands are sent and both are drawn (more dots than with one hand)
    sent2, out2, pix2, builds2 = _run_sender_loop(["--hands", "2"])
    sent1, out1, pix1, builds1 = _run_sender_loop(["--hands", "1"])
    assert builds2 == [(2, None)] and builds1 == [(1, None)]
    assert _per_slot(sent2) == {0: 14, 1: 14}, _per_slot(sent2)
    labels = {(m.get("slot", 0), m["hand"]) for m in sent2 if m and "lm" in m}
    from slots import LEFT, RIGHT
    assert labels == {(RIGHT, "Right"), (LEFT, "Left")}, labels  # each hand drives its own rig
    assert _per_slot(sent1) == {0: 14, 1: 0}
    assert pix2[-1] > 1.5 * pix1[-1], (pix2[-1], pix1[-1], "the second hand's points are drawn")
    # auto: Blender answers "2 rigs" after 3 frames; the sender rebuilds MediaPipe for two hands
    sent, out, pix, builds = _run_sender_loop([], ack=b"ack hands=2")
    assert builds == [(1, None), (2, None)], builds
    assert "now tracking 2 hands" in out and _per_slot(sent)[1] > 0 and _per_slot(sent)[0] == 14
    first_two_hand_frame = min(i for i, m in enumerate(sent) if m and m.get("slot") == 1 and "lm" in m)
    assert pix[-1] > 1.5 * pix[0], "dots for both hands once it switched"
    # auto with a one-rig answer or no answer: stays on one hand, never sends a second hand
    for ack in (b"ack", None):
        sent, out, pix, builds = _run_sender_loop([], ack=ack)
        assert builds == [(1, None)] and _per_slot(sent) == {0: 14, 1: 0}, (ack, builds, _per_slot(sent))
        assert "now tracking" not in out
    # the receiver's reply decides, and a mismatch is explained instead of silently ignored
    _, out, _, _ = _run_sender_loop(["--hands", "1"], ack=b"ack hands=2")
    assert "--hands 1 tracks one" in out
    _, out, _, _ = _run_sender_loop(["--hands", "2"], ack=b"ack")
    assert "only one hand rig" in out
    # the detection confidence can be set, and two hands default lower than one
    _, _, _, builds = _run_sender_loop(["--hands", "2", "--detect-confidence", "0.4"])
    assert builds == [(2, 0.4)]


def test_ack_carries_the_hand_count():
    import sender
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.setblocking(False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.bind(("127.0.0.1", 0))
    tx.setblocking(False)
    for states, expected in (({0: br.State()}, 1),
                             ({0: br.State(), 1: br.State()}, 2)):
        for st in states.values():
            st.sid = None
        tx.sendto(protocol.encode_frame("ackme", 1, 1.0, "Right", make_hand()), rx.getsockname())
        time.sleep(0.02)
        br.drain_all(rx, states)
        time.sleep(0.02)
        assert sender.poll_ack(tx) == expected, expected
    assert sender.poll_ack(tx) is False
    tx.sendto(b"ack hands=oops", tx.getsockname())  # junk after the prefix still counts as an answer
    time.sleep(0.01)
    assert sender.poll_ack(tx) == 1
    rx.close()
    tx.close()


def _random_rotation(rng):
    axis = br._unit((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
    angle = rng.uniform(0, 170)  # one angle for all three axes, or it is not a rotation
    cols = [_rot_about(axis, angle, v) for v in ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))]
    return tuple(tuple(cols[j][i] for j in range(3)) for i in range(3))


def test_shape_from_finger_bends():
    rig = br.Rig(_fake_rig_arm())
    assert rig.bend_sign in (1.0, -1.0)
    bent = _hand_points(rig, {f: (30.0, 80.0, 50.0) for f in FOUR})
    flat = _hand_points(rig, {f: (0.0, 0.0, 0.0) for f in FOUR})
    mirror = lambda pts: [(-x, y, z) for x, y, z in pts]  # the other hand
    rng = random.Random(4)

    def decide(points, label=None, rotate=None):
        st = br.State()
        st.rig = rig
        pts = [br._matvec(rotate, p) for p in points] if rotate else points
        br.update_mirror(st, {"hand": label}, pts)
        return st

    for _ in range(5):  # the answer cannot depend on how the hand is turned
        rot = _random_rotation(rng)
        same, other = decide(bent, rotate=rot), decide(mirror(bent), rotate=rot)
        assert same.shape_confirmed and same.mirror is False and same.shape_score > 0.3
        assert other.shape_confirmed and other.mirror is True and other.shape_score < -0.3

    f = decide(flat)  # a flat hand says nothing yet
    assert not f.shape_confirmed and f.mirror is False
    wrong_guess = decide(flat, label="Right")  # the label's first guess may be wrong ...
    assert wrong_guess.mirror is True and not wrong_guess.shape_confirmed
    br.update_mirror(wrong_guess, {"hand": "Right"}, bent)  # ... the first bent frame corrects it
    assert wrong_guess.mirror is False and wrong_guess.shape_confirmed
    assert abs(br.bend_evidence(flat, br.palm_frame(flat[0], flat[5], flat[9], flat[17])[0])) < br.EVIDENCE_MIN

    st = decide(bent)  # hysteresis: one odd frame does not flip a confirmed shape, a lasting change does
    for _ in range(6):
        br.update_mirror(st, {}, bent)
    br.update_mirror(st, {}, mirror(bent))
    assert st.mirror is False, "one opposite frame must not flip it"
    for _ in range(5):
        br.update_mirror(st, {}, mirror(bent))
    assert st.mirror is True, "lasting opposite evidence flips it"


def test_saved_while_mirrored_file_still_works():
    """A .blend saved while the receiver had the rig mirrored holds a mirrored transform. The curl
    direction must come out the same as from a clean file (this made fingers refuse to curl)."""
    rot = br._matmul(_rot("z", -199), _rot("x", 25))
    refl = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    saved_mirrored = br._matmul(rot, refl)  # determinant -1
    results = {}
    for name, m0 in (("clean", rot), ("saved mirrored", saved_mirrored)):
        arm = _fake_rig_arm(m0)
        br.bpy = types.SimpleNamespace(data=types.SimpleNamespace(objects=types.SimpleNamespace(
            get=lambda n, arm=arm: arm if n == br.ARMATURE_NAME else None)))
        rig = br.Rig(arm)
        assert br._dot(rig.m0[0], br._cross(rig.m0[1], rig.m0[2])) > 0, "reference must be a rotation"
        bent = _hand_points(rig, {f: (30.0, 80.0, 50.0) for f in FOUR})
        for shape, pts in (("same", bent), ("other", [(-x, y, z) for x, y, z in bent])):
            st = br.State()
            st.rig = rig
            br.update_mirror(st, {}, pts)
            matrix = br.drive_armature(st, pts, (0.0, 0.0, 0.0))
            det = br._dot(matrix[0], br._cross(matrix[1], matrix[2]))
            results[(name, shape)] = (st.mirror, det > 0)
        br.restore_armature(st)
        assert arm.matrix_world == rig.matrix0, "put back exactly as found"
    br.bpy = None
    assert results[("clean", "same")] == (False, True) and results[("clean", "other")] == (True, False), results
    assert results[("saved mirrored", "same")] == results[("clean", "same")], results
    assert results[("saved mirrored", "other")] == results[("clean", "other")], results


def test_receiver_constants_match_rig_tools():
    import rig_tools
    assert br.FLEXION_RANGE == rig_tools.FLEXION_RANGE
    assert br.MCP_ABDUCTION_DEG == rig_tools.MCP_ABDUCTION_DEG
    assert br.SIDE_MARGIN_DEG == rig_tools.SIDE_MARGIN_DEG
    assert {k: v for k, v in br.BONE_CHAINS.items() if k != "thumb"} == rig_tools.FINGERS
    assert set(b for chain in br.BONE_CHAINS.values() for b in chain) == set(rig_tools.BONE_TARGETS)
    for finger, names in br.BONE_CHAINS.items():
        for s, bone in enumerate(names):  # each bone's landmark in the receiver matches rig_tools
            assert rig_tools.BONE_TARGETS[bone] == protocol.LANDMARK_NAMES[br.CHAIN_LANDMARKS[finger][s + 1]]


def test_record_and_replay():
    import tempfile, os
    import recording, replay
    path = os.path.join(tempfile.mkdtemp(), "rec.jsonl")
    rec = recording.Recorder(path)
    for i in range(5):
        rec.write(protocol.encode_frame("orig", i + 1, 1000.0 + i, "Right", make_hand(), (0.0, 0.0, 0.5), 9.0))
    rec.write(protocol.encode_lost("orig", 6))
    rec.close()
    with open(path, "a") as fh:
        fh.write("not json\n\n")
    frames = recording.load(path)
    assert len(frames) == 6, "corrupt lines are skipped"

    rx, tx, addr = make_rx()
    sent = replay.replay(frames, addr, speed=1000.0, sleep=lambda s: None, sock=tx)
    import time; time.sleep(0.05)
    st = br.State()
    newest, stop = br.drain(rx, st)
    assert sent == 6 and stop is True, "replay ends with a stop packet"
    assert st.sid not in (None, "orig") and newest.get("lost"), "fresh session id, ends on the lost packet"
    rx.close(); tx.close()


def test_sender_pieces():
    import numpy as np
    import sender
    # Windows: sending to a closed port must not crash the ack poll
    s = sender.make_socket()
    s.sendto(b"x", ("127.0.0.1", free_port()))
    import time; time.sleep(0.1)
    assert sender.poll_ack(s) is False
    s.close()

    # real landmarker on a blank frame: runs, finds no hand
    import mediapipe as mp
    lm = sender.build_landmarker()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=blank), 1)
    assert sender.extract_world_points(res) is None
    lm.close()


TESTS = [test_one_euro, test_protocol, test_drain, test_mapping_and_scale,
         test_axis_mapping_is_a_rotation, test_translation_logic, test_outlier_gate,
         test_palm_math, test_mirror_votes, test_drive_armature_with_fake_armature,
         test_retarget, test_straight_dead_zone, test_hand_yaw, test_unmirror, test_turn_about_the_hands_own_axis,
         test_movement_directions_match_the_user, test_fit_triangulation, test_finger_collision, test_collision_radii_and_cost, test_rigid_palm, test_thumb_limits, test_relax_when_hand_is_lost,
         test_status_and_control_keys, test_protocol_slot, test_receiver_and_sender_agree_on_which_hand_is_which, test_slot_assigner,
         test_two_hands_are_routed_by_slot, test_sender_loop_follows_the_receivers_hand_count,
         test_ack_carries_the_hand_count, test_shape_from_finger_bends, test_saved_while_mirrored_file_still_works,
         test_receiver_constants_match_rig_tools,
         test_estimate_root, test_record_and_replay, test_end_to_end_with_fake_bpy,
         test_sender_pieces]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL {t.__name__}: {exc!r}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
