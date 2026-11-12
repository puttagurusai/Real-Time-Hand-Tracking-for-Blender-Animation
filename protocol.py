"""Wire format shared by the sender and the Blender receiver (UDP, JSON).

Packet kinds (all carry "v", "sid", "seq"):
  frame : {"v":1,"sid":..,"seq":..,"t":..,"hand":"Right","lm":[[x,y,z] x21],
           "root":[x,y,z],"ms":12.3}
  lost  : {"v":1,"sid":..,"seq":..,"lost":true}   (also carries "slot" when it is not 0)
  stop  : {"v":1,"sid":..,"seq":..,"stop":true}

"lm"   MediaPipe *world* landmarks (metres, hand-centred, y down) in MediaPipe
       index order, so no per-packet names are needed.
"slot" which of the sender's hands this packet is about: 0 or 1. Left out when it is 0, so a
       one-hand sender's packets are unchanged. The receiver follows each slot with its own rig.
"hand" MediaPipe's handedness label, "score" its confidence (0..1). MediaPipe assumes a
       mirrored (selfie) image, so the label describes the hand as seen in that image;
       the receiver uses it to decide whether the rig must be mirrored to match.
"root" optional: estimated WRIST position in camera space, metres (x right,
       y down, z away). Only used when the receiver enables hand translation.
"t"    sender wall-clock time (time.time()). Comparable with the receiver's clock
       on the same machine; used for latency stats and stale-frame rejection.
"ms"   sender-side processing time for this frame (capture to send), for profiling.
"sid"  random per sender run; "seq" increases within a session. The receiver uses
       both to drop stale or duplicated packets and to detect a sender restart.

blender_receiver.py is pasted into a Blender text block and cannot import this
module, so it carries its own copy of LANDMARK_NAMES; test_pipeline.py checks
that the two copies agree.
"""
import json

PROTOCOL_VERSION = 1
DEFAULT_ADDRESS = ("127.0.0.1", 5004)
MAX_PACKET_BYTES = 4096

LANDMARK_NAMES = (
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
)


def _dumps(obj):
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def encode_frame(sid, seq, t, hand, points, root=None, sender_ms=None, score=None, slot=0):
    """points: 21 (x, y, z) tuples in metres, MediaPipe index order."""
    if len(points) != len(LANDMARK_NAMES):
        raise ValueError(f"expected {len(LANDMARK_NAMES)} landmarks, got {len(points)}")
    msg = {"v": PROTOCOL_VERSION, "sid": sid, "seq": seq, "t": round(t, 4), "hand": hand,
           "lm": [[round(x, 5), round(y, 5), round(z, 5)] for x, y, z in points]}
    if slot:
        msg["slot"] = int(slot)
    if score is not None:
        msg["score"] = round(score, 3)
    if root is not None:
        msg["root"] = [round(v, 5) for v in root]
    if sender_ms is not None:
        msg["ms"] = round(sender_ms, 2)
    data = _dumps(msg)
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError(f"packet too large: {len(data)} bytes")
    return data


def encode_lost(sid, seq, slot=0):
    msg = {"v": PROTOCOL_VERSION, "sid": sid, "seq": seq, "lost": True}
    if slot:
        msg["slot"] = int(slot)
    return _dumps(msg)


def encode_stop(sid, seq):
    return _dumps({"v": PROTOCOL_VERSION, "sid": sid, "seq": seq, "stop": True})


def decode(data):
    """Return the packet dict, or None if it is malformed or from another version."""
    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(msg, dict) or msg.get("v") != PROTOCOL_VERSION:
        return None
    if "sid" not in msg or "seq" not in msg:
        return None
    return msg
