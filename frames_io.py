"""Load recorded hand landmarks for the offline tools (analyze_recording_headless.py,
bake_recording.py). Pure Python, no Blender needed.

Two kinds of file:
  .jsonl          made by `sender.py --record`: the packets exactly as the sender transmitted them,
                  so they are already outlier-gated and filtered
  .json           the old landmarks.json: raw per-frame normalised landmarks (a flipped 640x480
                  image, z on x's scale) at an assumed 30 fps; the sender's gate and filter are
                  not applied yet (emulate_sender does that)
"""
import json
import math

import recording

LEGACY_FPS = 30.0
LEGACY_SIZE = (640.0, 480.0)
LEGACY_PALM_SPAN_M = 0.09   # the legacy file has no metric scale: the median palm span is set to 9 cm


def is_recording(path):
    return path.endswith(".jsonl")


def slots_in(path):
    """The hand slots a .jsonl recording has frames for (a legacy file is slot 0 only)."""
    if not is_recording(path):
        return [0]
    return sorted({pkt.get("slot", 0) for _, pkt in recording.load(path) if "lm" in pkt})


def load_frames(path, landmark_names, slot=0):
    """Return [(dt_seconds, label_or_None, score_or_None, [[x, y, z] * 21])], hand frames only,
    of one hand slot (0 or 1; a recording made with `sender.py --hands 2` has both)."""
    if is_recording(path):
        return [(dt, pkt.get("hand"), pkt.get("score"), pkt["lm"])
                for dt, pkt in recording.load(path) if "lm" in pkt and pkt.get("slot", 0) == slot]
    w, h = LEGACY_SIZE
    raw = [[(fr[n]["x"] * w, fr[n]["y"] * h, fr[n]["z"] * w) for n in landmark_names]
           for fr in json.load(open(path)) if fr]
    spans = sorted(math.dist(f[0], f[9]) for f in raw)
    k = LEGACY_PALM_SPAN_M / spans[len(spans) // 2]
    frames = []
    for i, f in enumerate(raw):
        centre = tuple(sum(p[c] for p in f) / len(f) for c in range(3))
        frames.append((i / LEGACY_FPS, None, None, [[(p[c] - centre[c]) * k for c in range(3)] for p in f]))
    return frames


def emulate_sender(frames):
    """Run frames through the sender's outlier gate and One Euro filter, the way a live sender
    would. Returns (frames, number_rejected); a rejected frame repeats the previous pose."""
    from one_euro import LandmarkFilter
    from outlier import OutlierGate
    gate, smoother = OutlierGate(), LandmarkFilter()
    out, rejected = [], 0
    for dt, label, score, lm in frames:
        ok, _ = gate.check([tuple(p) for p in lm], dt)
        if not ok:
            rejected += 1
            out.append(out[-1] if out else (dt, label, score, lm))
            continue
        out.append((dt, label, score, [list(p) for p in smoother([tuple(p) for p in lm], dt)]))
    return out, rejected
