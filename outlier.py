"""Reject single-frame tracking glitches before they reach the filter and Blender.

A frame is rejected when either
  * some landmark moved faster than max_speed since the last accepted frame, or
  * the palm span (WRIST -> MIDDLE_FINGER_MCP, rigid in reality) deviates from its
    recent median by more than span_tolerance.
After max_consecutive rejections in a row the new frame is accepted anyway, so a
genuine fast change or a re-acquired hand can never lock the gate shut.

MediaPipe's hand model gives no per-landmark confidence, so this plausibility
check is the practical substitute.
"""
import math
from collections import deque

WRIST = 0
MIDDLE_MCP = 9


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class OutlierGate:
    def __init__(self, max_speed=4.0, span_tolerance=0.30, max_consecutive=5, slack=0.02):
        self.max_speed = max_speed            # metres / second, per landmark
        self.span_tolerance = span_tolerance  # fraction of the median palm span
        self.max_consecutive = max_consecutive
        self.slack = slack                    # metres of jitter always allowed
        self._spans = deque(maxlen=30)
        self.reset()

    def reset(self):
        self._last = None
        self._prev_t = None
        self._rejected = 0
        self._spans.clear()

    def check(self, points, t):
        """Return (accepted, reason). reason is '' when accepted."""
        span = _dist(points[WRIST], points[MIDDLE_MCP])
        reason = ""
        # Speed is measured over one frame interval, not since the last accepted frame;
        # otherwise every rejected frame would loosen the limit for the next one.
        dt = min(max(t - self._prev_t, 1e-3), 0.25) if self._prev_t is not None else None
        self._prev_t = t
        if self._last is not None:
            if len(self._spans) >= 5:
                ordered = sorted(self._spans)
                median = ordered[len(ordered) // 2]
                if median > 1e-4 and abs(span - median) > self.span_tolerance * median:
                    reason = f"palm span {span * 100:.1f} cm vs {median * 100:.1f} cm"
            if not reason and dt is not None:
                limit = self.max_speed * dt + self.slack
                worst = max(_dist(p, q) for p, q in zip(points, self._last))
                if worst > limit:
                    reason = f"landmark jumped {worst * 100:.1f} cm in {dt * 1000:.0f} ms"

        if reason and self._rejected < self.max_consecutive:
            self._rejected += 1
            return False, reason

        self._rejected = 0
        self._last = list(points)
        self._spans.append(span)
        return True, ""
