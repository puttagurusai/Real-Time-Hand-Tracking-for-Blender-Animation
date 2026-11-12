"""One Euro filter (Casiez, Roussel, Vogel 2012) for noisy tracking signals.

The cutoff rises with signal speed: strong smoothing while the hand is still,
low lag while it moves fast. Inputs are in metres, so beta is much larger than
the values quoted for normalised pixel coordinates.
"""
import math


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    # The defaults come from a sweep on real recorded curling (30 fps). The earlier 1.5 / 20 / 1.0
    # smoothed a still hand more (it kept 41% of 2 mm noise, these keep 65%) but lagged finger
    # bends by 5 degrees on average and 39 at the worst moment; these lag 1.9 and 16.
    def __init__(self, min_cutoff=2.0, beta=80.0, d_cutoff=5.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._t = None
        self._x = 0.0
        self._dx = 0.0

    def __call__(self, x, t):
        if self._t is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = max(t - self._t, 1e-3)
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx += a_d * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x += a * (x - self._x)
        self._t = t
        return self._x


class LandmarkFilter:
    """One OneEuroFilter per axis of each landmark."""

    def __init__(self, count=21, **kwargs):
        self._filters = [[OneEuroFilter(**kwargs) for _ in range(3)] for _ in range(count)]

    def reset(self):
        for axes in self._filters:
            for f in axes:
                f.reset()

    def __call__(self, points, t):
        return [tuple(f(v, t) for f, v in zip(axes, p))
                for axes, p in zip(self._filters, points)]
