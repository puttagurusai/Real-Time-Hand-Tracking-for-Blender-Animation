"""Record what the sender transmits and replay it later (no webcam needed).

File format: JSON Lines, one object per sent packet:
    {"dt": seconds_since_first_packet, "pkt": {...packet dict as sent...}}
Replay re-sends the packets with a fresh session id, fresh sequence numbers and
current timestamps, so the receiver treats it like a live sender.
"""
import json
import time


class Recorder:
    def __init__(self, path):
        self._fh = open(path, "w", encoding="utf-8")
        self._t0 = None

    def write(self, packet_bytes):
        """Store a packet exactly as it was sent (the bytes passed to sendto)."""
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        pkt = json.loads(packet_bytes.decode("utf-8"))
        self._fh.write(json.dumps({"dt": round(now - self._t0, 4), "pkt": pkt},
                                  separators=(",", ":")) + "\n")

    def close(self):
        self._fh.close()


def load(path):
    """Return a list of (dt, packet_dict). Skips blank or corrupt lines."""
    frames = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                frames.append((float(row["dt"]), dict(row["pkt"])))
            except (ValueError, KeyError, TypeError):
                continue
    return frames
