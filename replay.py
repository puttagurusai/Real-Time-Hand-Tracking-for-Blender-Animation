"""Replay a recording made with `sender.py --record FILE`.

    python v5/replay.py FILE [--loop] [--speed 1.0] [--host H] [--port P]

Useful for testing the Blender side without a webcam and for comparing changes
on identical input.
"""
import argparse
import socket
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import protocol
import recording


def replay(frames, address, speed=1.0, loop=False, sleep=time.sleep, sock=None):
    """Send every recorded packet with its original timing. Returns packets sent."""
    own = sock is None
    if own:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0
    try:
        while True:
            sid, seq = uuid.uuid4().hex[:8], 0
            start = time.monotonic()
            for dt, pkt in frames:
                wait = dt / speed - (time.monotonic() - start)
                if wait > 0:
                    sleep(wait)
                seq += 1
                pkt = dict(pkt, sid=sid, seq=seq)
                if "lm" in pkt:
                    pkt["t"] = round(time.time(), 4)
                sock.sendto(protocol._dumps(pkt), address)
                sent += 1
            if not loop:
                break
        seq += 1
        sock.sendto(protocol.encode_stop(sid, seq), address)
    finally:
        if own:
            sock.close()
    return sent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--host", default=protocol.DEFAULT_ADDRESS[0])
    ap.add_argument("--port", type=int, default=protocol.DEFAULT_ADDRESS[1])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true", help="repeat until Ctrl+C")
    args = ap.parse_args()
    frames = recording.load(args.file)
    if not frames:
        sys.exit(f"No frames in {args.file}")
    print(f"Replaying {len(frames)} packets ({frames[-1][0]:.1f}s) to {args.host}:{args.port}")
    try:
        n = replay(frames, (args.host, args.port), args.speed, args.loop)
        print(f"Sent {n} packets.")
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
