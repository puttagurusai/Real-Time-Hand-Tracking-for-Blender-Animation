"""Webcam -> MediaPipe world landmarks -> outlier gate -> One Euro filter -> UDP to Blender.

Run:  python v5/sender.py            (ESC in the preview window quits)
      python v5/sender.py --hands 2  track both hands: your right hand drives the rig on the right of the
                                     front view (Armature.001), your left hand the one at the origin (Armature)
      python v5/sender.py --help
"""
import argparse
import math
import socket
import sys
import time
import uuid
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).parent))
import protocol
from one_euro import LandmarkFilter
from outlier import OutlierGate
from recording import Recorder
from slots import SLOT_NAMES, SlotAssigner

MODEL_PATH = Path(__file__).resolve().parent.parent / "hand_landmarker.task"
ACK = b"ack"


def build_landmarker(model_path=MODEL_PATH, num_hands=1, detect_confidence=None):
    """detect_confidence: how sure MediaPipe must be that it sees a hand. 0.7 for one hand; a second
    hand that is smaller, turned or partly hidden scores lower, so two hands use 0.5."""
    if detect_confidence is None:
        detect_confidence = 0.7 if num_hands == 1 else 0.5
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=detect_confidence,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_world_points(result):
    """Return (handedness, score, [(x, y, z)] * 21) in metres, or None if no hand."""
    if not result.hand_world_landmarks:
        return None
    world = result.hand_world_landmarks[0]
    hand, score = "Unknown", None
    if result.handedness and result.handedness[0]:
        hand = result.handedness[0][0].category_name
        score = result.handedness[0][0].score
    return hand, score, [(p.x, p.y, p.z) for p in world]


def extract_hands(result):
    """Every detected hand: [{"hand": label, "score": .., "points": [(x, y, z)] * 21 (metres),
    "norm": the 21 normalised image landmarks}]. MediaPipe lists them in no fixed order."""
    hands = []
    for i, world in enumerate(result.hand_world_landmarks or []):
        label, score = "Unknown", None
        if result.handedness and i < len(result.handedness) and result.handedness[i]:
            label = result.handedness[i][0].category_name
            score = result.handedness[i][0].score
        hands.append({"hand": label, "score": score, "points": [(p.x, p.y, p.z) for p in world],
                      "norm": result.hand_landmarks[i]})
    return hands


class Channel:
    """The filtering state of one hand slot: outlier gate, landmark filter and root filter."""

    def __init__(self, args):
        self.filt = LandmarkFilter(min_cutoff=args.min_cutoff, beta=args.beta, d_cutoff=args.d_cutoff)
        self.root_filt = LandmarkFilter(count=1, min_cutoff=0.5, beta=2.0)
        self.gate = None if args.no_outlier_gate else OutlierGate(max_speed=args.max_speed)
        self.visible = False

    def lose(self):
        """The hand is gone: do not smooth from its stale pose when it returns."""
        self.filt.reset()
        self.root_filt.reset()
        if self.gate:
            self.gate.reset()
        self.visible = False


def estimate_root(norm_landmarks, world_points, width, height, hfov_deg):
    """Estimate the WRIST position in camera space, in metres (x right, y down, z away).

    The world landmarks give the hand's shape and orientation in metres but not where it is.
    Where the hand is follows from where it appears in the image: with the hand centre at
    T = (Tx, Ty, Tz), every landmark i must satisfy (pinhole, u', v' = normalised image
    coordinates)  Tx + wx_i = u'_i (Tz + wz_i)  and  Ty + wy_i = v'_i (Tz + wz_i).
    That is linear in T, so all 21 landmarks give a small least-squares problem.

    Returns None when the hand has almost no spread in the image (pointing at the camera),
    where depth cannot be recovered. Accuracy is limited by the camera field of view
    (--hfov) and by how well MediaPipe estimates the world landmarks' depth.
    """
    focal_px = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    u = np.array([(p.x * width - width / 2.0) / focal_px for p in norm_landmarks])
    v = np.array([(p.y * height - height / 2.0) / focal_px for p in norm_landmarks])
    w = np.asarray(world_points, dtype=float)
    spread = math.sqrt(float(np.var(u) + np.var(v)))
    if spread < 0.01:
        return None
    ones, zeros = np.ones_like(u), np.zeros_like(u)
    a = np.concatenate([np.stack([ones, zeros, -u], axis=1), np.stack([zeros, ones, -v], axis=1)])
    b = np.concatenate([u * w[:, 2] - w[:, 0], v * w[:, 2] - w[:, 1]])
    t, *_ = np.linalg.lstsq(a, b, rcond=None)
    if t[2] <= 0:
        return None
    return tuple(float(c) for c in t + w[0])


def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # some backends ignore this
    return cap


def make_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    return sock


def poll_ack(sock):
    """How many hand rigs the receiver has (1 or more) once it has answered, else False. On Windows,
    sending to a port nobody listens on makes the next recvfrom raise ConnectionResetError; that
    means 'no receiver', not a sender fault."""
    answered = False
    try:
        while True:
            data, _ = sock.recvfrom(64)
            if data == ACK:
                answered = 1
            elif data.startswith(ACK + b" hands="):
                try:
                    answered = max(1, int(data[len(ACK) + 7:]))
                except ValueError:
                    answered = 1
    except (BlockingIOError, ConnectionResetError):
        return answered


class StageTimer:
    """Rolling per-stage averages so capture, inference and post-processing can be
    told apart when something is slow."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {}
        self.count = 0

    def add(self, **stages):
        for k, v in stages.items():
            self.sums[k] = self.sums.get(k, 0.0) + v
        self.count += 1

    def summary(self):
        if not self.count:
            return "no frames"
        return " ".join(f"{k}={v / self.count * 1000:.1f}ms" for k, v in self.sums.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=protocol.DEFAULT_ADDRESS[0])
    ap.add_argument("--port", type=int, default=protocol.DEFAULT_ADDRESS[1])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="camera horizontal field of view in degrees (for hand translation)")
    ap.add_argument("--min-cutoff", type=float, default=2.0,
                    help="One Euro: smoothing when still (Hz). Lower = smoother.")
    ap.add_argument("--beta", type=float, default=80.0,
                    help="One Euro: speed responsiveness. Higher = less lag when moving.")
    ap.add_argument("--d-cutoff", type=float, default=5.0,
                    help="One Euro: smoothing of the speed estimate (Hz). Higher = faster to react.")
    ap.add_argument("--max-speed", type=float, default=4.0,
                    help="outlier gate: reject landmark jumps faster than this (m/s)")
    ap.add_argument("--no-outlier-gate", action="store_true")
    ap.add_argument("--hands", choices=("auto", "1", "2"), default="auto",
                    help="how many hands to track. auto (default): as many as the Blender file has rigs "
                         "(it tells the sender when it answers). With two, your RIGHT hand drives the "
                         "rig on the right of the front view and your LEFT hand the one on the left")
    ap.add_argument("--detect-confidence", type=float, default=None,
                    help="MediaPipe hand detection confidence (default 0.7 for one hand, 0.5 for two)")
    ap.add_argument("--no-mirror", action="store_true", help="do not flip the image")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--record", metavar="FILE", help="also save the sent packets to FILE (.jsonl)")
    ap.add_argument("--dry-run", action="store_true",
                    help="track and print status but send nothing")
    args = ap.parse_args()

    address = (args.host, args.port)
    sid = uuid.uuid4().hex[:8]
    seq = 0
    hands = 2 if args.hands == "2" else 1  # auto starts with one and follows the receiver's answer
    landmarker = build_landmarker(num_hands=hands, detect_confidence=args.detect_confidence)
    channels = [Channel(args) for _ in range(hands)]
    assigner = SlotAssigner(slots=hands)
    receiver_hands = None
    recorder = Recorder(args.record) if args.record else None
    sock = make_socket()
    cap = open_camera(args.camera, args.width, args.height, args.fps)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")

    print(f"session {sid} -> {address[0]}:{address[1]}"
          f"{' (dry run, nothing sent)' if args.dry_run else ''}. ESC to quit.")
    print("UDP has no connection; 'receiver=' below reports whether Blender answered.")
    print(f"tracking {hands} hand{'s' if hands > 1 else ''}"
          + (" (auto: follows how many hand rigs Blender reports)" if args.hands == "auto" else ""))

    def set_hands(n):
        """Track n hands from now on (rebuilds MediaPipe's landmarker with num_hands=n)."""
        nonlocal landmarker, assigner, hands
        landmarker.close()
        landmarker = build_landmarker(num_hands=n, detect_confidence=args.detect_confidence)
        while len(channels) < n:
            channels.append(Channel(args))
        del channels[n:]
        assigner = SlotAssigner(slots=n)
        hands = n

    def send(data):
        if recorder:
            recorder.write(data)
        if not args.dry_run:
            sock.sendto(data, address)

    timer = StageTimer()
    last_ts_ms = -1
    receiver_ok = False
    last_status = time.monotonic()
    frames = sent_frames = rejected = seen_max = 0
    reject_note = ""
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                print("Camera read failed; stopping.")
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            ts_ms = max(int(now * 1000), last_ts_ms + 1)  # must be strictly increasing
            last_ts_ms = ts_ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
            t2 = time.perf_counter()
            detections = extract_hands(result)
            seq += 1
            frames += 1
            h, w = frame.shape[:2]
            seen_max = max(seen_max, len(detections))
            slot_of = assigner.assign([(d["norm"][0].x, d["norm"][0].y) for d in detections],
                                      [d["hand"] for d in detections])
            for slot, ch in enumerate(channels):
                if slot not in slot_of:
                    if ch.visible:
                        ch.lose()
                    send(protocol.encode_lost(sid, seq, slot))
                    continue
                det = detections[slot_of[slot]]
                points = det["points"]
                accepted, reason = (True, "") if ch.gate is None else ch.gate.check(points, now)
                if accepted:
                    root = estimate_root(det["norm"], points, w, h, args.hfov)
                    if root is not None:
                        root = ch.root_filt([root], now)[0]
                    sender_ms = (time.perf_counter() - t1) * 1000
                    send(protocol.encode_frame(sid, seq, time.time(), det["hand"], ch.filt(points, now),
                                               root, sender_ms, det["score"], slot))
                    sent_frames += 1
                    ch.visible = True
                else:
                    rejected += 1  # nothing is sent, so Blender holds the previous pose
                    reject_note = reason
                if not args.no_preview:
                    colour = ((0, 255, 0) if slot == 0 else (255, 160, 0)) if accepted else (0, 0, 255)
                    for p in det["norm"]:
                        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, colour, -1)
                    if hands > 1:
                        cv2.putText(frame, f"{SLOT_NAMES[slot]} hand", (int(det["norm"][0].x * w) - 30,
                                                                         int(det["norm"][0].y * h) + 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            if not args.no_preview:
                cv2.putText(frame, f"hands seen {len(detections)}, tracking {hands}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            t3 = time.perf_counter()
            timer.add(capture=t1 - t0, infer=t2 - t1, post=t3 - t2)

            ack = poll_ack(sock)
            if ack:
                receiver_ok = True
                if ack != receiver_hands:
                    receiver_hands = ack
                    if args.hands == "auto" and ack > hands:
                        print(f"Blender has {ack} hand rigs: now tracking {min(ack, 2)} hands")
                        set_hands(min(ack, 2))
                    elif args.hands == "1" and ack > 1:
                        print(f"Blender has {ack} hand rigs but --hands 1 tracks one; "
                              "drop --hands (auto) or use --hands 2 to drive both")
                    elif args.hands == "2" and ack < 2:
                        print("WARNING: Blender has only one hand rig, so the second hand is ignored; "
                              "open v5/untitled_two_hands_swapped.blend and run v5_run_receiver again")

            if now - last_status >= 2.0:
                elapsed = now - last_status
                last_status = now
                print(f"fps={frames / elapsed:.0f} sent={sent_frames} rejected={rejected}"
                      f"{' (' + reject_note + ')' if reject_note else ''} "
                      f"hand={'+'.join('yes' if c.visible else 'no' for c in channels)} "
                      f"seen<={seen_max} "
                      f"receiver={'acked' if receiver_ok else 'no reply yet'} | {timer.summary()}")
                frames = sent_frames = rejected = seen_max = 0
                reject_note = ""
                timer.reset()

            if not args.no_preview:
                cv2.imshow("Hand Tracking (ESC to quit)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        seq += 1
        try:
            send(protocol.encode_stop(sid, seq))
        except OSError:
            pass
        if recorder:
            recorder.close()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        sock.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
