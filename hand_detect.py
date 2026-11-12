"""Find the 21 hand landmarks in one picture with MediaPipe (the model the sender uses).

    python v5/hand_detect.py IMAGE.png OUT.json [--model hand_landmarker.task]

OUT.json: {"found": bool, "label": "Left"/"Right", "score": .., "landmarks": [[x, y, z] * 21]} with x and y
normalised to the picture (x right, y down). Used by fit_skeleton.py, which runs it as a separate
process because Blender's own Python has no MediaPipe. No camera is used.
"""
import argparse
import json
import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision

_PROJECT = Path(__file__).resolve().parent.parent
# from `python v5/setup_tools.py` (tools/models/); a copy in the project root still works
DEFAULT_MODEL = next((p for p in (_PROJECT / "tools" / "models" / "hand_landmarker.task", _PROJECT / "hand_landmarker.task")
                      if p.exists()), _PROJECT / "tools" / "models" / "hand_landmarker.task")


def detect(image_path, model=DEFAULT_MODEL, confidence=0.3):
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
    )
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp.Image.create_from_file(str(image_path)))
    if not result.hand_landmarks:
        return {"found": False}
    top = result.handedness[0][0]
    return {"found": True, "label": top.category_name, "score": top.score,
            "landmarks": [[p.x, p.y, p.z] for p in result.hand_landmarks[0]],
            "world": [[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--confidence", type=float, default=0.3)
    args = ap.parse_args()
    result = detect(args.image, args.model, args.confidence)
    Path(args.out).write_text(json.dumps(result))
    print("found" if result["found"] else "NO HAND FOUND",
          f"{result.get('label', '')} {result.get('score', 0):.2f}" if result["found"] else "")
    sys.exit(0 if result["found"] else 1)


if __name__ == "__main__":
    main()
